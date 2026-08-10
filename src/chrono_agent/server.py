"""HTTP service. Unity talks to this; the web demo is the same API with a face.

Agents are cached per (npc, backend) because each one owns an httpx connection
pool — building a fresh agent per request would open a new pool per turn and
throw away the connection reuse that keeps latency down.

Conversation history is deliberately *not* kept here. The caller owns it and
sends it with each turn. A game already has a save file and a dialogue state; a
second, invisible copy living in the service is a synchronisation bug waiting
for someone to reload a save.
"""

from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .agent import NpcAgent
from .config import PROJECT_ROOT, Settings, build_provider
from .factory import available_npcs, build_agent
from .models import Language, Message, PlayerState, Role, StreamEventKind
from .providers import LLMProvider

WEB_DIR = PROJECT_ROOT / "web"

_agents: dict[tuple[str, str], NpcAgent] = {}
_providers: dict[str, LLMProvider] = {}


def _get_agent(npc_id: str, backend: str) -> NpcAgent:
    key = (npc_id, backend)
    if key not in _agents:
        settings = Settings.from_env()
        if backend not in _providers:
            _providers[backend] = build_provider(settings, name=backend)
        _agents[key] = build_agent(
            npc_id,
            provider=_providers[backend],
            settings=settings,
            # Local models are slower by nature; judging them against the cloud
            # budget would report model latency as infrastructure failure.
            timeout_ms=30000 if backend == "ollama" else settings.timeout_ms,
        )
    return _agents[key]


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    for provider in _providers.values():
        await provider.aclose()


app = FastAPI(
    title="chrono-npc-agent",
    description="Runtime LLM NPC agent for ChronoTraveler.",
    version="0.1.0",
    lifespan=lifespan,
)


class Turn(BaseModel):
    role: str
    text: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    npc_id: str = "npc_china_historian"
    state: PlayerState = Field(default_factory=PlayerState)
    history: list[Turn] = Field(default_factory=list)
    language: Language | None = None
    backend: str = "deepseek"


class ChatResponse(BaseModel):
    text: str
    speaker: str
    source: str
    latency_ms: float
    server_ms: float
    tool_calls: list[str]
    guardrail_flags: list[str]
    tokens: int
    backend: str
    model: str


@app.get("/api/health")
async def health() -> dict:
    settings = Settings.from_env()
    return {
        "ok": True,
        "npcs": available_npcs(),
        "default_backend": settings.provider,
        "backends": ["deepseek", "ollama", "echo"],
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    if request.npc_id not in available_npcs():
        raise HTTPException(404, f"unknown npc: {request.npc_id}")
    if request.backend not in ("deepseek", "ollama", "echo"):
        raise HTTPException(400, f"unknown backend: {request.backend}")

    try:
        agent = _get_agent(request.npc_id, request.backend)
    except ValueError as exc:
        # Missing API key is the common case here, and the message says so.
        raise HTTPException(503, str(exc)) from exc

    language = request.language or request.state.language
    history = [
        Message(
            role=Role.USER if turn.role == "user" else Role.ASSISTANT,
            content=turn.text,
        )
        for turn in request.history
    ]

    started = time.perf_counter()
    reply = await agent.reply(
        request.message, request.state, history=history, language=language
    )
    server_ms = (time.perf_counter() - started) * 1000

    return ChatResponse(
        text=reply.text,
        speaker=agent.persona.display_name(language),
        source=reply.source.value,
        latency_ms=round(reply.latency_ms, 1),
        server_ms=round(server_ms, 1),
        tool_calls=reply.tool_calls_made,
        guardrail_flags=reply.guardrail_flags,
        tokens=reply.usage.total_tokens,
        backend=request.backend,
        model=agent.provider.model,
    )


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """Same turn as /api/chat, delivered as server-sent events.

    Event kinds are `delta`, `replace` and `done`. **A client that ignores
    `replace` is broken**: it is how a guardrail retracts text that is already
    on screen, and dropping it leaves a leaked answer in front of the player.
    """
    if request.npc_id not in available_npcs():
        raise HTTPException(404, f"unknown npc: {request.npc_id}")
    if request.backend not in ("deepseek", "ollama", "echo"):
        raise HTTPException(400, f"unknown backend: {request.backend}")

    try:
        agent = _get_agent(request.npc_id, request.backend)
    except ValueError as exc:
        raise HTTPException(503, str(exc)) from exc

    language = request.language or request.state.language
    history = [
        Message(
            role=Role.USER if turn.role == "user" else Role.ASSISTANT,
            content=turn.text,
        )
        for turn in request.history
    ]
    speaker = agent.persona.display_name(language)

    async def events() -> AsyncIterator[str]:
        started = time.perf_counter()
        try:
            async for event in agent.reply_stream(
                request.message, request.state, history=history, language=language
            ):
                frame: dict = {"kind": event.kind.value}
                if event.kind is StreamEventKind.DONE and event.reply:
                    reply = event.reply
                    frame["reply"] = {
                        "text": reply.text,
                        "speaker": speaker,
                        "source": reply.source.value,
                        "latency_ms": round(reply.latency_ms, 1),
                        "first_token_ms": round(reply.first_token_ms, 1),
                        "server_ms": round((time.perf_counter() - started) * 1000, 1),
                        "tool_calls": reply.tool_calls_made,
                        "guardrail_flags": reply.guardrail_flags,
                        "tokens": reply.usage.total_tokens,
                        "backend": request.backend,
                        "model": agent.provider.model,
                    }
                else:
                    frame["text"] = event.text
                yield _sse(frame)
        except Exception as exc:  # noqa: BLE001 - the stream is already open, so
            # the only way to report a late failure is inside it.
            yield _sse({"kind": "error", "text": str(exc)})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # nginx and friends buffer by default, which would defeat the point.
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/")
async def index() -> FileResponse:
    page = WEB_DIR / "index.html"
    if not page.is_file():
        raise HTTPException(404, "web/index.html not found")
    return FileResponse(page)


if WEB_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
