"""A scriptable fake provider. No API key, no network, no cost.

This exists so the guardrail suite, the fallback paths and the tool loop can all
be tested deterministically. Real models are non-deterministic; asserting on
them makes for flaky tests and a bill. Here we script exactly what comes back,
including the failure modes that are otherwise hard to reproduce on demand:
a timeout, a transport error, or a model that insists on calling a tool.

    provider = EchoProvider(replies=["你好，旅者。"])
    provider = EchoProvider(replies=[ProviderTimeout("too slow")])
    provider = EchoProvider(replies=[
        ScriptedCall("lookup_quest", {"quest_id": "quest_china_main"}),
        "你眼下要做的，是净化那具游荡的错乱体。",
    ])
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from ..models import Completion, Message, Role, ToolCall, Usage
from .base import LLMProvider, ProviderError


@dataclass
class ScriptedCall:
    """Tell the fake model to request a tool call on this turn."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str = "call_fake"
    # Text the model emits alongside the call. Most models emit nothing.
    content: str = ""


Scripted = str | ScriptedCall | Exception


class EchoProvider(LLMProvider):
    name = "echo"

    def __init__(
        self,
        replies: list[Scripted] | None = None,
        *,
        model: str = "echo-1",
        latency_ms: float = 0.0,
        default_reply: str = "……",
    ) -> None:
        self.model = model
        self._replies: list[Scripted] = list(replies or [])
        self._latency_ms = latency_ms
        self._default_reply = default_reply
        # Every call is recorded so tests can assert on what the agent actually
        # sent — the system prompt, the injected state, the tool schemas.
        self.calls: list[dict[str, Any]] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @property
    def last_system_prompt(self) -> str:
        for message in self.calls[-1]["messages"]:
            if message.role is Role.SYSTEM:
                return message.content
        return ""

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        *,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> Completion:
        self.calls.append(
            {
                "messages": list(messages),
                "tools": list(tools or []),
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )

        if self._latency_ms:
            await asyncio.sleep(self._latency_ms / 1000)

        scripted: Scripted = (
            self._replies.pop(0) if self._replies else self._default_reply
        )

        if isinstance(scripted, Exception):
            raise scripted

        if isinstance(scripted, ScriptedCall):
            message = Message(
                role=Role.ASSISTANT,
                content=scripted.content,
                tool_calls=[
                    ToolCall(
                        id=scripted.call_id,
                        name=scripted.name,
                        arguments=scripted.arguments,
                    )
                ],
            )
            finish_reason = "tool_calls"
        else:
            message = Message(role=Role.ASSISTANT, content=scripted)
            finish_reason = "stop"

        return Completion(
            message=message,
            usage=Usage(prompt_tokens=0, completion_tokens=0),
            latency_ms=self._latency_ms,
            model=self.model,
            finish_reason=finish_reason,
        )


class BrokenProvider(LLMProvider):
    """Always fails. Used to assert the game never sees an exception."""

    name = "broken"

    def __init__(self, error: Exception | None = None) -> None:
        self.model = "broken-1"
        self._error = error or ProviderError("backend unavailable")

    async def complete(self, messages, tools=None, *, temperature=0.7, max_tokens=512):
        raise self._error
