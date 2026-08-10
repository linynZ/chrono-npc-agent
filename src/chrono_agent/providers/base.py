"""The provider seam.

Everything above this layer — persona, state injection, guardrails, fallback —
is model-agnostic. Swapping DeepSeek for a local Ollama model must not require
touching anything except which provider gets constructed.

Both backends speak the OpenAI wire format, so the abstraction is thin on
purpose: it converts our `Message` objects to and from that format and hands
back a `Completion`. Resisting the urge to invent a richer interface is what
keeps the two backends genuinely comparable.
"""

from __future__ import annotations

import abc
import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from ..models import Completion, Message, Role, ToolCall, Usage


@dataclass
class StreamDelta:
    """One piece of a streamed reply.

    Text arrives as `text` on successive deltas. The final delta carries
    `done=True` plus whatever the turn accumulated — tool calls (which stream as
    fragments and are only usable once complete) and usage, if the backend
    reports it.
    """

    text: str = ""
    done: bool = False
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    finish_reason: str = ""


class ProviderError(RuntimeError):
    """Any failure that should trigger fallback rather than crash the game."""


class ProviderTimeout(ProviderError):
    """The model did not answer inside the latency budget."""


class LLMProvider(abc.ABC):
    """A chat model that may call tools."""

    name: str = "base"
    model: str = ""

    @abc.abstractmethod
    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        *,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> Completion:
        """One round-trip. Raises ProviderTimeout / ProviderError on failure."""

    def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        *,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> AsyncIterator[StreamDelta]:
        """Same call, delivered incrementally.

        Providers that cannot stream should leave this alone; the default
        wraps `complete` and emits the whole reply as a single delta, so
        callers never need to branch on capability.
        """
        return self._stream_via_complete(
            messages, tools, temperature=temperature, max_tokens=max_tokens
        )

    async def _stream_via_complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
        *,
        temperature: float,
        max_tokens: int,
    ) -> AsyncIterator[StreamDelta]:
        completion = await self.complete(
            messages, tools, temperature=temperature, max_tokens=max_tokens
        )
        if completion.message.content:
            yield StreamDelta(text=completion.message.content)
        yield StreamDelta(
            done=True,
            tool_calls=completion.message.tool_calls,
            usage=completion.usage,
            finish_reason=completion.finish_reason,
        )

    async def aclose(self) -> None:
        """Release any held connections. Safe to call more than once."""


# --- OpenAI wire-format conversion ---------------------------------------
# Shared by every provider that speaks the OpenAI dialect (DeepSeek, Ollama,
# and most others), so the conversion lives here rather than being duplicated.


def message_to_wire(message: Message) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": message.role.value}

    if message.role is Role.TOOL:
        payload["content"] = message.content
        payload["tool_call_id"] = message.tool_call_id
        return payload

    payload["content"] = message.content
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments, ensure_ascii=False),
                },
            }
            for call in message.tool_calls
        ]
    return payload


def message_from_wire(raw: dict[str, Any]) -> Message:
    tool_calls = []
    for call in raw.get("tool_calls") or []:
        function = call.get("function", {})
        raw_args = function.get("arguments") or "{}"
        try:
            arguments = json.loads(raw_args)
        except json.JSONDecodeError:
            # A model can emit malformed JSON. Surface it as an empty call rather
            # than exploding — the tool layer reports the error back to the model,
            # which usually recovers on the next turn.
            arguments = {"__malformed__": raw_args}
        tool_calls.append(
            ToolCall(
                id=call.get("id", ""),
                name=function.get("name", ""),
                arguments=arguments if isinstance(arguments, dict) else {},
            )
        )

    return Message(
        role=Role.ASSISTANT,
        content=raw.get("content") or "",
        tool_calls=tool_calls,
    )
