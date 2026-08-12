"""The agent loop: prompt, tools, guardrails, fallback.

Ordering matters here and is worth stating explicitly.

The latency budget wraps the *whole* interaction, not each model call. A turn
that calls a tool costs two round trips, and the player standing in front of the
NPC is waiting for the sum. Budgeting per-call would let a two-tool turn quietly
take three times the limit while every individual call passed.

Guardrails run on both ends. Input flags do not block; they append an
in-character instruction so the NPC refuses as Mo rather than as a validator.
The output check is the backstop for when the model refuses badly or not at all,
and there the only remedy is to drop the generated line and serve a written one:
a reply that leaked an answer cannot be repaired by asking again inside the
player's patience.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any, AsyncIterator

from .fallback import FallbackLibrary
from .guardrails.rules import (
    inspect_npc_reply,
    inspect_player_message,
    steering_note,
)
from .models import (
    Language,
    Message,
    NpcReply,
    PlayerState,
    ReplySource,
    Role,
    StreamEvent,
    StreamEventKind,
    Usage,
)
from .persona import NpcPersona, build_system_prompt
from .providers import LLMProvider, ProviderError, ProviderTimeout
from .tools import ToolContext, ToolRegistry

# When the player has pasted a live question, the strictness goes up a level.
# The lore tool is withheld rather than the model being asked to use it wisely —
# an NPC holding the relevant historical record will find a way to allude to it,
# and Finding 01 showed 66.8% of those records name the answer outright. Taking
# the record away is a smaller change than trying to police what he does with it.
STRICT_FLAGS = frozenset({"pasted_question"})
WITHHELD_UNDER_STRICT = frozenset({"lookup_lore"})

# Guardrails need whole thoughts, not fragments — running them per-token would
# fire on half a word. Checking at sentence boundaries bounds the damage to one
# sentence while keeping the check cheap.
SENTENCE_END = re.compile(r"[。！？；…\n]|[.!?](?:\s|$)")


def _merge_flags(incoming: list[str], outgoing: list[str]) -> list[str]:
    """Keep both sides of the record — what was attempted and what was said."""
    return incoming + [flag for flag in outgoing if flag not in incoming]


def _tool_label(name: str, result: str) -> str:
    """A tool may report which rung of its ladder answered (`lookup_lore`
    returns `retrieval: vector | bm25 | substring`). Folding the rung into the
    recorded call name surfaces a degraded index in the diagnostics strip,
    instead of a server log nobody watches."""
    try:
        rung = json.loads(result).get("retrieval")
    except (ValueError, AttributeError):
        return name
    return f"{name}·{rung}" if isinstance(rung, str) and rung else name


class NpcAgent:
    def __init__(
        self,
        persona: NpcPersona,
        provider: LLMProvider,
        registry: ToolRegistry,
        fallback: FallbackLibrary,
        *,
        quests: dict[str, Any] | None = None,
        lore: list[dict[str, Any]] | None = None,
        retriever: Any = None,
        timeout_ms: int = 4000,
        max_tool_rounds: int = 3,
        history_turns: int = 6,
    ) -> None:
        self.persona = persona
        self.provider = provider
        self.registry = registry.subset(persona.tools)
        self.fallback = fallback
        self.quests = quests or {}
        self.lore = lore or []
        self.retriever = retriever
        self.timeout_ms = timeout_ms
        self.max_tool_rounds = max_tool_rounds
        self.history_turns = history_turns

    async def reply(
        self,
        player_message: str,
        state: PlayerState,
        history: list[Message] | None = None,
        language: Language | None = None,
        free_form: bool = False,
    ) -> NpcReply:
        language = language or state.language
        started = time.perf_counter()

        input_verdict = inspect_player_message(player_message)

        system_prompt = build_system_prompt(self.persona, state, language)
        if note := steering_note(input_verdict, language):
            system_prompt = f"{system_prompt}\n\n{note}"

        messages: list[Message] = [Message(role=Role.SYSTEM, content=system_prompt)]
        if history:
            messages.extend(history[-self.history_turns * 2 :])
        messages.append(Message(role=Role.USER, content=player_message))

        context = ToolContext(
            state=state,
            language=language,
            quests=self.quests,
            lore=self.lore,
            npc_id=self.persona.npc_id,
            retriever=self.retriever,
        )

        registry = self.registry
        if STRICT_FLAGS.intersection(input_verdict.flags):
            registry = registry.subset(
                [name for name in self.persona.tools if name not in WITHHELD_UNDER_STRICT]
            )

        tools_used: list[str] = []
        usage = Usage()

        try:
            text = await asyncio.wait_for(
                self._run_loop(messages, context, tools_used, usage, registry),
                timeout=self.timeout_ms / 1000,
            )
        except (asyncio.TimeoutError, ProviderTimeout):
            return self._fallback(
                player_message, state, language, ReplySource.FALLBACK_TIMEOUT,
                started, tools_used, usage, free_form,
            )
        except ProviderError:
            return self._fallback(
                player_message, state, language, ReplySource.FALLBACK_ERROR,
                started, tools_used, usage, free_form,
            )

        output_verdict = inspect_npc_reply(text)
        if output_verdict.tripped:
            reply = self._fallback(
                player_message, state, language, ReplySource.FALLBACK_GUARDRAIL,
                started, tools_used, usage, free_form,
            )
            # Both sides, not just the output. Overwriting with the output flags
            # loses the record of what the player was attempting, which is the
            # more useful half when reviewing why a turn degraded.
            reply.guardrail_flags = _merge_flags(
                input_verdict.flags, output_verdict.flags
            )
            return reply

        return NpcReply(
            text=text.strip(),
            source=ReplySource.MODEL,
            latency_ms=(time.perf_counter() - started) * 1000,
            tool_calls_made=tools_used,
            guardrail_flags=input_verdict.flags,
            usage=usage,
        )

    async def _run_loop(
        self,
        messages: list[Message],
        context: ToolContext,
        tools_used: list[str],
        usage: Usage,
        registry: ToolRegistry | None = None,
    ) -> str:
        """Call the model, service any tool calls, return the final text."""
        registry = registry if registry is not None else self.registry
        schemas = registry.schemas() or None

        for _ in range(self.max_tool_rounds + 1):
            completion = await self.provider.complete(messages, schemas)

            usage.prompt_tokens += completion.usage.prompt_tokens
            usage.completion_tokens += completion.usage.completion_tokens
            usage.cached_tokens += completion.usage.cached_tokens

            reply = completion.message
            if not reply.tool_calls:
                return reply.content

            messages.append(reply)
            for call in reply.tool_calls:
                result = registry.execute(context, call.name, call.arguments)
                tools_used.append(_tool_label(call.name, result))
                messages.append(
                    Message(
                        role=Role.TOOL,
                        content=result,
                        tool_call_id=call.id,
                        name=call.name,
                    )
                )

        # Out of rounds with the model still reaching for tools. Rather than
        # loop forever, ask once more with tools withheld so it has to answer.
        final = await self.provider.complete(messages, None)
        usage.prompt_tokens += final.usage.prompt_tokens
        usage.completion_tokens += final.usage.completion_tokens
        return final.message.content

    async def reply_stream(
        self,
        player_message: str,
        state: PlayerState,
        history: list[Message] | None = None,
        language: Language | None = None,
        free_form: bool = False,
    ) -> AsyncIterator[StreamEvent]:
        """Same turn, delivered as it is produced.

        The tension worth naming: streaming and output guardrails want opposite
        things. A guardrail wants the finished reply; the player wants the first
        words now. Buffering to satisfy the guardrail throws away the entire
        benefit, so instead the check runs at each sentence boundary and the
        contract carries a REPLACE event for when a later sentence trips one.
        Exposure is bounded to a sentence rather than eliminated — which is the
        honest trade, and the reason REPLACE is not optional for clients.
        """
        language = language or state.language
        started = time.perf_counter()
        first_token_ms = 0.0

        input_verdict = inspect_player_message(player_message)
        system_prompt = build_system_prompt(self.persona, state, language)
        if note := steering_note(input_verdict, language):
            system_prompt = f"{system_prompt}\n\n{note}"

        messages: list[Message] = [Message(role=Role.SYSTEM, content=system_prompt)]
        if history:
            messages.extend(history[-self.history_turns * 2 :])
        messages.append(Message(role=Role.USER, content=player_message))

        context = ToolContext(
            state=state,
            language=language,
            quests=self.quests,
            lore=self.lore,
            npc_id=self.persona.npc_id,
            retriever=self.retriever,
        )

        registry = self.registry
        if STRICT_FLAGS.intersection(input_verdict.flags):
            registry = registry.subset(
                [n for n in self.persona.tools if n not in WITHHELD_UNDER_STRICT]
            )

        tools_used: list[str] = []
        usage = Usage()
        emitted = ""

        def elapsed() -> float:
            return (time.perf_counter() - started) * 1000

        def degrade(source: ReplySource, flags: list[str]) -> NpcReply:
            reply = self._fallback(
                player_message, state, language, source, started, tools_used,
                usage, free_form
            )
            reply.first_token_ms = first_token_ms
            reply.guardrail_flags = flags
            return reply

        try:
            for _ in range(self.max_tool_rounds + 1):
                schemas = registry.schemas() or None
                round_text = ""
                tool_calls = []

                async for delta in self.provider.stream(messages, schemas):
                    if delta.done:
                        tool_calls = delta.tool_calls
                        usage.prompt_tokens += delta.usage.prompt_tokens
                        usage.completion_tokens += delta.usage.completion_tokens
                        usage.cached_tokens += delta.usage.cached_tokens
                        break

                    if not delta.text:
                        continue
                    if first_token_ms == 0.0:
                        first_token_ms = elapsed()

                    round_text += delta.text
                    emitted += delta.text
                    yield StreamEvent(kind=StreamEventKind.DELTA, text=delta.text)

                    # Only check on a completed sentence — mid-word text trips
                    # nothing useful and costs a regex pass per token.
                    if SENTENCE_END.search(delta.text):
                        verdict = inspect_npc_reply(emitted)
                        if verdict.tripped and "empty_reply" not in verdict.flags:
                            reply = degrade(
                                ReplySource.FALLBACK_GUARDRAIL,
                                _merge_flags(input_verdict.flags, verdict.flags),
                            )
                            yield StreamEvent(
                                kind=StreamEventKind.REPLACE, text=reply.text
                            )
                            yield StreamEvent(kind=StreamEventKind.DONE, reply=reply)
                            return

                if not tool_calls:
                    break

                # Optimistic streaming: text went out before we knew a tool was
                # coming. In practice models emit nothing alongside a tool call,
                # so this retraction is rare — but when it fires, leaving the
                # preamble on screen would strand it in front of the real reply.
                if emitted:
                    yield StreamEvent(kind=StreamEventKind.REPLACE, text="")
                    emitted = ""

                messages.append(
                    Message(
                        role=Role.ASSISTANT, content=round_text, tool_calls=tool_calls
                    )
                )
                for call in tool_calls:
                    result = registry.execute(context, call.name, call.arguments)
                    tools_used.append(_tool_label(call.name, result))
                    messages.append(
                        Message(
                            role=Role.TOOL,
                            content=result,
                            tool_call_id=call.id,
                            name=call.name,
                        )
                    )
            else:
                # Rounds exhausted with the model still reaching for tools. Ask
                # once more with none offered so it has to answer in words.
                if emitted:
                    yield StreamEvent(kind=StreamEventKind.REPLACE, text="")
                    emitted = ""
                async for delta in self.provider.stream(messages, None):
                    if delta.done:
                        usage.prompt_tokens += delta.usage.prompt_tokens
                        usage.completion_tokens += delta.usage.completion_tokens
                        break
                    if delta.text:
                        if first_token_ms == 0.0:
                            first_token_ms = elapsed()
                        emitted += delta.text
                        yield StreamEvent(kind=StreamEventKind.DELTA, text=delta.text)

        except ProviderTimeout:
            reply = degrade(ReplySource.FALLBACK_TIMEOUT, input_verdict.flags)
            yield StreamEvent(kind=StreamEventKind.REPLACE, text=reply.text)
            yield StreamEvent(kind=StreamEventKind.DONE, reply=reply)
            return
        except ProviderError:
            reply = degrade(ReplySource.FALLBACK_ERROR, input_verdict.flags)
            yield StreamEvent(kind=StreamEventKind.REPLACE, text=reply.text)
            yield StreamEvent(kind=StreamEventKind.DONE, reply=reply)
            return

        # Final pass over the whole reply. The per-sentence checks can miss a
        # rule that only matches across a boundary, and an empty reply is only
        # knowable once nothing more is coming.
        verdict = inspect_npc_reply(emitted)
        if verdict.tripped:
            reply = degrade(
                ReplySource.FALLBACK_GUARDRAIL,
                _merge_flags(input_verdict.flags, verdict.flags),
            )
            yield StreamEvent(kind=StreamEventKind.REPLACE, text=reply.text)
            yield StreamEvent(kind=StreamEventKind.DONE, reply=reply)
            return

        yield StreamEvent(
            kind=StreamEventKind.DONE,
            reply=NpcReply(
                text=emitted.strip(),
                source=ReplySource.MODEL,
                latency_ms=elapsed(),
                first_token_ms=first_token_ms,
                tool_calls_made=tools_used,
                guardrail_flags=input_verdict.flags,
                usage=usage,
            ),
        )

    def _fallback(
        self,
        player_message: str,
        state: PlayerState,
        language: Language,
        source: ReplySource,
        started: float,
        tools_used: list[str],
        usage: Usage,
        free_form: bool = False,
    ) -> NpcReply:
        # In scripted dialogue a written line is the right substitute. In free
        # conversation it is not: the player just asked something specific, and
        # answering with an unrelated story beat reads worse than a silence.
        text = ""
        if free_form:
            text = self.persona.last_resort(language)
        if not text:
            text = self.fallback.pick(self.persona, state, player_message, language)

        return NpcReply(
            text=text,
            source=source,
            latency_ms=(time.perf_counter() - started) * 1000,
            tool_calls_made=tools_used,
            usage=usage,
        )
