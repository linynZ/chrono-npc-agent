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
import time
from typing import Any

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
        self.timeout_ms = timeout_ms
        self.max_tool_rounds = max_tool_rounds
        self.history_turns = history_turns

    async def reply(
        self,
        player_message: str,
        state: PlayerState,
        history: list[Message] | None = None,
        language: Language | None = None,
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
                started, tools_used, usage,
            )
        except ProviderError:
            return self._fallback(
                player_message, state, language, ReplySource.FALLBACK_ERROR,
                started, tools_used, usage,
            )

        output_verdict = inspect_npc_reply(text)
        if output_verdict.tripped:
            reply = self._fallback(
                player_message, state, language, ReplySource.FALLBACK_GUARDRAIL,
                started, tools_used, usage,
            )
            # Both sides, not just the output. Overwriting with the output flags
            # loses the record of what the player was attempting, which is the
            # more useful half when reviewing why a turn degraded.
            reply.guardrail_flags = input_verdict.flags + [
                flag for flag in output_verdict.flags if flag not in input_verdict.flags
            ]
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
                tools_used.append(call.name)
                result = registry.execute(context, call.name, call.arguments)
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

    def _fallback(
        self,
        player_message: str,
        state: PlayerState,
        language: Language,
        source: ReplySource,
        started: float,
        tools_used: list[str],
        usage: Usage,
    ) -> NpcReply:
        return NpcReply(
            text=self.fallback.pick(self.persona, state, player_message, language),
            source=source,
            latency_ms=(time.perf_counter() - started) * 1000,
            tool_calls_made=tools_used,
            usage=usage,
        )
