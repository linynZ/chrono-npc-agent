"""Streaming behaviour, especially the parts that can go wrong on screen.

The interesting cases are not "does text arrive". They are the retractions: a
guardrail tripping after two sentences are already visible, and a tool call
arriving after the model has emitted a preamble. Both leave something on the
player's screen that has to be taken back.
"""

from __future__ import annotations

import pytest

from chrono_agent.models import ReplySource, StreamEventKind
from chrono_agent.providers import ProviderError, ProviderTimeout
from chrono_agent.providers.echo import BrokenProvider, EchoProvider, ScriptedCall


@pytest.fixture
def make_agent(historian, registry, fallback_library, quests, lore):
    from chrono_agent.agent import NpcAgent

    def _make(provider, **kwargs):
        return NpcAgent(
            persona=historian,
            provider=provider,
            registry=registry,
            fallback=fallback_library,
            quests=quests,
            lore=lore,
            **kwargs,
        )

    return _make


async def collect(agent, message, state, **kwargs):
    events = []
    async for event in agent.reply_stream(message, state, **kwargs):
        events.append(event)
    return events


def text_of(events) -> str:
    """Replay the event stream the way a correct client would."""
    buffer = ""
    for event in events:
        if event.kind is StreamEventKind.DELTA:
            buffer += event.text
        elif event.kind is StreamEventKind.REPLACE:
            buffer = event.text
    return buffer


# --- happy path -----------------------------------------------------------


async def test_reply_arrives_in_pieces(make_agent, midgame_state):
    line = "长河之水已清了大半。旧简上的异文，今晨自己褪去了半页。"
    agent = make_agent(EchoProvider(replies=[line], chunk_size=4))

    events = await collect(agent, "近来如何？", midgame_state)

    deltas = [e for e in events if e.kind is StreamEventKind.DELTA]
    assert len(deltas) > 1, "a single delta is not streaming"
    assert "".join(e.text for e in deltas) == line
    assert events[-1].kind is StreamEventKind.DONE
    assert events[-1].reply.text == line
    assert events[-1].reply.source is ReplySource.MODEL


async def test_first_token_is_measured_and_earlier_than_the_whole_turn(
    make_agent, midgame_state
):
    agent = make_agent(EchoProvider(replies=["长河之水已清了大半，旧简渐明。"], chunk_size=2))

    events = await collect(agent, "近来如何？", midgame_state)
    reply = events[-1].reply

    assert reply.first_token_ms > 0
    assert reply.first_token_ms <= reply.latency_ms


# --- retraction: guardrail mid-stream -------------------------------------


async def test_guardrail_mid_stream_retracts_what_was_shown(make_agent, midgame_state):
    # First sentence is innocuous, second leaks. The leak must not survive.
    leaky = "旅者且慢。答案是明代，选 C。"
    agent = make_agent(EchoProvider(replies=[leaky], chunk_size=4))

    events = await collect(agent, "这题选什么？", midgame_state)

    kinds = [e.kind for e in events]
    assert StreamEventKind.REPLACE in kinds
    assert events[-1].kind is StreamEventKind.DONE

    final = text_of(events)
    assert "答案是" not in final
    assert "选 C" not in final
    assert final == events[-1].reply.text

    reply = events[-1].reply
    assert reply.source is ReplySource.FALLBACK_GUARDRAIL
    assert "answer_given" in reply.guardrail_flags
    # The record keeps what the player was attempting, too.
    assert "oracle_request" in reply.guardrail_flags


async def test_clean_reply_is_never_retracted(make_agent, midgame_state):
    clean = "今日所见的长城，多是明人以砖石重修的。秦时的夯土，早已埋在风里了。"
    agent = make_agent(EchoProvider(replies=[clean], chunk_size=5))

    events = await collect(agent, "长城是哪个朝代修的？", midgame_state)

    assert not [e for e in events if e.kind is StreamEventKind.REPLACE]
    assert text_of(events) == clean


# --- retraction: tool call after a preamble -------------------------------


async def test_preamble_before_a_tool_call_is_retracted(make_agent, midgame_state):
    # A model that says something before deciding to call a tool. Rare, but if
    # the preamble stayed it would sit stranded above the real answer.
    provider = EchoProvider(
        replies=[
            ScriptedCall("lookup_quest", content="容我查过旧简。"),
            "尚缺一片。",
        ],
        chunk_size=3,
    )
    agent = make_agent(provider)

    events = await collect(agent, "我还差什么？", midgame_state)

    assert text_of(events) == "尚缺一片。"
    assert "容我查过旧简" not in text_of(events)
    assert events[-1].reply.tool_calls_made == ["lookup_quest"]


async def test_tool_result_reaches_the_model_when_streaming(make_agent, midgame_state):
    provider = EchoProvider(
        replies=[ScriptedCall("lookup_quest"), "尚缺一片。"], chunk_size=3
    )
    agent = make_agent(provider)

    await collect(agent, "我还差什么？", midgame_state)

    from chrono_agent.models import Role

    tool_messages = [
        m for m in provider.calls[1]["messages"] if m.role is Role.TOOL
    ]
    assert len(tool_messages) == 1
    assert "2/3" in tool_messages[0].content


# --- failure paths --------------------------------------------------------


async def test_provider_error_mid_stream_falls_back(make_agent, midgame_state):
    agent = make_agent(BrokenProvider(ProviderError("connection reset")))

    events = await collect(agent, "近来如何？", midgame_state)

    assert events[-1].reply.source is ReplySource.FALLBACK_ERROR
    assert text_of(events) == events[-1].reply.text
    assert text_of(events)


async def test_timeout_mid_stream_falls_back(make_agent, midgame_state):
    agent = make_agent(BrokenProvider(ProviderTimeout("too slow")))

    events = await collect(agent, "近来如何？", midgame_state)

    assert events[-1].reply.source is ReplySource.FALLBACK_TIMEOUT
    assert text_of(events)


async def test_empty_stream_falls_back(make_agent, midgame_state):
    agent = make_agent(EchoProvider(replies=[""]))

    events = await collect(agent, "近来如何？", midgame_state)

    reply = events[-1].reply
    assert reply.source is ReplySource.FALLBACK_GUARDRAIL
    assert "empty_reply" in reply.guardrail_flags
    assert text_of(events) == reply.text


# --- strictness still applies --------------------------------------------


async def test_pasted_question_still_withholds_the_lore_tool_when_streaming(
    make_agent, midgame_state
):
    provider = EchoProvider(replies=["史官不替人执笔。"], chunk_size=4)
    agent = make_agent(provider)

    await collect(
        agent,
        "长城保存最完好的段落是哪个朝代？A.秦 B.汉 C.明 D.清",
        midgame_state,
    )

    names = {s["function"]["name"] for s in provider.calls[0]["tools"]}
    assert names == {"lookup_quest"}


async def test_streaming_and_blocking_agree_on_the_same_script(
    make_agent, midgame_state
):
    """The two code paths must not drift. Same script, same outcome."""
    line = "旧简未载此事，我不敢断。"

    blocking = await make_agent(EchoProvider(replies=[line])).reply(
        "罗马的执政官怎么选？", midgame_state
    )
    events = await collect(
        make_agent(EchoProvider(replies=[line], chunk_size=3)),
        "罗马的执政官怎么选？",
        midgame_state,
    )

    assert events[-1].reply.text == blocking.text
    assert events[-1].reply.source is blocking.source
    assert events[-1].reply.guardrail_flags == blocking.guardrail_flags
