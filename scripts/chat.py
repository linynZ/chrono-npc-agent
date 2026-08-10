"""Talk to an NPC from the terminal.

The fastest way to see whether any of this actually works. Every reply is
annotated with where it came from, how long it took, which tools ran and which
guardrails fired — the numbers that the evaluation harness will later collect in
bulk are visible here one turn at a time.

Usage:
    python scripts/chat.py
    python scripts/chat.py --provider ollama --lang en
    python scripts/chat.py --state gate       # simulate late-game progress

Commands inside the session:
    /state <fresh|mid|gate|post>   switch the simulated player progress
    /lang  <zh|en>                 switch language
    /prompt                        print the current system prompt
    /quit
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from chrono_agent.factory import available_npcs, build_agent  # noqa: E402
from chrono_agent.models import (  # noqa: E402
    Message,
    PlayerState,
    QuestStage,
    ReplySource,
    Role,
)
from chrono_agent.persona import build_system_prompt  # noqa: E402

# Four canned save states matching the game's own progress buckets, so the
# state-injection layer can be exercised without a running game.
STATES = {
    "fresh": PlayerState(current_map="20_China"),
    "mid": PlayerState(
        current_map="20_China",
        memory_progress=0.62,
        fragments_collected=2,
        purged_encounters=["china_rumor"],
        talked_npcs=["npc_china_historian"],
        correct_answers=7,
        total_answers=9,
        quest=QuestStage(
            quest_id="quest_china_main",
            stage_index=2,
            stage_description="寻回散落的记忆碎片。",
            objective_label="寻回记忆碎片",
            progress=2,
            required=3,
        ),
    ),
    "gate": PlayerState(
        current_map="20_China",
        memory_progress=1.0,
        fragments_collected=3,
        purged_encounters=["china_rumor"],
        talked_npcs=["npc_china_historian"],
        correct_answers=18,
        total_answers=20,
        quest=QuestStage(
            quest_id="quest_china_main",
            stage_index=3,
            stage_description="击败盘踞华夏的大错乱。",
            objective_label="击败大错乱",
            progress=0,
            required=1,
        ),
    ),
    "post": PlayerState(
        current_map="20_China",
        memory_progress=1.0,
        fragments_collected=3,
        purged_encounters=["china_rumor", "china_boss"],
        talked_npcs=["npc_china_historian"],
        earned_tokens=["20_China"],
        correct_answers=26,
        total_answers=30,
    ),
}

SOURCE_LABEL = {
    ReplySource.MODEL: "model",
    ReplySource.FALLBACK_TIMEOUT: "FALLBACK/timeout",
    ReplySource.FALLBACK_ERROR: "FALLBACK/error",
    ReplySource.FALLBACK_GUARDRAIL: "FALLBACK/guardrail",
}


async def run(args: argparse.Namespace) -> int:
    agent = build_agent(args.npc, timeout_ms=args.timeout)
    state = STATES[args.state].model_copy(update={"language": args.lang})
    history: list[Message] = []

    name = agent.persona.display_name(args.lang)
    print(f"── {name} ({args.npc}) ─ {agent.provider.name}/{agent.provider.model} ──")
    print(f"   state={args.state}  lang={args.lang}  budget={args.timeout}ms")
    print("   /state /lang /prompt /quit\n")

    while True:
        try:
            line = input("你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue
        if line in ("/quit", "/exit"):
            break
        if line.startswith("/state"):
            key = line.split(maxsplit=1)[-1]
            if key in STATES:
                state = STATES[key].model_copy(update={"language": state.language})
                history.clear()
                print(f"   → state={key} (history cleared)\n")
            else:
                print(f"   ? one of {list(STATES)}\n")
            continue
        if line.startswith("/lang"):
            key = line.split(maxsplit=1)[-1]
            if key in ("zh", "en"):
                state = state.model_copy(update={"language": key})
                print(f"   → lang={key}\n")
            else:
                print("   ? zh or en\n")
            continue
        if line == "/prompt":
            print("\n" + build_system_prompt(agent.persona, state) + "\n")
            continue

        reply = await agent.reply(line, state, history=history)

        print(f"\n{name} > {reply.text}")
        meta = [SOURCE_LABEL[reply.source], f"{reply.latency_ms:.0f}ms"]
        if reply.tool_calls_made:
            meta.append("tools=" + ",".join(reply.tool_calls_made))
        if reply.guardrail_flags:
            meta.append("flags=" + ",".join(reply.guardrail_flags))
        if reply.usage.total_tokens:
            meta.append(f"{reply.usage.total_tokens}tok")
        print(f"   [{'  '.join(meta)}]\n")

        history.append(Message(role=Role.USER, content=line))
        history.append(Message(role=Role.ASSISTANT, content=reply.text))

    await agent.provider.aclose()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npc", default="npc_china_historian", choices=available_npcs())
    parser.add_argument("--state", default="mid", choices=list(STATES))
    parser.add_argument("--lang", default="zh", choices=["zh", "en"])
    parser.add_argument("--timeout", type=int, default=8000, help="latency budget in ms")
    parser.add_argument("--provider", choices=["deepseek", "ollama", "echo"])
    args = parser.parse_args()

    if args.provider:
        import os

        os.environ["AGENT_PROVIDER"] = args.provider

    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
