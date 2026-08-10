"""Verify the API key works before anything else depends on it.

Checks three things in order, because they fail differently and the error
messages for each are unhelpful when they arrive later in the stack:

  1. the key is present and accepted
  2. the configured model exists and answers
  3. tool calling works — the whole agent leans on it, and support varies by
     model far more than the docs suggest

Usage:
    python scripts/check_key.py
    python scripts/check_key.py --provider ollama
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from chrono_agent.config import Settings, build_provider  # noqa: E402
from chrono_agent.models import Message, Role  # noqa: E402
from chrono_agent.providers import ProviderError  # noqa: E402

PING_TOOL = {
    "type": "function",
    "function": {
        "name": "get_current_era",
        "description": "Return which era of the Chronicle River the traveler is in.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}


async def run(provider_name: str | None) -> int:
    settings = Settings.from_env()
    choice = provider_name or settings.provider

    print(f"provider : {choice}")
    if choice == "deepseek":
        key = settings.deepseek_api_key
        if not key:
            print("\n[FAIL] DEEPSEEK_API_KEY is empty.")
            print("       Copy .env.example to .env and paste your key in.")
            print("       Get one at https://platform.deepseek.com/api_keys")
            return 1
        print(f"key      : {key[:6]}...{key[-4:]} ({len(key)} chars)")
        print(f"model    : {settings.deepseek_model}")
        print(f"base_url : {settings.deepseek_base_url}")
    elif choice == "ollama":
        print(f"model    : {settings.ollama_model}")
        print(f"base_url : {settings.ollama_base_url}")

    provider = build_provider(settings, name=choice)

    try:
        print("\n[1/3] plain completion ...", end=" ", flush=True)
        completion = await provider.complete(
            [
                Message(
                    role=Role.USER,
                    content="用一句话回答：编年长河是什么？不超过二十字。",
                )
            ],
            max_tokens=64,
        )
        print(f"ok  ({completion.latency_ms:.0f} ms)")
        print(f"      reply: {completion.message.content.strip()[:60]}")
        print(f"      usage: {completion.usage.total_tokens} tokens "
              f"({completion.usage.cached_tokens} cached)")

        print("[2/3] model identity ...", end=" ", flush=True)
        print(f"ok  (served by {completion.model})")

        print("[3/3] tool calling ...", end=" ", flush=True)
        tool_completion = await provider.complete(
            [
                Message(
                    role=Role.USER,
                    content="Which era is the traveler in? Use the tool to find out.",
                )
            ],
            tools=[PING_TOOL],
            max_tokens=128,
        )
        calls = tool_completion.message.tool_calls
        if calls:
            print(f"ok  (requested {calls[0].name}, {tool_completion.latency_ms:.0f} ms)")
        else:
            print("NOT USED")
            print("      The model answered without calling the tool. That is not")
            print("      necessarily broken, but the agent depends on tool calls —")
            print("      re-run, and if it never calls, try a stronger model.")

    except ProviderError as exc:
        print("FAIL")
        print(f"\n[FAIL] {exc}")
        if "401" in str(exc) or "403" in str(exc):
            print("       That looks like a bad or revoked key.")
        elif "402" in str(exc) or "Insufficient" in str(exc):
            print("       That looks like an empty balance — top up on the console.")
        return 1
    finally:
        await provider.aclose()

    print("\nAll good. Try: python scripts/chat.py")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["deepseek", "ollama", "echo"])
    args = parser.parse_args()
    return asyncio.run(run(args.provider))


if __name__ == "__main__":
    raise SystemExit(main())
