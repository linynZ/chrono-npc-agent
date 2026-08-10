"""Assemble a ready-to-use agent from files on disk.

Every entry point — the CLI, the HTTP service, the evaluation harness — needs
the same wiring, and it is the kind of wiring that drifts if each one does it
itself. The loaded data is cached because the quiz bank is a few hundred KB and
re-reading it per request would be silly.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from .agent import NpcAgent
from .config import DATA_DIR, Settings, build_provider
from .fallback import FallbackLibrary
from .persona import NpcPersona
from .providers import LLMProvider
from .tools import ALL_TOOLS, ToolRegistry, build_lore_index


class MissingGameData(FileNotFoundError):
    pass


def _load(name: str) -> Any:
    path = DATA_DIR / name
    if not path.is_file():
        raise MissingGameData(
            f"{path} is missing. Run: python scripts/extract_game_data.py"
        )
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def load_quiz() -> dict[str, list[dict]]:
    return _load("quiz.json")


@lru_cache(maxsize=1)
def load_quests() -> dict[str, Any]:
    return _load("quests.json")


@lru_cache(maxsize=1)
def load_fallback_library() -> FallbackLibrary:
    _load("npc_lines.json")
    return FallbackLibrary.load()


@lru_cache(maxsize=8)
def load_lore(era: str) -> tuple[dict, ...]:
    # Tuple so the cache can hold it; callers get a list back below.
    return tuple(build_lore_index(load_quiz(), era))


@lru_cache(maxsize=32)
def load_persona(npc_id: str) -> NpcPersona:
    return NpcPersona.load(npc_id)


def build_agent(
    npc_id: str,
    provider: LLMProvider | None = None,
    settings: Settings | None = None,
    **overrides: Any,
) -> NpcAgent:
    settings = settings or Settings.from_env()
    persona = load_persona(npc_id)

    return NpcAgent(
        persona=persona,
        provider=provider or build_provider(settings),
        registry=ToolRegistry(ALL_TOOLS),
        fallback=load_fallback_library(),
        quests=load_quests(),
        lore=list(load_lore(persona.era)),
        timeout_ms=overrides.pop("timeout_ms", settings.timeout_ms),
        **overrides,
    )


def available_npcs() -> list[str]:
    from .config import CHARACTERS_DIR

    return sorted(path.stem for path in CHARACTERS_DIR.glob("*.yaml"))
