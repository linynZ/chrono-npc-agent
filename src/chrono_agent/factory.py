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
from .config import DATA_DIR, LORE_INDEX_DIR, Settings, build_provider
from .fallback import FallbackLibrary
from .persona import NpcPersona
from .providers import LLMProvider
from .retrieval import OllamaEmbedder, load_retriever
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
    # `lore.json` is the answer-free export covering every era — preferred,
    # because answers that never enter the repo cannot leak from it. Building
    # from the quiz bank remains as the fallback for a checkout that has only
    # run the older extractor.
    path = DATA_DIR / "lore.json"
    if path.is_file():
        with path.open(encoding="utf-8") as fh:
            by_era = json.load(fh)
        return tuple(
            {
                "category": entry.get("category", ""),
                "zh": entry.get("zh", ""),
                "en": entry.get("en", ""),
                "_topic_zh": (entry.get("topic") or {}).get("zh", ""),
                "_topic_en": (entry.get("topic") or {}).get("en", ""),
            }
            for entry in by_era.get(era, [])
        )
    # Tuple so the cache can hold it; callers get a list back below.
    return tuple(build_lore_index(load_quiz(), era))


@lru_cache(maxsize=8)
def load_era_retriever(era: str):
    """Hybrid retriever for one era's lore, or None when the index is absent.

    Cached per era: the BM25 corpus and the chroma handle are shared across
    every agent of that era, which is what makes per-request construction cheap.
    """
    settings = Settings.from_env()
    embedder = OllamaEmbedder(
        base_url=settings.ollama_base_url, model=settings.embed_model
    )
    retriever = load_retriever(list(load_lore(era)), LORE_INDEX_DIR / era, embedder)
    if retriever is not None:
        # First embed after model load costs ~2s; pay it here, not on a player.
        embedder.warmup()
    return retriever


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
        retriever=overrides.pop("retriever", None) or load_era_retriever(persona.era),
        timeout_ms=overrides.pop("timeout_ms", settings.timeout_ms),
        **overrides,
    )


def available_npcs() -> list[str]:
    from .config import CHARACTERS_DIR

    return sorted(path.stem for path in CHARACTERS_DIR.glob("*.yaml"))
