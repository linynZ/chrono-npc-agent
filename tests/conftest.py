from __future__ import annotations

import json

import pytest

from chrono_agent.config import DATA_DIR
from chrono_agent.fallback import FallbackLibrary
from chrono_agent.models import PlayerState, QuestStage
from chrono_agent.persona import NpcPersona
from chrono_agent.tools import ALL_TOOLS, ToolRegistry, build_lore_index


def _load(name: str):
    path = DATA_DIR / name
    if not path.is_file():
        pytest.skip(f"{path} missing — run scripts/extract_game_data.py")
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="session")
def quiz():
    return _load("quiz.json")


@pytest.fixture(scope="session")
def quests():
    return _load("quests.json")


@pytest.fixture(scope="session")
def lore(quiz):
    return build_lore_index(quiz, "china")


@pytest.fixture(scope="session")
def fallback_library():
    _load("npc_lines.json")  # skip early with a clear message if absent
    return FallbackLibrary.load()


@pytest.fixture(scope="session")
def historian():
    return NpcPersona.load("npc_china_historian")


@pytest.fixture
def registry():
    return ToolRegistry(ALL_TOOLS)


@pytest.fixture
def midgame_state():
    """Two fragments in, Rumor purged, still short of the gate."""
    return PlayerState(
        current_map="20_China",
        language="zh",
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
    )


@pytest.fixture
def fresh_state():
    """First meeting, nothing done."""
    return PlayerState(current_map="20_China", language="zh")
