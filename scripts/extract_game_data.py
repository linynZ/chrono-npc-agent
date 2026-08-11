"""Extract the ground-truth game data this agent has to stay faithful to.

Nothing in `data/` is hand-written. Everything is pulled straight out of the
ChronoTraveler Unity project so that the agent's persona, its knowledge tool and
its fallback lines cannot drift away from what the shipped game actually says.

Point it at a ChronoTraveler checkout with --game-root, or set
CHRONOTRAVELER_ROOT once and forget about it.

Usage:
    python scripts/extract_game_data.py --game-root path/to/ChronoTraveler
    CHRONOTRAVELER_ROOT=path/to/ChronoTraveler python scripts/extract_game_data.py

Outputs (all UTF-8 JSON, written to ./data):
    npc_lines.json    each NPC's pre-written dialogue, keyed by npcId, with the
                      four progress variants (base / mid / gate / post) grouped
    quiz.json         questions incl. correctIndex and explanation, by region
    quests.json       quest stages and objectives, parsed from the Unity assets
    lore.json         answer-free knowledge records (explanation + question stem
                      only), always exported for all five eras — see extract_lore

By default only the eras this project actually uses are exported. The full game
has five; Historian Mo needs one, and shipping the other four would publish a
few hundred questions no code here ever reads. Pass `--eras all` for everything.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ENV_VAR = "CHRONOTRAVELER_ROOT"

# Ids follow `npc_<era>_<role>` and `quest_<era>_main`, and quiz regions use the
# same era names, so one filter covers all three files.
ALL_ERAS = ("china", "egypt", "greece", "rome", "trade")
DEFAULT_ERAS = ("china",)


def default_game_root() -> Path | None:
    """Where to look for the Unity project, in order of preference."""
    if configured := os.getenv(ENV_VAR):
        return Path(configured)

    # A sibling checkout is the common layout for anyone cloning both repos.
    sibling = Path(__file__).resolve().parents[2] / "ChronoTraveler"
    return sibling if sibling.is_dir() else None

# NPC dialogue ids look like `npc_china_historian` plus progress variants
# `_mid` / `_gate` / `_post`. Anything else is the base entry.
VARIANT_SUFFIXES = ("_mid", "_gate", "_post")

OBJECTIVE_TYPES = {
    0: "talk",
    1: "defeat",
    2: "collect",
    3: "memory",
}


def _read_json(path: Path) -> object:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def split_npc_id(raw_id: str) -> tuple[str, str]:
    """`npc_china_historian_mid` -> (`npc_china_historian`, `mid`)."""
    for suffix in VARIANT_SUFFIXES:
        if raw_id.endswith(suffix):
            return raw_id[: -len(suffix)], suffix[1:]
    return raw_id, "base"


def era_of(identifier: str) -> str:
    """`npc_china_historian` / `quest_china_main` -> `china`."""
    parts = identifier.split("_")
    return parts[1] if len(parts) > 1 else ""


def extract_npc_lines(resources: Path, eras: set[str]) -> dict:
    entries = _read_json(resources / "DialogueData.json")["dialogues"]
    npcs: dict[str, dict] = {}
    for entry in entries:
        raw_id = entry["npcId"]
        if era_of(raw_id) not in eras:
            continue
        base_id, variant = split_npc_id(raw_id)
        npc = npcs.setdefault(
            base_id,
            {"npcId": base_id, "name": entry.get("npcName", {}), "variants": {}},
        )
        # The base entry carries the canonical display name; variants repeat it.
        if variant == "base":
            npc["name"] = entry.get("npcName", npc["name"])
        npc["variants"][variant] = [
            {
                "speaker": line.get("speaker", {}),
                "text": line.get("text", {}),
            }
            for line in entry.get("lines", [])
        ]
    return npcs


def extract_quiz(resources: Path, eras: set[str]) -> dict:
    quiz_sets = _read_json(resources / "QuizData.json")["quizSets"]
    by_region: dict[str, list] = {}
    for quiz_set in quiz_sets:
        for question in quiz_set.get("questions", []):
            region = question.get("region", "unknown")
            if region not in eras:
                continue
            by_region.setdefault(region, []).append(question)
    return by_region


def extract_lore(resources: Path) -> dict:
    """The answer-free knowledge base, for every era regardless of --eras.

    quiz.json carries `correctIndex`, so exporting it is publishing answers —
    which is why it stays filtered to the eras this project actually plays.
    lore.json strips the options and the answer at export time, before anything
    enters the repo: what was never written down cannot leak, from the tool or
    from the git history. That makes it safe to ship all five eras, which is
    what the retrieval index and any future non-china NPC run on.
    """
    by_region: dict[str, list] = {}
    for region, questions in extract_quiz(resources, set(ALL_ERAS)).items():
        for question in questions:
            explanation = question.get("explanation") or {}
            if not explanation:
                continue
            by_region.setdefault(region, []).append(
                {
                    "category": question.get("category", ""),
                    "topic": question.get("question", {}),
                    "zh": explanation.get("zh", ""),
                    "en": explanation.get("en", ""),
                }
            )
    return by_region


# --- Unity .asset parsing -------------------------------------------------
# The quest assets are Unity YAML with `!u!` tags and \uXXXX-escaped CJK.
# Pulling in a YAML dependency just to read four files is not worth it, and a
# generic loader still chokes on the tags — so we scrape the handful of fields
# we actually need.

_UNICODE_ESCAPE = re.compile(r"\\u([0-9A-Fa-f]{4})|\\x([0-9A-Fa-f]{2})")


def _unescape(value: str) -> str:
    value = value.strip()
    if value.startswith('"') and value.endswith('"'):
        value = value[1:-1]

    def replace(match: re.Match) -> str:
        group = match.group(1) or match.group(2)
        return chr(int(group, 16))

    return _UNICODE_ESCAPE.sub(replace, value)


def parse_quest_asset(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    quest: dict = {"stages": []}

    for field in ("questId", "title", "description"):
        match = re.search(rf"^\s*{field}:\s*(.+)$", text, re.MULTILINE)
        if match:
            quest[field] = _unescape(match.group(1))

    # Stages start at `- description:` under `stages:`; objectives are nested.
    stages_block = text.split("stages:", 1)
    if len(stages_block) < 2:
        return quest

    stage: dict | None = None
    objective: dict | None = None
    for line in stages_block[1].splitlines():
        stripped = line.strip()
        if stripped.startswith("- description:"):
            stage = {
                "description": _unescape(stripped.split(":", 1)[1]),
                "objectives": [],
            }
            quest["stages"].append(stage)
            objective = None
        elif stripped.startswith("- type:") and stage is not None:
            type_id = int(stripped.split(":", 1)[1].strip())
            objective = {"type": OBJECTIVE_TYPES.get(type_id, str(type_id))}
            stage["objectives"].append(objective)
        elif objective is not None and ":" in stripped:
            key, _, raw = stripped.partition(":")
            key = key.strip()
            if key in ("targetId", "label", "required"):
                value = _unescape(raw)
                objective[key] = int(value) if key == "required" else value

    return quest


def extract_quests(resources: Path, eras: set[str]) -> dict:
    quests = {}
    for path in sorted((resources / "Quests").glob("*.asset")):
        quest = parse_quest_asset(path)
        quest_id = quest.get("questId")
        if quest_id and era_of(quest_id) in eras:
            quests[quest_id] = quest
    return quests


def _write(out_dir: Path, name: str, payload: object) -> Path:
    path = out_dir / name
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-root", type=Path, default=default_game_root())
    parser.add_argument(
        "--out", type=Path, default=Path(__file__).resolve().parent.parent / "data"
    )
    parser.add_argument(
        "--eras",
        default=",".join(DEFAULT_ERAS),
        help=f"comma-separated, or 'all'. one of {ALL_ERAS}. default: china",
    )
    args = parser.parse_args()

    eras = (
        set(ALL_ERAS)
        if args.eras.strip().lower() == "all"
        else {e.strip() for e in args.eras.split(",") if e.strip()}
    )
    if unknown := eras - set(ALL_ERAS):
        print(f"[!] unknown era(s): {sorted(unknown)}", file=sys.stderr)
        print(f"    expected some of {ALL_ERAS}, or 'all'.", file=sys.stderr)
        return 1

    if args.game_root is None:
        print("[!] Could not locate the ChronoTraveler project.", file=sys.stderr)
        print(f"    Pass --game-root, or set {ENV_VAR}.", file=sys.stderr)
        return 1

    resources = args.game_root / "Assets" / "Resources"
    if not resources.is_dir():
        print(f"[!] No Assets/Resources under {args.game_root}", file=sys.stderr)
        print(f"    Pass --game-root, or set {ENV_VAR}.", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)

    npcs = extract_npc_lines(resources, eras)
    quiz = extract_quiz(resources, eras)
    quests = extract_quests(resources, eras)
    lore = extract_lore(resources)

    _write(args.out, "npc_lines.json", npcs)
    _write(args.out, "quiz.json", quiz)
    _write(args.out, "quests.json", quests)
    _write(args.out, "lore.json", lore)

    print(f"eras            {', '.join(sorted(eras))}")
    print(f"npc_lines.json  {len(npcs)} NPCs")
    for npc_id, npc in sorted(npcs.items()):
        variants = ",".join(sorted(npc["variants"]))
        print(f"                  {npc_id:<24} {variants}")
    print(f"quiz.json       {sum(len(v) for v in quiz.values())} questions "
          f"across {len(quiz)} regions: {', '.join(sorted(quiz))}")
    print(f"lore.json       {sum(len(v) for v in lore.values())} answer-free records "
          f"across {len(lore)} regions (always all eras)")
    print(f"quests.json     {len(quests)} quests")
    for quest_id, quest in sorted(quests.items()):
        objectives = sum(len(s["objectives"]) for s in quest["stages"])
        print(f"                  {quest_id:<24} "
              f"{len(quest['stages'])} stages / {objectives} objectives")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
