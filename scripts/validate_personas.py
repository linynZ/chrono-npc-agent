"""Check every persona file before it reaches a player.

Adding an NPC is meant to be a config change, which only holds if a bad config
fails loudly at the door instead of silently at runtime. Three of the four
failures below are ones I hit personally while writing the first three NPCs.

The plain-scalar colon in particular caught me three times: YAML reads

    - Reach for a sentry's words: gate, bar, line

as a mapping, not a string, and the resulting pydantic error points at a type
mismatch several layers from the actual mistake. That is exactly the kind of
thing a machine should be remembering instead of me.

Usage:
    python scripts/validate_personas.py
    python scripts/validate_personas.py --strict   # warnings become failures
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from chrono_agent.persona import NpcPersona  # noqa: E402
from chrono_agent.tools import ALL_TOOLS  # noqa: E402

LANGUAGES = ("zh", "en")
KNOWN_TOOLS = {tool.name for tool in ALL_TOOLS}

# The trap: a list item that reads as a sentence but ends up parsed as a mapping.
#
# `- id: stay_in_character` is a legitimate mapping and must not fire. The tell
# is whitespace before the colon — YAML mapping keys here are identifiers, while
# the trap is always a phrase. Items opening with a quote or block scalar are
# already safe.
# `\s` would match the newline and run the pattern into the next line, so the
# inner whitespace is spaces and tabs only.
COLON_TRAP = re.compile(r"^[ \t]*-[ \t]+(?![\"'>|&*])[^\n:]*[ \t][^\n:]*:(?:[ \t]|$)", re.M)


def check_file(path: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    raw = path.read_text(encoding="utf-8")
    for match in COLON_TRAP.finditer(raw):
        line = raw[: match.start()].count("\n") + 1
        errors.append(
            f"line {line}: unquoted ': ' in a list item — YAML will read this as "
            f"a mapping. Wrap it in double quotes.\n"
            f"           {match.group(0).strip()[:70]}"
        )
    if errors:
        return errors, warnings   # the file will not parse; stop here

    try:
        persona = NpcPersona.load(path.stem, path.parent)
    except Exception as exc:  # noqa: BLE001 - report, do not crash the run
        return [f"failed to load: {exc}"], warnings

    if persona.npc_id != path.stem:
        errors.append(f"npc_id {persona.npc_id!r} does not match filename {path.stem!r}")

    for language in LANGUAGES:
        if not persona.name.get(language):
            errors.append(f"name.{language} is missing")
        if not persona.persona.get(language, "").strip():
            errors.append(f"persona.{language} is empty")
        if not persona.voice.get(language):
            warnings.append(f"voice.{language} has no rules")
        if not persona.openers(language):
            warnings.append(f"suggested_questions.{language} is empty — the panel "
                            f"will show no openers")
        if not persona.last_resort(language):
            warnings.append(f"fallback.last_resort.{language} is missing — free "
                            f"conversation will fall back to a scripted line instead")

    if unknown := set(persona.tools) - KNOWN_TOOLS:
        errors.append(f"unknown tool(s): {sorted(unknown)}; known: {sorted(KNOWN_TOOLS)}")

    ids = [b.id for b in persona.boundaries]
    if duplicates := {i for i in ids if ids.count(i) > 1}:
        errors.append(f"duplicate boundary id(s): {sorted(duplicates)}")

    # Every NPC administers part of the same quiz, so this one is not optional.
    if "no_quiz_answers" not in ids:
        warnings.append("no boundary with id 'no_quiz_answers' — this NPC has "
                        "nothing telling it to withhold quiz answers")

    for boundary in persona.boundaries:
        for language in LANGUAGES:
            if not boundary.text(language):
                warnings.append(f"boundary {boundary.id!r} has no {language} text")

    if not persona.era:
        warnings.append("era is empty — the lore index will be built from nothing")

    return errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--dir", type=Path, default=ROOT / "characters")
    args = parser.parse_args()

    files = sorted(args.dir.glob("*.yaml"))
    if not files:
        print(f"no persona files in {args.dir}")
        return 1

    total_errors = total_warnings = 0
    for path in files:
        errors, warnings = check_file(path)
        total_errors += len(errors)
        total_warnings += len(warnings)

        mark = "FAIL" if errors else ("warn" if warnings else "ok")
        print(f"[{mark:>4}] {path.name}")
        for error in errors:
            print(f"        error: {error}")
        for warning in warnings:
            print(f"        warn:  {warning}")

    print(f"\n{len(files)} persona(s), {total_errors} error(s), {total_warnings} warning(s)")
    if total_errors:
        return 1
    return 1 if (args.strict and total_warnings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
