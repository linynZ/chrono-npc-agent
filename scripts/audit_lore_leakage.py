"""How often does a quiz explanation give the answer away?

The lore index is built from the quiz bank with `correctIndex` and the option
list stripped, which was supposed to make answer leaks structurally impossible.
Then the very first manual query proved otherwise:

    lookup_lore("长城")
    -> "今天所见保存最完好的长城多为明代(1368-1644)修筑的砖石长城。"

The matching question is "长城保存最完好的段落主要由哪个朝代修建？" with 明 as
the correct option. No field named `correctIndex` was involved. The explanation
*is* the answer, written out in prose.

This script measures the size of that hole so the fix can be aimed rather than
guessed at. For every question it asks:

  - does the explanation contain the correct option's text?
  - does it contain any of the wrong options' text? (a mention of a distractor
    makes the explanation less of a giveaway on its own)

Usage:
    python scripts/audit_lore_leakage.py
    python scripts/audit_lore_leakage.py --region china --show 10
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# The Windows console defaults to a legacy code page; without this the CJK
# samples come out as mojibake and the report is unreadable where it matters.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def contains(haystack: str, needle: str) -> bool:
    needle = needle.strip()
    if not needle:
        return False
    return needle.lower() in haystack.lower()


def audit_question(question: dict, language: str) -> dict:
    options = question.get("options") or []
    correct_index = question.get("correctIndex", -1)
    explanation = (question.get("explanation") or {}).get(language, "")

    if not options or not (0 <= correct_index < len(options)) or not explanation:
        return {"skipped": True}

    correct_text = (options[correct_index] or {}).get(language, "")
    wrong_texts = [
        (option or {}).get(language, "")
        for i, option in enumerate(options)
        if i != correct_index
    ]

    leaks_correct = contains(explanation, correct_text)
    mentions_wrong = sum(1 for text in wrong_texts if contains(explanation, text))

    return {
        "skipped": False,
        "leaks_correct": leaks_correct,
        "mentions_wrong": mentions_wrong,
        # The worst case: the explanation names the right answer and no other
        # option, so quoting it to a player is functionally the same as
        # telling them the answer.
        "unambiguous_leak": leaks_correct and mentions_wrong == 0,
        "category": question.get("category", ""),
        "question": (question.get("question") or {}).get(language, ""),
        "correct": correct_text,
        "explanation": explanation,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", default="all")
    parser.add_argument("--language", default="zh", choices=["zh", "en"])
    parser.add_argument("--show", type=int, default=5, help="sample leaks to print")
    parser.add_argument("--data", type=Path, default=DATA_DIR / "quiz.json")
    args = parser.parse_args()

    with args.data.open(encoding="utf-8") as fh:
        quiz = json.load(fh)

    regions = sorted(quiz) if args.region == "all" else [args.region]

    grand = Counter()
    samples: list[dict] = []

    print(f"Lore leakage audit — language={args.language}\n")
    header = f"{'region':<10}{'n':>5}{'leaks':>8}{'rate':>9}{'unambig':>9}{'rate':>9}"
    print(header)
    print("-" * len(header))

    for region in regions:
        counts = Counter()
        for question in quiz.get(region, []):
            result = audit_question(question, args.language)
            if result["skipped"]:
                counts["skipped"] += 1
                continue
            counts["total"] += 1
            if result["leaks_correct"]:
                counts["leaks"] += 1
                if result["unambiguous_leak"]:
                    counts["unambiguous"] += 1
                    if len(samples) < args.show:
                        samples.append({"region": region, **result})

        total = counts["total"] or 1
        print(
            f"{region:<10}{counts['total']:>5}{counts['leaks']:>8}"
            f"{counts['leaks'] / total:>8.1%}"
            f"{counts['unambiguous']:>9}{counts['unambiguous'] / total:>8.1%}"
        )
        grand.update(counts)

    total = grand["total"] or 1
    print("-" * len(header))
    print(
        f"{'ALL':<10}{grand['total']:>5}{grand['leaks']:>8}"
        f"{grand['leaks'] / total:>8.1%}"
        f"{grand['unambiguous']:>9}{grand['unambiguous'] / total:>8.1%}"
    )

    if samples:
        print(f"\n--- sample unambiguous leaks ({len(samples)}) ---")
        for sample in samples:
            print(f"\n[{sample['region']}/{sample['category']}]")
            print(f"  Q: {sample['question']}")
            print(f"  A: {sample['correct']}")
            print(f"  explanation: {sample['explanation']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
