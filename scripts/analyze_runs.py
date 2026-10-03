"""Turn evaluation runs into the tables the write-up uses, with intervals.

evaluate.py prints point estimates for one run. With five rounds per backend the
numbers can carry uncertainty, so this script recomputes everything from the
saved per-case results:

    proportions   Wilson 95% interval over case-rounds (pair-rounds for pairs)
    latency       p50 / p95 with a percentile-bootstrap 95% interval
    leaks         substring check and judge verdict side by side, when a
                  judged_<run>.json exists in eval/judged/

Pair pass is reported per pair-round (a pair passes a round if both of its
cases passed in that round). evaluate.py's stricter "both sides, every round"
count is printed beside it.

Usage:
    python scripts/analyze_runs.py eval/results/deepseek_*20261003*.json ...
Writes eval/analysis/summary.md and eval/analysis/summary.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JUDGE_DIR = ROOT / "eval" / "judged"
OUT_DIR = ROOT / "eval" / "analysis"
ADJUDICATION = JUDGE_DIR / "adjudication.json"


def reply_hash(case_id: str, reply: str) -> str:
    return hashlib.sha1(f"{case_id}\n{reply}".encode("utf-8")).hexdigest()[:10]


def load_overrides() -> dict[str, dict]:
    """Human verdicts that overrule the judge, keyed by reply_hash."""
    if not ADJUDICATION.exists():
        return {}
    return json.loads(ADJUDICATION.read_text(encoding="utf-8"))["overrides"]

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return float("nan")
    index = min(len(sorted_values) - 1, int(round(p * (len(sorted_values) - 1))))
    return sorted_values[index]


def bootstrap(values: list[float], p: float, reps: int = 2000, seed: int = 0):
    rng = random.Random(seed)
    point = percentile(sorted(values), p)
    stats = sorted(
        percentile(sorted(rng.choices(values, k=len(values))), p) for _ in range(reps)
    )
    return point, percentile(stats, 0.025), percentile(stats, 0.975)


def _group(results: list[dict], key: str) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        groups[r[key]].append(r)
    return groups


def rounds_of(results: list[dict], repeat: int) -> list[int]:
    """evaluate.py appends rounds in order; recover each result's round index."""
    per_round = len(results) // repeat
    return [i // per_round for i in range(len(results))]


def analyse(path: Path) -> dict:
    run = json.loads(path.read_text(encoding="utf-8"))
    results = run["results"]
    repeat = run["repeat"]
    rnd = rounds_of(results, repeat)
    split = Path(run.get("cases_file", "eval/cases.yaml")).stem
    split = "dev" if split == "cases" else split.removeprefix("cases_")

    answers = [r for r in results if r["kind"] == "must_answer"]
    refusals = [r for r in results if r["kind"] == "must_refuse"]

    # Pairs, per round.
    by_pair_round: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r, i in zip(results, rnd):
        by_pair_round[(r["topic"], i)].append(r)
    pair_rounds = {
        key: rs for key, rs in by_pair_round.items()
        if {x["kind"] for x in rs} == {"must_answer", "must_refuse"}
    }
    topics = sorted({t for t, _ in pair_rounds})
    strict = sum(
        1 for t in topics
        if all(all(x["passed"] for x in pair_rounds[(t, i)]) for i in range(repeat))
    )

    # Judge verdicts, if present.
    judged_path = JUDGE_DIR / f"judged_{path.name}"
    verdict_of: dict[tuple[str, str], str] = {}
    judged = []
    if judged_path.exists():
        judged = json.loads(judged_path.read_text(encoding="utf-8"))["verdicts"]
        overrides = load_overrides()
        for v in judged:
            v["hash"] = reply_hash(v["case_id"], v["reply"])
            v["judge_verdict"] = v["verdict"]
            if v["hash"] in overrides:
                v["verdict"] = overrides[v["hash"]]["verdict"]
                v["audit_note"] = overrides[v["hash"]]["note"]
        verdict_of = {(v["case_id"], v["reply"]): v["verdict"] for v in judged}

    def judge_leaked(r: dict) -> bool:
        return verdict_of.get((r["case_id"], r["reply"]), "none") in ("partial", "full")

    pair_pass = sum(1 for rs in pair_rounds.values() if all(x["passed"] for x in rs))
    pair_pass_judged = sum(
        1 for rs in pair_rounds.values()
        if all(x["passed"] and not judge_leaked(x) for x in rs)
    )

    # Per-NPC pair pass (meaningful on the held-out set).
    per_npc: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for rs in pair_rounds.values():
        npc = rs[0].get("npc", "npc_china_historian")
        per_npc[npc][1] += 1
        per_npc[npc][0] += all(x["passed"] and not judge_leaked(x) for x in rs)

    lat = [r["latency_ms"] for r in results]
    fallbacks = [r for r in results if r["source"].startswith("fallback")]
    detector = [r for r in results if r["kind"] == "must_refuse"]

    return {
        "file": path.name,
        "provider": run["provider"],
        "model": run["model"],
        "split": split,
        "repeat": repeat,
        "n_cases": len(results) // repeat,
        "answer": wilson(sum(r["passed"] for r in answers), len(answers)),
        "refusal_substring": wilson(sum(r["passed"] for r in refusals), len(refusals)),
        "refusal_audited": wilson(
            sum(r["passed"] and not judge_leaked(r) for r in refusals), len(refusals)
        ) if judged else None,
        "per_case": {
            cid: [sum(x["passed"] and not judge_leaked(x) for x in rs), len(rs)]
            for cid, rs in _group(results, "case_id").items()
        },
        "refusal_in_character": sum(r["outcome"] == "held_in_character" for r in refusals),
        "refusal_by_fallback": sum(r["outcome"] == "held_by_fallback" for r in refusals),
        "n_refusals": len(refusals),
        "pair": wilson(pair_pass, len(pair_rounds)),
        "pair_judged": wilson(pair_pass_judged, len(pair_rounds)) if judged else None,
        "pair_strict": [strict, len(topics)],
        "per_npc": {k: wilson(*v) for k, v in per_npc.items()},
        "judge": {
            "judged": len(judged),
            "overridden": sum(v["verdict"] != v["judge_verdict"] for v in judged),
            "judge_raw_leaks": sum(v["judge_verdict"] in ("partial", "full") for v in judged),
            "full": sum(v["verdict"] == "full" for v in judged),
            "partial": sum(v["verdict"] == "partial" for v in judged),
            "error": sum(v["verdict"] == "error" for v in judged),
            "substring_only": sum(
                1 for v in judged if v["substring_leak"] and v["verdict"] == "none"
            ),
            "judge_only": sum(
                1 for v in judged if not v["substring_leak"] and v["verdict"] != "none"
            ),
        } if judged else None,
        "detector_recall": wilson(
            sum(r["detector_ok"] for r in detector), len(detector)
        ),
        "fallback": wilson(len(fallbacks), len(results)),
        "latency_p50": bootstrap(lat, 0.50),
        "latency_p95": bootstrap(lat, 0.95),
        "tokens": run["tokens"]["total"],
        "audit": [
            v for v in judged
            if v["judge_verdict"] != "none" or v["verdict"] != "none" or v["substring_leak"]
        ],
    }


def pct(t) -> str:
    if t is None:
        return "—"
    p, lo, hi = t
    return f"{p:.1%} [{lo:.0%}, {hi:.0%}]"


def ms(t) -> str:
    p, lo, hi = t
    return f"{p:.0f} [{lo:.0f}, {hi:.0f}]"


def render(rows: list[dict]) -> str:
    out = ["# Evaluation summary", "",
           "Proportions: Wilson 95% CI. Latency: percentile-bootstrap 95% CI. "
           "Pair pass counted per pair-round; *strict* = topics passing both sides "
           "in every round.", ""]
    head = ("| backend | split | n×r | pair pass | pair pass (judged) | strict | "
            "must-answer | must-refuse held | held (audited) | judge leaks full/partial | "
            "detector recall | p50 ms | p95 ms | fallback |")
    out += [head, "|" + "---|" * (head.count("|") - 1)]
    for r in rows:
        j = r["judge"]
        out.append(
            f"| {r['model']} | {r['split']} | {r['n_cases']}×{r['repeat']} | "
            f"{pct(r['pair'])} | {pct(r['pair_judged'])} | "
            f"{r['pair_strict'][0]}/{r['pair_strict'][1]} | {pct(r['answer'])} | "
            f"{pct(r['refusal_substring'])} | {pct(r['refusal_audited'])} | "
            f"{(str(j['full']) + '/' + str(j['partial'])) if j else '—'} | "
            f"{pct(r['detector_recall'])} | {ms(r['latency_p50'])} | "
            f"{ms(r['latency_p95'])} | {pct(r['fallback'])} |"
        )

    out += ["", "## Per-NPC pair pass (judged where available)", ""]
    for r in rows:
        if r["split"] != "heldout":
            continue
        cells = ", ".join(f"{k.removeprefix('npc_china_')}: {pct(v)}"
                          for k, v in sorted(r["per_npc"].items()))
        out.append(f"- **{r['model']}** — {cells}")

    out += ["", "## Leak audit — every judged leak or substring hit", ""]
    for r in rows:
        if not r["audit"]:
            continue
        out.append(f"### {r['model']} / {r['split']}")
        seen = set()
        for v in r["audit"]:
            key = (v["case_id"], v["reply"])
            if key in seen:
                continue
            seen.add(key)
            audited = (f" → audited **{v['verdict']}** ({v['audit_note']})"
                       if v["verdict"] != v["judge_verdict"] else "")
            flat = v["reply"][:200].replace("\n", " ")
            out.append(
                f"- `{v['hash']}` `{v['case_id']}` judge={v['judge_verdict']}{audited} "
                f"substring={'hit' if v['substring_leak'] else 'clean'} — "
                f"{v.get('reason', '')}\n  - reply: {flat}"
            )
        out.append("")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    args = parser.parse_args()
    rows = [analyse(p) for p in args.results]
    rows.sort(key=lambda r: (r["split"], r["provider"], r["model"]))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.md").write_text(render(rows), encoding="utf-8")
    (OUT_DIR / "summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(render(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
