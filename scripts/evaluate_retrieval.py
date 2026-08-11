"""Measure the retrieval ladder: substring vs BM25 vs vector vs hybrid.

Recall@k on the paraphrase query set (see build_retrieval_eval.py): a query is
counted as recalled when the record it was derived from appears in the top k.
That gold standard slightly *understates* every method equally — a few records
cover the same topic, and returning a twin of the gold record counts as a miss
— so the numbers are conservative and the deltas between methods are the point.

The verbatim stems are also run as a sanity ceiling: substring should look
good there, because the stem is stored inside the record. If it does not, the
harness itself is broken.

Usage:
    python scripts/evaluate_retrieval.py                # needs index + Ollama
    python scripts/evaluate_retrieval.py --eras china --k 1,3,5
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chrono_agent.config import PROJECT_ROOT  # noqa: E402
from chrono_agent.factory import load_era_retriever, load_lore  # noqa: E402
from chrono_agent.tools.game_tools import _score  # noqa: E402

QUERIES_PATH = PROJECT_ROOT / "eval" / "retrieval_queries.json"
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"

METHODS = ("substring", "bm25", "vector", "hybrid")


def substring_rank(lore: list[dict], query: str, k: int) -> list[int]:
    """The original tool search, verbatim: same term split, same scoring."""
    terms = [t for t in query.split() if t]
    if len(terms) <= 1 and any("一" <= ch <= "鿿" for ch in query):
        terms = [ch for ch in query if "一" <= ch <= "鿿"]
    scored = [
        (score, index)
        for index, entry in enumerate(lore)
        if (score := _score(entry, terms, "zh")) > 0
    ]
    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    return [index for _, index in scored[:k]]


def evaluate(
    eras: list[str], ks: list[int], query_sets: dict
) -> tuple[dict, dict, int]:
    """Returns (recall[method][k] for paraphrase, same for verbatim, n)."""
    hits = {m: {k: 0 for k in ks} for m in METHODS}
    verbatim_hits = {m: {k: 0 for k in ks} for m in METHODS}
    total = 0
    max_k = max(ks)

    for era in eras:
        lore = list(load_lore(era))
        retriever = load_era_retriever(era)
        if retriever is None:
            raise SystemExit(
                f"[!] no index for {era} — run scripts/build_lore_index.py first"
            )

        cases = [(q["query"], q["doc"]) for q in query_sets.get(era, [])]
        # Verbatim stems: the sanity ceiling, one per paraphrase case so both
        # sets are the same size and directly comparable.
        verbatim = [(lore[doc].get("_topic_zh", ""), doc) for _, doc in cases]

        for case_set, bucket in ((cases, hits), (verbatim, verbatim_hits)):
            # One batched embedding pass for the whole set: per-request
            # overhead measured ~14× the per-item cost of a batch of 64.
            queries = [query for query, _ in case_set]
            embeddings: list[list[float]] = []
            for start in range(0, len(queries), 64):
                embeddings.extend(retriever.embedder.embed(queries[start : start + 64]))

            for (query, gold), embedding in zip(case_set, embeddings):
                results = {
                    "substring": substring_rank(lore, query, max_k),
                    "bm25": retriever.search(query, "zh", max_k, mode="bm25")[0],
                }
                for mode in ("vector", "hybrid"):
                    ranked, used = retriever.search(
                        query, "zh", max_k, mode=mode, query_embedding=embedding
                    )
                    if used != mode:
                        raise SystemExit(
                            f"[!] asked for {mode}, got {used} — is Ollama up? "
                            "Refusing to report degraded numbers under the wrong label."
                        )
                    results[mode] = ranked
                for method in METHODS:
                    for k in ks:
                        if gold in results[method][:k]:
                            bucket[method][k] += 1
        total += len(cases)
        print(f"{era}: {len(cases)} queries evaluated", flush=True)

    return hits, verbatim_hits, total


def table(title: str, hits: dict, ks: list[int], total: int) -> str:
    lines = [f"### {title}", "", "| method | " + " | ".join(f"recall@{k}" for k in ks) + " |"]
    lines.append("|---" * (len(ks) + 1) + "|")
    for method in METHODS:
        cells = " | ".join(f"{hits[method][k] / total:.1%}" for k in ks)
        lines.append(f"| {method} | {cells} |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eras", default="all")
    parser.add_argument("--k", default="1,3,5")
    args = parser.parse_args()

    if not QUERIES_PATH.is_file():
        raise SystemExit("[!] eval/retrieval_queries.json missing — run build_retrieval_eval.py")
    with QUERIES_PATH.open(encoding="utf-8") as fh:
        query_sets = json.load(fh)

    eras = (
        sorted(query_sets)
        if args.eras.strip().lower() == "all"
        else [e.strip() for e in args.eras.split(",") if e.strip()]
    )
    ks = sorted(int(k) for k in args.k.split(","))

    hits, verbatim_hits, total = evaluate(eras, ks, query_sets)

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S+0000")
    report = "\n\n".join(
        [
            f"# Retrieval evaluation — {stamp}",
            f"Eras: {', '.join(eras)} · {total} paraphrase queries "
            f"(gold = the record each query was rewritten from; conservative, "
            f"see script docstring).",
            table("Paraphrase queries (what players actually do)", hits, ks, total),
            table("Verbatim stems (sanity ceiling)", verbatim_hits, ks, total),
        ]
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"retrieval_{stamp}.md"
    out.write_text(report + "\n", encoding="utf-8")
    print()
    print(report)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
