"""Build the per-era vector index the hybrid retriever runs on.

One chroma collection per era, under data/index/<era>/. Requires Ollama with
the embedding model pulled (`ollama pull bge-m3`); everything else is local.
Safe to re-run — each build replaces the era's collection wholesale, so a
half-finished previous run cannot leave stale rows behind.

Usage:
    python scripts/build_lore_index.py              # every era in lore.json
    python scripts/build_lore_index.py --eras china
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chrono_agent.config import LORE_INDEX_DIR, Settings  # noqa: E402
from chrono_agent.factory import load_lore  # noqa: E402
from chrono_agent.retrieval import OllamaEmbedder, build_index  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eras", default="all", help="comma-separated, or 'all'")
    args = parser.parse_args()

    import json

    from chrono_agent.config import DATA_DIR

    lore_path = DATA_DIR / "lore.json"
    if not lore_path.is_file():
        print("[!] data/lore.json missing. Run scripts/extract_game_data.py first.")
        return 1
    with lore_path.open(encoding="utf-8") as fh:
        available = sorted(json.load(fh))

    eras = (
        available
        if args.eras.strip().lower() == "all"
        else [e.strip() for e in args.eras.split(",") if e.strip()]
    )

    settings = Settings.from_env()
    embedder = OllamaEmbedder(
        base_url=settings.ollama_base_url, model=settings.embed_model
    )

    for era in eras:
        entries = list(load_lore(era))
        if not entries:
            print(f"[!] {era}: no lore entries, skipped")
            continue
        started = time.perf_counter()
        count = build_index(entries, embedder, LORE_INDEX_DIR / era)
        print(
            f"{era:<8} {len(entries)} entries -> {count} documents "
            f"({time.perf_counter() - started:.1f}s)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
