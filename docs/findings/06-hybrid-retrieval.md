# Finding 06 — The eval was ordered to justify vectors; it convicted hybrid instead

**Date:** 2026-08-11
**Status:** vector retrieval (bge-m3 + Chroma) shipped as the default; hybrid demoted to an option; BM25 kept as the degradation rung
**Reproduce:** `python scripts/build_lore_index.py && python scripts/evaluate_retrieval.py` (queries in `eval/retrieval_queries.json`, committed)

## What I expected

Two assumptions went in. First: substring overlap (the shipped search) would be
mediocre on paraphrased queries, and a vector index would fix it. Second, the
industry default: hybrid — BM25 + vectors fused with RRF — beats either method
alone, so hybrid would be the production mode. The first survived. The second
did not.

## How it was measured

Querying the index with the stored question stems would be self-dealing — the
stem is *inside* the record, so substring gets handed its own answer. Instead,
DeepSeek rewrote all 500 stems into colloquial player questions ("古罗马那个能
坐五万人看角斗的大圆场子叫啥来着？"), committed for reproducibility and
spot-checked by eye. Gold = the record each query was rewritten from; recall is
counted conservatively (a near-duplicate record on the same topic counts as a
miss, for every method equally).

The verbatim stems run alongside as a sanity ceiling — if substring does not
look good there, the harness itself is broken.

## What actually happened

500 paraphrase queries, recall against the origin record:

| method | recall@1 | recall@3 | recall@5 |
|---|---|---|---|
| substring (shipped baseline) | 71.2% | 86.6% | 91.8% |
| BM25 (jieba tokens) | 69.2% | 83.0% | 86.6% |
| **vector (bge-m3)** | **95.8%** | **99.8%** | **99.8%** |
| hybrid (RRF of both) | 81.6% | 93.4% | 96.2% |

Verbatim stems as the sanity ceiling: every method ≥99.6% @1 — the harness is
sound, the differences above are real.

## What it means

**Vectors earned the upgrade** — recall@1 jumped 71.2% → 95.8%, and at k=3 (the
tool's default `limit`) retrieval is effectively solved for this corpus.

**Hybrid lost to its own ingredient.** BM25 collapses on paraphrases: jieba
tokens from "能坐五万人看角斗的大圆场子" share almost nothing with a record
about the Colosseum, so its ranking is noise — and equal-weight RRF dutifully
mixed that noise into a vector ranking that was nearly perfect alone, costing
14 points at k=1. Fusion helps when both signals carry information; here one
didn't. (Substring out-scoring BM25 is the same story in miniature: per-CJK-
character matching accidentally keeps entity fragments that whole-word tokens
lose.)

**Why not delete BM25 then:** its failure mode is *different*, which is what a
degradation rung is for. Vectors need the embedding server up; BM25 needs
nothing but the process itself. The ladder is now vector → BM25 → substring,
and the tool reports which rung answered (`retrieval: vector | bm25 |
substring`), so a degraded lookup is visible in diagnostics instead of reading
as a bad model day.

**Caveats, honestly held:** one corpus (500 short bilingual records), one query
style (LLM paraphrases — fluent, full-sentence; real players also type two-word
fragments where BM25 does better), gold-by-origin scoring that counts a
near-duplicate twin as a miss for every method equally. Hybrid stays in the
eval precisely so this decision gets re-checked when the corpus or the query
mix changes.

## The side lesson: batch your embeddings

The first eval run was killed at the 45-minute mark, still unfinished. Cause:
one HTTP embed call per query — ~2.8 s each under load, where a batch of 64
amortizes to ~200 ms per item (~14×). Batching the same workload finished in
about 3 minutes. The embedder now holds a persistent connection, and the
service pays the ~2 s model-load warmup at startup instead of on the first
player's question.
