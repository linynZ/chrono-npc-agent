"""Retrieval over the lore index: vectors first, BM25 as the fallback rung.

This shipped as hybrid-by-default (BM25 + vectors, RRF-fused) on the industry
default that fusion beats either method alone. The eval said otherwise: on 500
paraphrased player queries, pure vector recall@1 was 95.8% while hybrid came in
at 81.6% — BM25 degrades badly when a paraphrase shares no wording with the
record, and equal-weight RRF mixes those noisy ranks into a vector ranking that
was nearly perfect on its own. So the default is `vector`, hybrid stays as an
option (and in the eval, so the decision re-checks itself when the corpus
changes), and BM25's real job is the degradation ladder. Finding 06 has the
full numbers.

Degradation matches the project's general stance that quality may drop but
replies may not: vector → BM25 (embedding server down) → the caller's own
substring fallback (this package not set up at all).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .embedder import EmbeddingUnavailable
from .text import tokenize

COLLECTION = "lore"


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> list[str]:
    """Reciprocal Rank Fusion. Pure, so it is trivially testable.

    k=60 is the constant from the original TREC paper; it damps the advantage
    of rank-1 items just enough that one list cannot dictate the result alone.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))


def _doc_text(entry: dict, language: str) -> str:
    # Topic (the question stem) plus the record itself: the stem carries the
    # entity names a player is most likely to use, the record carries the fact.
    return f"{entry.get(f'_topic_{language}', '')}\n{entry.get(language, '')}".strip()


@dataclass
class LoreRetriever:
    entries: list[dict]
    embedder: Any
    collection: Any  # chromadb Collection; typed loosely to keep imports lazy
    _bm25: dict[str, Any] = field(default_factory=dict, repr=False)

    def _bm25_for(self, language: str):
        if language not in self._bm25:
            from rank_bm25 import BM25Okapi

            corpus = [tokenize(_doc_text(e, language), language) for e in self.entries]
            # BM25Okapi chokes on an all-empty corpus; guard with a dot.
            self._bm25[language] = BM25Okapi([tok or ["."] for tok in corpus])
        return self._bm25[language]

    def _bm25_ranking(self, query: str, language: str, depth: int) -> list[str]:
        scores = self._bm25_for(language).get_scores(tokenize(query, language))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        return [str(i) for i in order[:depth] if scores[i] > 0]

    def _vector_ranking(
        self, query: str, language: str, depth: int, embedding: list[float] | None
    ) -> list[str]:
        if embedding is None:
            [embedding] = self.embedder.embed([query])
        result = self.collection.query(
            query_embeddings=[embedding],
            n_results=min(depth, len(self.entries)),
            where={"lang": language},
        )
        return [doc_id.split(":", 1)[1] for doc_id in result["ids"][0]]

    def search(
        self,
        query: str,
        language: str = "zh",
        k: int = 3,
        mode: str = "vector",
        query_embedding: list[float] | None = None,
    ) -> tuple[list[int], str]:
        """Return (entry indices, mode actually used). Never raises.

        `mode` is what the caller wants; what it gets may be weaker. The mode
        used goes back to the caller because a diagnostic that says "hybrid"
        while the embedder was down would hide exactly the failures worth
        seeing.

        `query_embedding` lets a batch caller (the eval harness) embed many
        queries in one request first — per-request embedding overhead measured
        ~14× the per-item cost of a batch of 64, which turns a 5-minute eval
        into a 90-minute one.
        """
        depth = max(k * 4, 10)

        if mode in ("hybrid", "vector"):
            try:
                vector = self._vector_ranking(query, language, depth, query_embedding)
                if mode == "vector":
                    return [int(i) for i in vector[:k]], "vector"
                fused = rrf_fuse([vector, self._bm25_ranking(query, language, depth)])
                return [int(i) for i in fused[:k]], "hybrid"
            except EmbeddingUnavailable:
                mode = "bm25"

        return [int(i) for i in self._bm25_ranking(query, language, depth)[:k]], "bm25"


def build_index(entries: list[dict], embedder: Any, persist_dir: str | Path) -> int:
    """(Re)build the vector collection. Returns the number of documents stored.

    Both languages of every entry go in as separate documents with a `lang`
    tag — bge-m3 is multilingual, but keeping the languages apart means a zh
    query is never answered with an en record the player cannot read.
    """
    import chromadb

    client = chromadb.PersistentClient(
        path=str(persist_dir), settings=chromadb.Settings(anonymized_telemetry=False)
    )
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass  # first build
    collection = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []
    for index, entry in enumerate(entries):
        for language in ("zh", "en"):
            text = _doc_text(entry, language)
            if not text:
                continue
            ids.append(f"{language}:{index}")
            documents.append(text)
            metadatas.append({"lang": language, "category": entry.get("category", "")})

    batch = 64
    for start in range(0, len(ids), batch):
        chunk = slice(start, start + batch)
        collection.add(
            ids=ids[chunk],
            documents=documents[chunk],
            embeddings=embedder.embed(documents[chunk]),
            metadatas=metadatas[chunk],
        )
    return len(ids)


def load_retriever(
    entries: list[dict], persist_dir: str | Path, embedder: Any
) -> LoreRetriever | None:
    """None when the index was never built — the caller keeps its old behavior.

    A missing index is a setup state, not an error: the service must run on a
    fresh clone with zero retrieval setup, exactly as it did before this
    package existed.
    """
    path = Path(persist_dir)
    if not path.is_dir():
        return None
    try:
        import chromadb

        client = chromadb.PersistentClient(
            path=str(path), settings=chromadb.Settings(anonymized_telemetry=False)
        )
        collection = client.get_collection(COLLECTION)
        # The index must describe the entries we actually loaded; a stale
        # index quietly returning wrong rows would be worse than none at all.
        expected = sum(
            1 for e in entries for lang in ("zh", "en") if _doc_text(e, lang)
        )
        if collection.count() != expected:
            return None
    except Exception:
        return None

    return LoreRetriever(entries=entries, embedder=embedder, collection=collection)
