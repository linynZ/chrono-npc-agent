"""Embedding backends behind one seam, same pattern as the LLM providers.

`OllamaEmbedder` is the real one (bge-m3 by default — multilingual, which
matters because the lore is bilingual and a player may ask in either language).
`HashEmbedder` exists so the whole retrieval stack — indexing, vector search,
fusion — runs in tests with no model, no server and no randomness: it hashes
character n-grams into a fixed-size vector, so texts that share substance share
dimensions. Crude, but deterministic and directionally correct, which is all a
test needs.
"""

from __future__ import annotations

import hashlib
import math

import httpx


class EmbeddingUnavailable(RuntimeError):
    """Raised when the embedding backend cannot be reached.

    Callers treat this as a signal to degrade (vector → BM25), never as a
    user-facing error — retrieval quality is allowed to drop, replies are not.
    """


class OllamaEmbedder:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "bge-m3",
        timeout_s: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/").removesuffix("/v1")
        self.model = model
        self.timeout_s = timeout_s
        # One connection for the embedder's lifetime. Measured on the eval run:
        # a fresh connection per request cost seconds where a reused one costs
        # ~150 ms — and the first request after model load costs ~2 s however
        # you connect, which is what `warmup()` is for.
        self._client = httpx.Client(timeout=timeout_s)

    def warmup(self) -> None:
        """Absorb the model-load cost now instead of on the first player."""
        try:
            self.embed(["warmup"])
        except EmbeddingUnavailable:
            pass  # server not up; the retriever will degrade per request

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = self._client.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": texts},
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise EmbeddingUnavailable(f"{self.model} via {self.base_url}: {exc}") from exc

        embeddings = response.json().get("embeddings") or []
        if len(embeddings) != len(texts):
            raise EmbeddingUnavailable(
                f"asked for {len(texts)} embeddings, got {len(embeddings)}"
            )
        return embeddings


class HashEmbedder:
    """Deterministic offline stand-in. See module docstring."""

    def __init__(self, dim: int = 256, ngram: int = 2) -> None:
        self.dim = dim
        self.ngram = ngram

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(text) for text in texts]

    def _one(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        text = (text or "").lower()
        grams = [text[i : i + self.ngram] for i in range(max(1, len(text) - self.ngram + 1))]
        for gram in grams:
            digest = hashlib.md5(gram.encode("utf-8")).digest()
            vector[int.from_bytes(digest[:4], "little") % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]
