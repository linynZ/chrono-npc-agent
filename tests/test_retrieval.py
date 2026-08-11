"""Retrieval-stack tests, all offline.

The vector path runs against a real chroma collection in a temp dir with the
deterministic HashEmbedder, so index build, metadata filtering and fusion are
exercised for real — only the embedding model is a stand-in. The degradation
ladder gets a rung-by-rung test because that ladder is a promise the service
makes ("quality may drop, replies may not"), and promises get tests.
"""

from __future__ import annotations

import pytest

from chrono_agent.models import PlayerState
from chrono_agent.retrieval import (
    HashEmbedder,
    LoreRetriever,
    build_index,
    load_retriever,
    rrf_fuse,
    tokenize,
)
from chrono_agent.retrieval.embedder import EmbeddingUnavailable
from chrono_agent.tools import ToolContext
from chrono_agent.tools.game_tools import _lookup_lore

ENTRIES = [
    {
        "category": "Architecture",
        "zh": "今天所见保存最完好的长城多为明代修筑的砖石长城。",
        "en": "The best-preserved Great Wall sections were built by the Ming dynasty.",
        "_topic_zh": "长城保存最完好的段落主要由哪个朝代修建？",
        "_topic_en": "Which dynasty built the best-preserved Great Wall sections?",
    },
    {
        "category": "Geography",
        "zh": "都江堰是战国时期李冰父子主持修建的无坝引水水利工程。",
        "en": "Dujiangyan is a dam-free irrigation system built under Li Bing.",
        "_topic_zh": "都江堰水利工程是谁主持修建的？",
        "_topic_en": "Who directed the construction of Dujiangyan?",
    },
    {
        "category": "Culture",
        "zh": "造纸术由东汉蔡伦改进，使纸张得以大量生产。",
        "en": "Papermaking was improved by Cai Lun in the Eastern Han dynasty.",
        "_topic_zh": "谁改进了造纸术？",
        "_topic_en": "Who improved papermaking?",
    },
]


# --- pure pieces -----------------------------------------------------------


def test_rrf_rewards_agreement_between_rankers():
    fused = rrf_fuse([["a", "b", "c"], ["b", "d", "a"]])
    # "b" (ranks 2+1) and "a" (ranks 1+3) both beat everything seen once.
    assert fused[0] in ("a", "b")
    assert set(fused[:2]) == {"a", "b"}
    assert fused.index("d") > fused.index("b")


def test_rrf_is_deterministic_on_ties():
    assert rrf_fuse([["x"], ["y"]]) == rrf_fuse([["x"], ["y"]])


def test_tokenize_zh_segments_words_not_chars():
    tokens = tokenize("都江堰是谁修建的", "zh")
    assert "都江堰" in tokens  # jieba keeps the proper noun whole
    assert len(tokens) < len("都江堰是谁修建的")


def test_tokenize_en_lowercases():
    assert tokenize("The Great Wall", "en") == ["the", "great", "wall"]


def test_hash_embedder_is_deterministic_and_normalized():
    embedder = HashEmbedder()
    [a1], [a2] = embedder.embed(["都江堰"]), embedder.embed(["都江堰"])
    assert a1 == a2
    assert abs(sum(v * v for v in a1) - 1.0) < 1e-6


# --- the real collection, fake embeddings ----------------------------------


@pytest.fixture()
def retriever(tmp_path):
    embedder = HashEmbedder()
    build_index(ENTRIES, embedder, tmp_path / "index")
    loaded = load_retriever(ENTRIES, tmp_path / "index", embedder)
    assert loaded is not None
    return loaded


def test_default_mode_is_vector(retriever):
    # Finding 06: vector beat hybrid on the paraphrase eval, so it is the default.
    indices, mode = retriever.search("长城是哪个朝代修的", "zh", k=1)
    assert mode == "vector"
    assert indices == [0]


def test_hybrid_still_available_and_finds_the_record(retriever):
    indices, mode = retriever.search("长城是哪个朝代修的", "zh", k=1, mode="hybrid")
    assert mode == "hybrid"
    assert indices == [0]


def test_missing_index_dir_means_no_retriever(tmp_path):
    assert load_retriever(ENTRIES, tmp_path / "nowhere", HashEmbedder()) is None


def test_stale_index_count_mismatch(tmp_path):
    # An index built for three entries must not serve a four-entry lore list:
    # row ids would silently point at the wrong records.
    embedder = HashEmbedder()
    build_index(ENTRIES, embedder, tmp_path / "index")
    longer = ENTRIES + [
        {
            "category": "X",
            "zh": "多出来的记录。",
            "en": "An extra record.",
            "_topic_zh": "多出来的题面？",
            "_topic_en": "An extra stem?",
        }
    ]
    assert load_retriever(longer, tmp_path / "index", embedder) is None


class _DownEmbedder:
    """Healthy at index time, dead at query time — the realistic failure."""

    def __init__(self):
        self._inner = HashEmbedder()
        self.down = False

    def embed(self, texts):
        if self.down:
            raise EmbeddingUnavailable("server went away")
        return self._inner.embed(texts)


def test_embedder_outage_degrades_to_bm25(tmp_path):
    embedder = _DownEmbedder()
    build_index(ENTRIES, embedder, tmp_path / "index")
    retriever = load_retriever(ENTRIES, tmp_path / "index", embedder)
    embedder.down = True

    indices, mode = retriever.search("都江堰是谁修建的", "zh", k=1)
    assert mode == "bm25"  # degraded, and says so
    assert indices == [1]  # jieba + BM25 still lands the proper noun


# --- the tool walks the ladder ----------------------------------------------


def _context(retriever=None):
    return ToolContext(
        state=PlayerState(current_map="20_China"),
        language="zh",
        lore=ENTRIES,
        retriever=retriever,
    )


def test_lookup_lore_reports_retrieval_mode(tmp_path):
    embedder = HashEmbedder()
    build_index(ENTRIES, embedder, tmp_path / "index")
    retriever = load_retriever(ENTRIES, tmp_path / "index", embedder)

    result = _lookup_lore(_context(retriever), topic="长城是哪个朝代修的")
    assert result["found"] is True
    assert result["retrieval"] == "vector"
    assert "明代" in result["records"][0]["fact"]


def test_lookup_lore_without_retriever_still_answers():
    result = _lookup_lore(_context(retriever=None), topic="长城")
    assert result["found"] is True
    assert result["retrieval"] == "substring"


def test_lookup_lore_no_hits_note_is_unchanged():
    result = _lookup_lore(_context(retriever=None), topic="quantum")
    assert result["found"] is False
