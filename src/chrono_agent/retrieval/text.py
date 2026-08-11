"""Tokenization for BM25.

Chinese has no word boundaries, so naive `.split()` turns a whole sentence into
one token and BM25 degenerates to exact-string matching — which is precisely the
weakness this package exists to fix. jieba is the standard segmenter; English
and anything else falls back to lowercase whitespace splitting.
"""

from __future__ import annotations

import re

_PUNCT = re.compile(r"[\s，。！？；：、「」『』（）《》【】·…—\-,.!?;:()\[\]\"']+")

_jieba = None


def _segmenter():
    global _jieba
    if _jieba is None:
        import jieba  # deferred: first import builds its dictionary cache

        jieba.setLogLevel(60)  # silence the "Building prefix dict" banner
        _jieba = jieba
    return _jieba


def tokenize(text: str, language: str = "zh") -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    if language == "zh":
        tokens = _segmenter().lcut(text)
    else:
        tokens = text.lower().split()
    return [t for t in (_PUNCT.sub("", tok) for tok in tokens) if t]
