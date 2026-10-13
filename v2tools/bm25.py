"""Okapi BM25 in pure Python, deterministic: ties are broken by document id."""
from __future__ import annotations

import math
import re
from collections import Counter

_WORD = re.compile(r"[A-Za-z0-9]+")
_PART = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")


def tokens(text: str) -> list[str]:
    """Words split on non-alphanumerics and camelCase, lowercased: "iFilterByCode" -> ["i", "filter", "by", "code"]."""
    return [p.lower() for w in _WORD.findall(text) for p in _PART.findall(w)]


class BM25:
    def __init__(self, docs: list[tuple[str, list[str]]], k1: float = 1.5, b: float = 0.75):
        self.ids = [d for d, _ in docs]
        self.tf = [Counter(t) for _, t in docs]
        self.len = [len(t) for _, t in docs]
        self.avg = (sum(self.len) / len(self.len)) if self.len else 0.0
        df = Counter(w for c in self.tf for w in c)
        n = len(docs)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}
        self.k1, self.b = k1, b

    def rank(self, query: list[str]) -> list[tuple[str, float]]:
        q = Counter(query)
        out = []
        for i, doc in enumerate(self.ids):
            tf, dl, s = self.tf[i], self.len[i], 0.0
            for w in q:
                f = tf.get(w, 0)
                if f:
                    s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / (self.avg or 1)))
            out.append((doc, s))
        out.sort(key=lambda x: (-x[1], x[0]))
        return out
