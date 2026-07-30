"""BM25 lexical baseline -- the baseline the memory-systems literature skips.

Zero LLM calls, zero embedding calls, pure CPU. Anything that cannot beat this
on cost-adjusted terms is not earning its keep.
"""

from __future__ import annotations

import re

from ..cost import CostLedger
from ..data.loader import MemoryUnit
from .base import Hit

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Retriever:
    name = "bm25"

    def __init__(self) -> None:
        self._bm25 = None
        self._unit_ids: list[int] = []

    def index(
        self,
        units: list[MemoryUnit],
        ledger: CostLedger,
        cache_key: str | None = None,
    ) -> None:
        from rank_bm25 import BM25Okapi

        with ledger.timer("write"):
            corpus = [tokenize(u.text) for u in units]
            self._unit_ids = [u.unit_id for u in units]
            # BM25Okapi requires a non-empty vocabulary per doc; pad empties.
            corpus = [c if c else ["<empty>"] for c in corpus]
            self._bm25 = BM25Okapi(corpus)

    def search(
        self,
        query: str,
        k: int,
        ledger: CostLedger,
        question_date: str | None = None,
    ) -> list[Hit]:
        assert self._bm25 is not None, "index() must be called first"
        with ledger.timer("read"):
            scores = self._bm25.get_scores(tokenize(query))
            order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
            return [(self._unit_ids[i], float(scores[i])) for i in order]
