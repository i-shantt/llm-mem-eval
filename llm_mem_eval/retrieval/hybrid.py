"""Hybrid BM25 + dense retrieval with an optional recency prior.

This is the "well-tuned trivial baseline" the memory-systems literature
compares against only in its weakest form. Still zero LLM calls.
"""

from __future__ import annotations

from ..cost import CostLedger
from ..data.loader import MemoryUnit, parse_date
from .base import Hit, rrf_fuse
from .bm25 import BM25Retriever
from .dense import DenseRetriever


class HybridRetriever:
    name = "hybrid"

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: str = "auto",
        recency_weight: float = 0.0,
        rrf_k: int = 60,
        fuse_depth: int = 100,
        cache=None,
    ) -> None:
        self.bm25 = BM25Retriever()
        self.dense = DenseRetriever(
            model_name=model_name, device=device, cache=cache
        )
        self.recency_weight = recency_weight
        self.rrf_k = rrf_k
        self.fuse_depth = fuse_depth
        self._units: list[MemoryUnit] = []

    def warmup(self) -> None:
        self.dense.warmup()

    def index(
        self,
        units: list[MemoryUnit],
        ledger: CostLedger,
        cache_key: str | None = None,
    ) -> None:
        self._units = units
        self.bm25.index(units, ledger, cache_key)
        self.dense.index(units, ledger, cache_key)

    def _recency_ranking(self, question_date: str | None) -> list[Hit]:
        """Most recent session first. Ordinal, so it fuses cleanly via RRF.

        `question_date` is accepted and deliberately unused. In LongMemEval-S it
        cannot change this ordering: the question is dated at or after every
        session in its own haystack for all 500 questions, so "closest to the
        question date" and "most recent" are the same permutation. See
        `scripts/audit_question_dates.py`, which checks that rather than
        assuming it.

        The stronger version of the same point: any recency prior that is
        monotone in session date produces this permutation, whatever origin it
        measures age from. Making the prior a function of `question_date - date`
        does not escape it either -- for exponential decay the origin cancels
        out of every pairwise ratio exactly.
        """
        order = sorted(
            self._units,
            key=lambda u: (parse_date(u.session_date), u.session_index),
            reverse=True,
        )
        return [(u.unit_id, 0.0) for u in order[: self.fuse_depth]]

    def search(
        self,
        query: str,
        k: int,
        ledger: CostLedger,
        question_date: str | None = None,
    ) -> list[Hit]:
        d = self.fuse_depth
        bm_hits = self.bm25.search(query, d, ledger)
        dn_hits = self.dense.search(query, d, ledger)

        rankings = [bm_hits, dn_hits]
        weights = [1.0, 1.0]
        if self.recency_weight > 0:
            # Weighted directly, so `recency_weight` reads as "this ranking's
            # influence relative to BM25 and dense, each of which is 1.0".
            # It used to be expressed as integer repetitions of the ranking,
            # which quantised it: every weight in (0, 0.75] rounded to one whole
            # ranking, so the knob had no setting between "off" and "as strong
            # as BM25". Nothing committed ran at a non-zero weight, so no
            # published number is affected by the change of meaning.
            rankings.append(self._recency_ranking(question_date))
            weights.append(self.recency_weight)

        with ledger.timer("read"):
            fused = rrf_fuse(rankings, k_rrf=self.rrf_k, weights=weights)
        return fused[:k]
