"""Retriever interface. Every system is indexed per example (each LongMemEval
question has its own haystack), then queried once."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..cost import CostLedger
from ..data.loader import MemoryUnit

Hit = tuple[int, float]  # (unit_id, score)


@runtime_checkable
class Retriever(Protocol):
    name: str

    def index(
        self,
        units: list[MemoryUnit],
        ledger: CostLedger,
        cache_key: str | None = None,
    ) -> None:
        """Build the memory store. All cost here is write-path."""
        ...

    def search(
        self,
        query: str,
        k: int,
        ledger: CostLedger,
        question_date: str | None = None,
    ) -> list[Hit]:
        """Return top-k (unit_id, score), best first. Cost here is read-path."""
        ...


def rrf_fuse(rankings: list[list[Hit]], k_rrf: int = 60) -> list[Hit]:
    """Reciprocal rank fusion. Score-agnostic, so BM25 and cosine can be
    combined without calibrating their scales against each other."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, (unit_id, _) in enumerate(ranking):
            scores[unit_id] = scores.get(unit_id, 0.0) + 1.0 / (k_rrf + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])
