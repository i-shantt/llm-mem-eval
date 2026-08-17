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


def rrf_fuse(
    rankings: list[list[Hit]],
    k_rrf: int = 60,
    weights: list[float] | None = None,
) -> list[Hit]:
    """Reciprocal rank fusion. Score-agnostic, so BM25 and cosine can be
    combined without calibrating their scales against each other.

    `weights` scales each ranking's contribution and defaults to 1.0 each. It
    exists because the alternative -- expressing a ranking's weight by repeating
    it in the list -- quantises the weight to integers. A caller wanting a
    recency prior at a *fraction* of BM25's influence cannot say so that way:
    the smallest expressible non-zero weight is one whole ranking.

    A zero weight is skipped rather than multiplied through, so `weights=0` is
    exactly equivalent to omitting the ranking. Multiplying would instead insert
    that ranking's unit ids at score 0.0, padding the tail of the result with
    units the caller asked to ignore.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError(
            f"weights has length {len(weights)}, rankings has {len(rankings)}"
        )
    scores: dict[int, float] = {}
    for ranking, w in zip(rankings, weights):
        if w == 0:
            continue
        for rank, (unit_id, _) in enumerate(ranking):
            scores[unit_id] = scores.get(unit_id, 0.0) + w / (k_rrf + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])
