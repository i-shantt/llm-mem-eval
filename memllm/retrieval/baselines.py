"""Reference points that bound the frontier.

OracleRetriever   -- upper bound on retrieval; any system that beats it has a bug
RecencyRetriever  -- "just keep the last N turns", the cheapest thing that works
RandomRetriever   -- lower bound; distinguishes real retrieval from lucky priors
"""

from __future__ import annotations

import random

from ..cost import CostLedger
from ..data.loader import MemoryUnit, parse_date
from .base import Hit


class OracleRetriever:
    """Returns gold evidence first, then arbitrary filler. Ceiling for the
    retrieval metric, and the arm that isolates 'can the model answer *given*
    perfect retrieval' from 'can we find the right memory'."""

    name = "oracle"

    def __init__(self) -> None:
        self._units: list[MemoryUnit] = []

    def index(
        self,
        units: list[MemoryUnit],
        ledger: CostLedger,
        cache_key: str | None = None,
    ) -> None:
        with ledger.timer("write"):
            self._units = units

    def search(
        self, query: str, k: int, ledger: CostLedger,
        question_date: str | None = None,
    ) -> list[Hit]:
        with ledger.timer("read"):
            ev = [u for u in self._units if u.is_evidence]
            rest = [u for u in self._units if not u.is_evidence]
            ordered = ev + rest
            return [(u.unit_id, 1.0 if u.is_evidence else 0.0)
                    for u in ordered[:k]]


class RecencyRetriever:
    """Most recent turns by session date, then position. No query use at all --
    which is exactly why it's the honest floor for 'memory' claims."""

    name = "recency"

    def __init__(self) -> None:
        self._units: list[MemoryUnit] = []

    def index(
        self,
        units: list[MemoryUnit],
        ledger: CostLedger,
        cache_key: str | None = None,
    ) -> None:
        with ledger.timer("write"):
            self._units = units

    def search(
        self, query: str, k: int, ledger: CostLedger,
        question_date: str | None = None,
    ) -> list[Hit]:
        with ledger.timer("read"):
            ordered = sorted(
                self._units,
                key=lambda u: (parse_date(u.session_date), u.session_index, u.unit_id),
                reverse=True,
            )
            return [(u.unit_id, float(-i)) for i, u in enumerate(ordered[:k])]


class RandomRetriever:
    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self._units: list[MemoryUnit] = []
        self._seed = seed

    def index(
        self,
        units: list[MemoryUnit],
        ledger: CostLedger,
        cache_key: str | None = None,
    ) -> None:
        with ledger.timer("write"):
            self._units = units

    def search(
        self, query: str, k: int, ledger: CostLedger,
        question_date: str | None = None,
    ) -> list[Hit]:
        with ledger.timer("read"):
            rng = random.Random(self._seed + len(query))
            picked = rng.sample(self._units, min(k, len(self._units)))
            return [(u.unit_id, 0.0) for u in picked]
