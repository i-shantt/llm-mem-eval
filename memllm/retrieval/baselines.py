"""Reference points that bound the frontier.

OracleRetriever   -- retrieves exactly the gold evidence turns
RecencyRetriever  -- "just keep the last N turns", the cheapest thing that works
RandomRetriever   -- lower bound; distinguishes real retrieval from lucky priors
NoMemoryRetriever -- retrieves nothing; the closed-book floor

Note what the oracle is and is not. It is a ceiling for the *retrieval metrics*
by construction: it ranks every evidence unit first, so any_hit, recall and MRR
are all 1.000 and nothing can beat it.

It is **not** an end-to-end ceiling. Questions carry ~1.9 evidence turns on
average and never more than six, so at k=10 the arm is gold evidence padded out
with non-evidence turns in conversation order -- roughly 80% filler. That is a
*different* context from a retriever's, not a superset of one, and a retriever
that surfaces an unlabelled turn which happens to carry the answer can beat it.
It does: hybrid beats oracle at 14B. See "Oracle is not a ceiling" in RESULTS.md.
"""

from __future__ import annotations

import random

from ..cost import CostLedger
from ..data.loader import MemoryUnit, parse_date
from .base import Hit


class OracleRetriever:
    """Ranks gold evidence first, then pads to k with non-evidence units in
    conversation order.

    Isolates 'can the model answer *given* the labelled evidence' from 'can we
    find the right memory'. Scores 1.000 on every retrieval metric by
    construction; see the module docstring for why the padding means that does
    not make it an end-to-end upper bound.
    """

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


class NoMemoryRetriever:
    """Retrieves nothing. The control every memory paper omits.

    A memory system's headline accuracy is meaningless without it: some
    LongMemEval questions are answerable from world knowledge or leak their own
    answer in the phrasing, and that fraction is credited to memory by default.
    Whatever this arm scores is the floor the real system has to beat before any
    of its accuracy can be attributed to remembering anything.

    Pair with `--no-memory-prompt closed_book` so the model is asked the
    question directly. Handing the normal template an empty excerpt block
    measures induced refusal, not prior knowledge, and understates the floor.
    """

    name = "none"

    def index(
        self,
        units: list[MemoryUnit],
        ledger: CostLedger,
        cache_key: str | None = None,
    ) -> None:
        with ledger.timer("write"):
            pass

    def search(
        self, query: str, k: int, ledger: CostLedger,
        question_date: str | None = None,
    ) -> list[Hit]:
        with ledger.timer("read"):
            return []


class RandomRetriever:
    """Matched-budget noise. Separates 'memory helped' from 'more tokens
    helped': it spends the same read budget as the real retriever on units
    chosen without reference to the query."""

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
