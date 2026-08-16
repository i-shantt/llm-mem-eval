"""Reference points that bound the frontier.

OracleRetriever   -- retrieves exactly the gold evidence turns
RecencyRetriever  -- "just keep the last N turns", the cheapest thing that works
RandomRetriever   -- lower bound; distinguishes real retrieval from lucky priors
NoMemoryRetriever -- retrieves nothing; the closed-book floor

Note what the oracle is and is not. It is a ceiling for the *retrieval metrics*
by construction, since it returns every evidence unit and nothing can score
higher than 1.000. It is **not** an end-to-end ceiling: it hands the model only
the turns LongMemEval labelled `has_answer`, which is a smaller prompt than a
real retriever builds on most questions, so a retriever that also picks up an
unlabelled turn carrying the answer can beat it -- and does. See "Oracle is not
a ceiling" in RESULTS.md.
"""

from __future__ import annotations

import random

from ..cost import CostLedger
from ..data.loader import MemoryUnit, parse_date
from .base import Hit


class OracleRetriever:
    """Returns gold evidence first, then arbitrary filler.

    Isolates 'can the model answer *given* the labelled evidence' from 'can we
    find the right memory'. Scores 1.000 on every retrieval metric by
    construction; see the module docstring for why that does not make it an
    end-to-end upper bound.
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
