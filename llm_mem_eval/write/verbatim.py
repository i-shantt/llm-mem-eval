"""Store the conversation as it was said. The control every other policy is read against.

This is the write path with no write path: it does nothing to the text, spends
no LLM calls, and keeps every token. Its survival rate is therefore the ceiling
any lossier policy is measured against -- and, importantly, that ceiling is
*measured, not assumed*. It is not 1.0, because a gold answer is sometimes a
computed value or a paraphrase that appears verbatim nowhere in the conversation
(see scripts/audit_benchmark.py, which quantifies exactly that).
"""

from __future__ import annotations

from llm_mem_eval.cost import CostLedger
from llm_mem_eval.data.loader import Example, Granularity, MemoryUnit


class VerbatimPolicy:
    """`Example.units()` behind the WritePolicy interface."""

    def __init__(self, granularity: Granularity = "turn"):
        self.granularity = granularity
        self.name = f"verbatim_{granularity}"

    def config(self) -> dict:
        return {"policy": "verbatim", "granularity": self.granularity}

    def build(self, ex: Example, ledger: CostLedger) -> list[MemoryUnit]:
        # Timed, though it is nearly free, so the write column of a verbatim
        # store is a real measurement rather than a hardcoded zero.
        with ledger.timer("write"):
            return ex.units(self.granularity)
