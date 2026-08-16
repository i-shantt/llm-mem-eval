"""Keep verbatim turns, but only a fixed fraction of the tokens.

This is the control that makes a survival gap interpretable. An LLM-extracted
store is much smaller than the conversation it came from, and *any* small store
survives less than a large one -- so a raw extraction-versus-verbatim comparison
confounds two things: losing information by compressing, and losing it by
rewriting.

Truncated-verbatim compresses without rewriting. Sweeping the fraction traces a
survival-versus-store-size curve for pure compression, and an extraction store
can then be placed on that curve at its own token budget. If it sits on the
curve, its loss is compression. If it sits below, the extraction step itself is
lossy beyond what its size explains.

Selection rule matters and is not incidental:

    recency  keeps the most recent turns. This is what a fixed-window memory
             does, and it is the strongest cheap heuristic on a benchmark whose
             questions are asked at the end of the conversation.
    random   keeps an unbiased sample. Run several seeds -- the spread across
             seeds is the honest noise floor for the whole comparison, and at
             tight budgets it is not small.
"""

from __future__ import annotations

import random

from memllm.cost import CostLedger, count_tokens
from memllm.data.loader import Example, MemoryUnit, parse_date
from memllm.write.base import renumber


class TruncatedVerbatimPolicy:
    """Verbatim turns filling a per-example fraction of the full token budget."""

    def __init__(self, fraction: float, rule: str = "recency", seed: int = 0,
                 granularity: str = "turn"):
        if not 0 < fraction <= 1:
            raise ValueError(f"fraction must be in (0, 1]; got {fraction}")
        if rule not in ("recency", "random"):
            raise ValueError(f"unknown rule: {rule}")
        self.fraction = fraction
        self.rule = rule
        self.seed = seed
        self.granularity = granularity
        pct = f"{fraction:.0%}".replace("%", "pct")
        self.name = (f"truncated_{rule}_{pct}"
                     + (f"_s{seed}" if rule == "random" else ""))

    def config(self) -> dict:
        return {
            "policy": "truncated_verbatim",
            "fraction": self.fraction,
            "rule": self.rule,
            "seed": self.seed,
            "granularity": self.granularity,
        }

    def build(self, ex: Example, ledger: CostLedger) -> list[MemoryUnit]:
        with ledger.timer("write"):
            units = ex.units(self.granularity)
            tokens = {u.unit_id: count_tokens(u.text) for u in units}
            budget = self.fraction * sum(tokens.values())

            if self.rule == "recency":
                order = sorted(
                    units,
                    key=lambda u: (parse_date(u.session_date), u.session_index,
                                   u.unit_id),
                    reverse=True,
                )
            else:
                # Seeded per example, not once per run, so a store is
                # reproducible from its manifest regardless of how many
                # examples ran before it or in what order.
                rng = random.Random(f"{self.seed}|{ex.question_id}")
                order = list(units)
                rng.shuffle(order)

            kept, used = [], 0
            for u in order:
                t = tokens[u.unit_id]
                # Skip rather than stop: a single very long turn early in the
                # order should not strand the rest of the budget unspent, which
                # would make the realised size depend on turn-length ordering.
                if used + t <= budget:
                    kept.append(u)
                    used += t

            # Back into conversation order. Retrievers do not care, but the
            # answering prompt reads units in list order and a shuffled
            # transcript is a different treatment from a truncated one.
            kept.sort(key=lambda u: u.unit_id)
            return renumber(kept)
