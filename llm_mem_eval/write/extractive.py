"""Keep a few sentences of every turn: breadth instead of depth.

`TruncatedVerbatimPolicy` spends a token budget on *some turns, complete*. This
policy spends the same budget on *every turn, shortened*. Both are verbatim and
neither calls an LLM, so putting them at the same budget isolates one variable:
how the budget is spent.

That matters because an LLM extraction store looks like this one, not like
truncation -- many short records covering the whole conversation. If the
extraction arm's survival matches lead-k at the same budget, its loss is a
consequence of *being small and broad*, not of the LLM having rewritten
anything. That is a much more interesting result than a bare extraction-versus-
verbatim gap, and it is available for free.

Selection is round-robin, not per-turn proportional: pass 1 keeps the first
sentence of every turn, pass 2 the second, and so on until the budget runs out.
Round-robin guarantees that every turn is represented before any turn gets a
second sentence, which is the property that makes this "breadth".

`rule="tail"` counts from the other end, and it is the control that decides what
a lead-k result means. "Broad beats deep" and "people state facts at the start of
a message" predict the same lead-k number and different tail-k ones, so without
this arm the two explanations are indistinguishable and the interesting one would
be assumed.

A centrality-based selector (LexRank, TextRank) is the natural next arm and is
deliberately not here: it needs a sentence tokenizer and a similarity matrix for
a third point on the same curve, and lead/tail already establish the axis.

Sentence splitting is a regex, not a parser. It over-splits abbreviations and
decimals. Since the same splitter is applied to every arm, the effect is a small
constant on where a record boundary falls, not a bias between arms.
"""

from __future__ import annotations

import re

from llm_mem_eval.cost import CostLedger, count_tokens
from llm_mem_eval.data.loader import Example, MemoryUnit
from llm_mem_eval.write.base import renumber

# Split after ., ! or ? when followed by whitespace and a capital or digit.
# Keeps the terminator with the sentence it ends.
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENT_RE.split(text) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


class ExtractiveSelectionPolicy:
    """Lead-k (or tail-k) sentences per turn, round-robin, to a token budget.

    `rule="tail"` is the control that decides what the lead-k result means. If
    breadth is what helps, taking the *last* sentences of every turn should do
    about as well as taking the first. If instead lead-k wins because people
    state facts at the start of a message, tail-k collapses. Without this arm,
    "broad beats deep" and "answers appear early" are indistinguishable.
    """

    def __init__(self, fraction: float, granularity: str = "turn",
                 rule: str = "lead"):
        if not 0 < fraction <= 1:
            raise ValueError(f"fraction must be in (0, 1]; got {fraction}")
        if rule not in ("lead", "tail"):
            raise ValueError(f"unknown rule: {rule}")
        self.fraction = fraction
        self.granularity = granularity
        self.rule = rule
        pct = f"{fraction:.0%}".replace("%", "pct")
        self.name = ("leadk_" if rule == "lead" else "tailk_") + pct

    def config(self) -> dict:
        return {
            "policy": "extractive_k",
            "fraction": self.fraction,
            "granularity": self.granularity,
            "rule": self.rule,
            "selector": f"round-robin {self.rule} sentences",
        }

    def build(self, ex: Example, ledger: CostLedger) -> list[MemoryUnit]:
        with ledger.timer("write"):
            units = ex.units(self.granularity)
            budget = self.fraction * sum(count_tokens(u.text) for u in units)

            sents = [split_sentences(u.text) for u in units]
            kept: list[list[int]] = [[] for _ in units]
            used = 0
            depth = 0
            max_depth = max((len(s) for s in sents), default=0)

            while depth < max_depth and used < budget:
                progressed = False
                for i, s in enumerate(sents):
                    if depth >= len(s):
                        continue
                    # Same round-robin, counted from the other end for `tail`.
                    j = depth if self.rule == "lead" else len(s) - 1 - depth
                    t = count_tokens(s[j])
                    if used + t > budget:
                        # Skip rather than stop, for the same reason as
                        # TruncatedVerbatimPolicy: one long sentence should not
                        # strand the remaining budget.
                        continue
                    kept[i].append(j)
                    used += t
                    progressed = True
                if not progressed:
                    # Every remaining sentence is individually over budget.
                    break
                depth += 1

            out = [
                MemoryUnit(
                    unit_id=u.unit_id,
                    # Kept sentences are re-joined in their original order, so
                    # the only difference between the two rules is *which*
                    # sentences survive, not how the record reads.
                    text=" ".join(s[j] for j in sorted(k)),
                    session_id=u.session_id,
                    session_date=u.session_date,
                    session_index=u.session_index,
                    # Labelled later, by containment, exactly as every other
                    # store is -- so all arms are read off one ruler.
                    is_evidence=False,
                    roles=u.roles,
                    provenance=(u.unit_id,),
                )
                for u, s, k in zip(units, sents, kept) if k
            ]
            return renumber(out)
