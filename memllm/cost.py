"""Cost instrumentation, split by write path vs read path.

Published memory systems report read-path tokens per query and are mostly silent
on the write path. Mem0's memory construction runs ~1.23M tokens for one LoCoMo
conversation, a figure only a competitor reports (RecMem, arXiv 2605.16045,
Table 1), because Mem0's own paper does not. This module accounts for both
phases separately, so total cost can be plotted against query volume rather than
quoted at the n -> infinity limit.

Two things this deliberately does not claim. Per-call cost is not the issue --
every construction figure quoted here already assumes a small, cheap extraction
model, which is what Mem0 OSS actually defaults to. And the dollar amount is not
the interesting quantity: scripts/model_write_cost.py shows the same shipped
code spanning 8.8x depending on how the *caller* batches, and
memllm/eval/survival.py asks the question cost cannot -- what the write path
gives up in answerable content.

See data/published_costs.json for every quoted figure and its source.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Literal

Phase = Literal["write", "read"]

# USD per 1M tokens. Local models cost no API dollars but still cost wall-clock,
# which we track separately; `local` is here so a self-hosted arm can be priced
# by amortised hardware if we ever want that.
PRICES: dict[str, tuple[float, float]] = {
    # model: (prompt, completion)
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "local": (0.0, 0.0),
}


@dataclass
class PhaseCost:
    llm_calls: int = 0
    llm_prompt_tokens: int = 0
    llm_completion_tokens: int = 0
    embed_calls: int = 0
    embed_tokens: int = 0
    wall_clock_s: float = 0.0

    def __iadd__(self, other: "PhaseCost") -> "PhaseCost":
        self.llm_calls += other.llm_calls
        self.llm_prompt_tokens += other.llm_prompt_tokens
        self.llm_completion_tokens += other.llm_completion_tokens
        self.embed_calls += other.embed_calls
        self.embed_tokens += other.embed_tokens
        self.wall_clock_s += other.wall_clock_s
        return self

    def dollars(self, model: str = "gpt-4o-mini") -> float:
        p, c = PRICES.get(model, (0.0, 0.0))
        return (
            self.llm_prompt_tokens * p + self.llm_completion_tokens * c
        ) / 1_000_000


@dataclass
class CostLedger:
    """Accumulates cost for one system on one example."""

    write: PhaseCost = field(default_factory=PhaseCost)
    read: PhaseCost = field(default_factory=PhaseCost)

    def _phase(self, phase: Phase) -> PhaseCost:
        return self.write if phase == "write" else self.read

    def add_llm(
        self, phase: Phase, prompt_tokens: int, completion_tokens: int = 0
    ) -> None:
        p = self._phase(phase)
        p.llm_calls += 1
        p.llm_prompt_tokens += prompt_tokens
        p.llm_completion_tokens += completion_tokens

    def add_embed(self, phase: Phase, n_items: int, tokens: int) -> None:
        p = self._phase(phase)
        p.embed_calls += n_items
        p.embed_tokens += tokens

    @contextmanager
    def timer(self, phase: Phase):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._phase(phase).wall_clock_s += time.perf_counter() - t0

    def __iadd__(self, other: "CostLedger") -> "CostLedger":
        self.write += other.write
        self.read += other.read
        return self

    def total_dollars(self, model: str = "gpt-4o-mini") -> float:
        return self.write.dollars(model) + self.read.dollars(model)

    def dollars_at_n_queries(self, n: int, model: str = "gpt-4o-mini") -> float:
        """Write cost is paid once; read cost is paid per query.

        This is the amortisation curve: a system with a huge write path only
        breaks even after enough queries hit the same memory.
        """
        return self.write.dollars(model) + n * self.read.dollars(model)

    def to_dict(self) -> dict:
        return {"write": asdict(self.write), "read": asdict(self.read)}


_ENCODER = None


def count_tokens(text: str) -> int:
    """Token count via tiktoken cl100k_base, with a 4-chars/token fallback.

    `disallowed_special=()` is required, not cosmetic. 21 turns in LongMemEval-S
    contain a literal `assistant<|end_header_id|>` -- chat-template leakage from
    whichever model generated the haystack -- and tiktoken raises on strings
    containing special-token text unless told to encode them as ordinary text.
    Encoding them as ordinary text is the correct behaviour here: they are
    conversation content being measured, not control tokens being sent.

    Without this, any pass over the full 500-question split dies partway. It
    went unnoticed because every stored artifact was built from a stratified
    n=100 subset that happens to miss all 21.
    """
    global _ENCODER
    if _ENCODER is None:
        try:
            import tiktoken

            _ENCODER = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _ENCODER = False
    if _ENCODER is False:
        return max(1, len(text) // 4)
    return len(_ENCODER.encode(text, disallowed_special=()))
