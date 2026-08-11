"""Cost instrumentation, split by write path vs read path.

This is the point of the project. Published memory systems report read-path
tokens per query and omit the write path, where the cost actually lives (Mem0's
memory construction runs ~1.5M tokens for one LoCoMo instance -- a figure only
RecMem's Table 8 reports, since Mem0's own paper is silent on it). We account
for both, separately, so total cost can be plotted against query volume.
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
    """Token count via tiktoken cl100k_base, with a 4-chars/token fallback."""
    global _ENCODER
    if _ENCODER is None:
        try:
            import tiktoken

            _ENCODER = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _ENCODER = False
    if _ENCODER is False:
        return max(1, len(text) // 4)
    return len(_ENCODER.encode(text))
