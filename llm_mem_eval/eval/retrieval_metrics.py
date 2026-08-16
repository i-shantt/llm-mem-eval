"""Judge-free retrieval metrics, using LongMemEval's per-turn `has_answer` labels.

These need no LLM and no API budget, which makes them the load-bearing metrics
for this project. End-to-end answer quality is graded without a judge too --
see `llm_mem_eval/eval/grade.py`, whose error rates are measured by
`llm_mem_eval/eval/grader_audit.py` rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean

from ..data.loader import MemoryUnit
from ..retrieval.base import Hit


@dataclass
class RetrievalResult:
    question_id: str
    question_type: str
    is_abstention: bool
    n_evidence: int
    hits: list[Hit]
    evidence_ranks: list[int] = field(default_factory=list)

    def recall_at(self, k: int) -> float | None:
        """Fraction of evidence units retrieved in the top k."""
        if self.n_evidence == 0:
            return None
        found = sum(1 for r in self.evidence_ranks if r < k)
        return found / self.n_evidence

    def any_hit_at(self, k: int) -> bool | None:
        """Was *any* evidence retrieved? The precondition for answerability."""
        if self.n_evidence == 0:
            return None
        return any(r < k for r in self.evidence_ranks)

    def reciprocal_rank(self) -> float | None:
        if self.n_evidence == 0:
            return None
        if not self.evidence_ranks:
            return 0.0
        return 1.0 / (min(self.evidence_ranks) + 1)


def score_example(
    question_id: str,
    question_type: str,
    is_abstention: bool,
    units: list[MemoryUnit],
    hits: list[Hit],
) -> RetrievalResult:
    evidence_ids = {u.unit_id for u in units if u.is_evidence}
    ranks = [
        rank for rank, (uid, _) in enumerate(hits) if uid in evidence_ids
    ]
    return RetrievalResult(
        question_id=question_id,
        question_type=question_type,
        is_abstention=is_abstention,
        n_evidence=len(evidence_ids),
        hits=hits,
        evidence_ranks=ranks,
    )


def aggregate(
    results: list[RetrievalResult], ks: tuple[int, ...] = (1, 3, 5, 10, 20)
) -> dict:
    """Aggregate over examples. Zero-evidence examples (abstention cases) are
    excluded from recall and reported separately rather than scored as 0 or 1."""
    scorable = [r for r in results if r.n_evidence > 0]
    out: dict = {
        "n_total": len(results),
        "n_scorable": len(scorable),
        "n_zero_evidence": len(results) - len(scorable),
    }
    if not scorable:
        return out

    for k in ks:
        out[f"recall@{k}"] = mean(r.recall_at(k) for r in scorable)
        out[f"any_hit@{k}"] = mean(float(r.any_hit_at(k)) for r in scorable)
    out["mrr"] = mean(r.reciprocal_rank() for r in scorable)

    by_type: dict[str, dict] = {}
    types = sorted({r.question_type for r in scorable})
    for qtype in types:
        group = [r for r in scorable if r.question_type == qtype]
        by_type[qtype] = {
            "n": len(group),
            "any_hit@10": mean(float(r.any_hit_at(10)) for r in group),
            "recall@10": mean(r.recall_at(10) for r in group),
            "mrr": mean(r.reciprocal_rank() for r in group),
        }
    out["by_question_type"] = by_type
    return out
