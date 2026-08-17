"""Fusion weighting, and the recency prior built on it.

`rrf_fuse` had no test at all, which is how the recency weight stayed quantised
without anything noticing: `HybridRetriever` expressed a ranking's weight by
repeating it in the list, so `round(w * 2)` integer-rounded the knob and every
weight in (0, 0.75] collapsed to one whole ranking. These pin the continuous
behaviour and the one equivalence a weight of zero has to satisfy.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_mem_eval.cost import CostLedger  # noqa: E402
from llm_mem_eval.data.loader import MemoryUnit  # noqa: E402
from llm_mem_eval.retrieval.base import Hit, rrf_fuse  # noqa: E402
from llm_mem_eval.retrieval.hybrid import HybridRetriever  # noqa: E402

# Relevance agrees on A > B > C; recency reverses it. Any influence the recency
# ranking has therefore shows up as movement, with none of it attributable to
# the two relevance rankings disagreeing with each other.
RELEVANCE: list[Hit] = [(0, 0.9), (1, 0.8), (2, 0.7)]
RECENCY: list[Hit] = [(2, 0.0), (1, 0.0), (0, 0.0)]


def _scores(fused: list[Hit]) -> dict[int, float]:
    return dict(fused)


def test_default_weights_are_uniform() -> None:
    """The no-weights call must stay exactly what it was before weighting."""
    fused = rrf_fuse([RELEVANCE, RELEVANCE])
    expected = {u: 2.0 / (60 + r + 1) for r, (u, _) in enumerate(RELEVANCE)}
    assert _scores(fused) == expected


def test_explicit_unit_weights_match_the_default() -> None:
    assert rrf_fuse([RELEVANCE, RECENCY]) == rrf_fuse(
        [RELEVANCE, RECENCY], weights=[1.0, 1.0]
    )


def test_zero_weight_is_equivalent_to_omitting_the_ranking() -> None:
    """Not merely "contributes 0.0 to the score".

    Multiplying a zero weight through would still insert that ranking's unit ids
    into the score dict at 0.0, so a caller asking to ignore a ranking would get
    its units padding the tail of the result. Only in an unlucky case -- fewer
    than `k` units with a positive score -- does that change what is returned,
    which is exactly the kind of defect that survives a careless test.
    """
    only_units = [(7, 0.5), (8, 0.4)]
    with_zero = rrf_fuse([RELEVANCE, only_units], weights=[1.0, 0.0])
    without = rrf_fuse([RELEVANCE], weights=[1.0])
    assert with_zero == without
    assert 7 not in _scores(with_zero)


def test_weight_is_continuous_below_one_whole_ranking() -> None:
    """The defect this file exists for.

    Under the old integer-repetition scheme both of these weights produced
    `max(1, round(w * 2)) == 1` -- one whole copy of the recency ranking -- so
    they were the same setting. `round(0.5)` is 0 under banker's rounding, and
    the `max` then floors it to one full-strength ranking, meaning no weight
    below 1.0 was expressible at all.
    """
    quarter = _scores(rrf_fuse([RELEVANCE, RECENCY], weights=[1.0, 0.25]))
    half = _scores(rrf_fuse([RELEVANCE, RECENCY], weights=[1.0, 0.5]))
    assert quarter != half
    # Monotone in the weight: the most recent unit gains as the prior strengthens.
    assert half[2] > quarter[2]


def test_a_strong_enough_prior_reorders_the_result() -> None:
    """Continuity is not enough on its own -- the knob has to be able to act."""
    assert [u for u, _ in rrf_fuse([RELEVANCE, RECENCY], weights=[1.0, 0.25])][0] == 0
    assert [u for u, _ in rrf_fuse([RELEVANCE, RECENCY], weights=[1.0, 3.0])][0] == 2


def test_mismatched_weights_length_raises() -> None:
    try:
        rrf_fuse([RELEVANCE, RECENCY], weights=[1.0])
    except ValueError:
        return
    raise AssertionError("a weights/rankings length mismatch must not pass silently")


# -- the recency prior on the real retriever -----------------------------------


class _StubRetriever:
    """Returns a fixed ranking, so the fusion is the only thing under test."""

    def __init__(self, ranking: list[Hit]) -> None:
        self.ranking = ranking

    def index(self, units, ledger, cache_key=None) -> None:
        pass

    def search(self, query: str, k: int, ledger: CostLedger) -> list[Hit]:
        return self.ranking[:k]


def _units() -> list[MemoryUnit]:
    # Unit 2 is the most recent, matching RECENCY above.
    dates = ["2023/01/01 (Sun) 10:00", "2023/02/01 (Wed) 10:00",
             "2023/03/01 (Wed) 10:00"]
    return [
        MemoryUnit(unit_id=i, text=f"unit {i}", session_id=f"s{i}",
                   session_date=d, session_index=i, is_evidence=False)
        for i, d in enumerate(dates)
    ]


def _hybrid(recency_weight: float) -> HybridRetriever:
    h = HybridRetriever.__new__(HybridRetriever)  # no embedding model download
    h.bm25 = _StubRetriever(RELEVANCE)
    h.dense = _StubRetriever(RELEVANCE)
    h.recency_weight = recency_weight
    h.rrf_k = 60
    h.fuse_depth = 100
    h._units = _units()
    return h


def _search(recency_weight: float) -> list[Hit]:
    return _hybrid(recency_weight).search("q", 3, CostLedger(), question_date=None)


def test_hybrid_recency_weight_is_continuous() -> None:
    """The end-to-end version of the quantisation test, through `search`."""
    assert _search(0.25) != _search(0.5)


def test_hybrid_zero_weight_leaves_relevance_untouched() -> None:
    off = _search(0.0)
    assert [u for u, _ in off] == [0, 1, 2]
    assert _scores(off) == _scores(rrf_fuse([RELEVANCE, RELEVANCE]))


def test_hybrid_ignores_question_date_by_construction() -> None:
    """Documented non-use, not an oversight.

    In LongMemEval-S the question is dated at or after every session in its own
    haystack for all 500 questions, so no monotone recency prior can order the
    units differently for a different question date. `audit_question_dates.py`
    checks that on the real data; this pins the code to it, so a future change
    that starts consuming the date has to come with a reason.
    """
    early = _hybrid(1.0).search("q", 3, CostLedger(),
                                question_date="2023/01/15 (Sun) 10:00")
    late = _hybrid(1.0).search("q", 3, CostLedger(),
                               question_date="2026/12/31 (Thu) 10:00")
    assert early == late


# -- the benchmark property the prior rests on ---------------------------------

DATA = Path("data/raw/longmemeval_s")


def test_question_dates_support_day_resolution_only() -> None:
    """Pins `audit_question_dates.py` against the real split.

    Two separate things, because a change to either would invalidate different
    claims: that a recency prior may ignore `question_date` at day resolution,
    and that it must not consume it at minute resolution.
    """
    if not DATA.exists():
        print(f"  skip question-date audit ({DATA} not downloaded)")
        return
    from scripts.audit_question_dates import audit

    from llm_mem_eval.data.loader import load_examples

    r = audit(load_examples(DATA))
    day, minute = r["day_resolution"], r["minute_resolution"]

    # Day resolution: the prior is free to ignore the question date.
    assert day["orderings_identical_everywhere"], day["recency_vs_proximity_ordering"]
    assert day["question_position"].get("at least one session after the question") is None
    assert day["exponential_origin_max_relative_deviation"] < 1e-9

    # Minute resolution: it is not free to consume it. A cutoff at the question
    # timestamp makes these questions unanswerable however good the retriever is.
    assert minute["questions_with_post_question_sessions"] == 76
    assert minute["questions_with_post_question_evidence"] == 43
    assert minute["questions_with_all_evidence_post_question"] == 21
    # Never a full day out, which is exactly why day truncation hides all of it.
    assert minute["max_overshoot_days"] < 1.0
    print(f"  ok  day resolution inert; {minute['questions_with_post_question_evidence']}"
          f" questions have post-question gold evidence")


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all passed")
