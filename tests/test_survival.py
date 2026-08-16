"""Tests for the answer-survival metric.

The things worth pinning here are the ones that would silently produce a
plausible-looking but meaningless number: the chance floor not firing, the
correction being computed the wrong way round, and the primary subset being
defined by something other than gold length.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memllm.data.loader import Example, MemoryUnit, Turn  # noqa: E402
from memllm.eval.survival import (  # noqa: E402
    CLEAN_TYPES,
    bootstrap_corrected,
    build_placebo_pool,
    chance_corrected,
    covers_gold_tokens,
    eligible_examples,
    gold_len_bucket,
    in_primary_subset,
    sample_placebos,
    score_store,
    summarise,
    survives_record,
    survives_soft,
    survives_union,
    wilson,
)

DATE = "2023/01/01 (Sun) 10:00"


def _units(*texts: str) -> list[MemoryUnit]:
    return [MemoryUnit(i, t, "s1", DATE, 0, False) for i, t in enumerate(texts)]


def _ex(answer: str, qtype: str = "single-session-user",
        qid: str = "q1") -> Example:
    return Example(qid, qtype, "What am I allergic to?", answer, DATE,
                   [Turn("user", "x", "s1", DATE, 0, 0, True)])


def test_record_and_union_differ_on_a_split_answer() -> None:
    """The distinction the two definitions exist to draw."""
    split = _units("I flew JetBlue and", "then Delta on the return leg")
    gold = "JetBlue and then Delta"
    assert not survives_record(split, gold)
    assert survives_union(split, gold)

    whole = _units("I flew JetBlue and then Delta on the return leg")
    assert survives_record(whole, gold)
    assert survives_union(whole, gold)


def test_soft_accepts_reordering_that_strict_rejects() -> None:
    units = _units("The user is allergic to peanuts, severely.")
    assert covers_gold_tokens(units[0].text, "peanuts allergic")
    assert not survives_record(units, "allergic peanuts severely to user")
    assert survives_soft(units, "allergic peanuts severely to user")


def test_soft_requires_every_gold_token() -> None:
    assert not covers_gold_tokens("allergic to peanuts", "allergic to cashews")
    assert not covers_gold_tokens("", "anything")


def test_placebo_null_is_zero_on_a_clean_store_and_high_on_a_stuffed_one() -> None:
    """The control has to be able to fire, and has to be able to not fire."""
    clean = _units("The weather was pleasant.", "We discussed train timetables.")
    stuffed = _units("peanuts", "cashews", "almonds", "walnuts")
    placebos = ["cashews", "almonds", "walnuts"]

    ex = _ex("peanuts")
    quiet = score_store(ex, clean, placebos)
    loud = score_store(ex, stuffed, placebos)

    assert quiet.null_record == 0.0
    assert loud.null_record == 1.0
    # And the correction reflects it: a store that contains every candidate
    # answer has used none of the available headroom.
    assert chance_corrected(loud.record, loud.null_record) != 1.0


def test_chance_correction_direction_and_edges() -> None:
    assert chance_corrected(1.0, 0.0) == 1.0
    assert chance_corrected(0.5, 0.0) == 0.5
    # Half the headroom above a floor of 0.6 is 0.8, not 0.5.
    assert abs(chance_corrected(0.8, 0.6) - 0.5) < 1e-9
    # Survival at the floor means nothing was learned.
    assert chance_corrected(0.3, 0.3) == 0.0
    # Below the floor is negative, not clipped -- a clipped 0.0 would read as
    # "no signal" when it actually means the store is worse than chance.
    assert chance_corrected(0.2, 0.4) < 0
    # Undefined rather than silently zero when the test has no power at all.
    assert chance_corrected(0.9, 1.0) != chance_corrected(0.9, 1.0)


def test_wilson_matches_hand_computed_values() -> None:
    lo, hi = wilson(0, 10)
    assert lo == 0.0 and 0.27 < hi < 0.31          # one-sided at zero successes
    lo, hi = wilson(10, 10)
    assert 0.69 < lo < 0.73 and hi == 1.0
    lo, hi = wilson(5, 10)
    assert 0.23 < lo < 0.25 and 0.75 < hi < 0.77   # symmetric at p = 0.5
    assert wilson(0, 0) == (0.0, 0.0)              # no division by zero
    lo, hi = wilson(92, 110)                       # near the real primary n
    assert lo > 0.75 and hi < 0.90


def test_gold_length_buckets_and_primary_subset() -> None:
    assert gold_len_bucket("3") == "1"
    assert gold_len_bucket("Target") == "1"
    assert gold_len_bucket("June 3rd") == "2-3"
    assert gold_len_bucket("University of California, Los Angeles") == "4+"
    # The primary subset is defined by gold length only -- never by whether the
    # control store happened to preserve the answer, which would pin the
    # control at 1.000 by construction.
    assert not in_primary_subset("Target")
    assert in_primary_subset("June 3rd")


def test_placebo_pool_matches_type_and_length_and_excludes_self() -> None:
    examples = [_ex("peanuts", qid="a"), _ex("cashews", qid="b"),
                _ex("almonds", qid="c"),
                _ex("a much longer gold answer here", qid="d")]
    pool = build_placebo_pool(examples)
    rng = random.Random(0)
    drawn = sample_placebos(pool, examples[0], m=5, rng=rng)
    assert "peanuts" not in drawn, "a question must not be its own placebo"
    assert set(drawn) <= {"cashews", "almonds"}, "length bucket not respected"


def test_eligible_subset_excludes_the_synthesis_types() -> None:
    examples = [
        _ex("peanuts", "single-session-user", "a"),
        _ex("peanuts", "knowledge-update", "b"),
        _ex("peanuts", "temporal-reasoning", "c"),
        _ex("peanuts", "multi-session", "d"),
        _ex("peanuts", "single-session-user", "e_abs"),
    ]
    got = {e.question_id for e in eligible_examples(examples)}
    assert got == {"a", "b"}
    assert "temporal-reasoning" not in CLEAN_TYPES
    assert "multi-session" not in CLEAN_TYPES


def test_bootstrap_brackets_the_point_estimate() -> None:
    ex = _ex("peanuts")
    hits = [score_store(ex, _units("allergic to peanuts"), ["cashews"])
            for _ in range(30)]
    misses = [score_store(ex, _units("nothing relevant"), ["cashews"])
              for _ in range(10)]
    point, lo, hi = bootstrap_corrected(hits + misses, "record", n_boot=400)
    assert lo <= point <= hi
    assert 0.0 < point < 1.0


def test_summarise_reports_floor_beside_every_rate() -> None:
    ex = _ex("peanuts")
    out = summarise(
        [score_store(ex, _units("allergic to peanuts"), ["cashews"])] * 8,
        "record",
    )
    # Raw survival must never be reported without its floor and correction.
    for key in ("survival", "null", "chance_corrected", "survival_ci95",
                "chance_corrected_ci95"):
        assert key in out, f"{key} missing from the summary"
    assert out["n"] == 8


if __name__ == "__main__":
    print("survival tests")
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")
    print("all passed")
