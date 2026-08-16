"""Tests for memory-lift attribution.

The statistics here decide whether a reported lift is real, so they are
checked against cases whose answers are known analytically rather than
against whatever the implementation happened to produce.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_mem_eval.eval.ablation import (
    ArmResult,
    compute_lift,
    contingency,
    mcnemar_p,
    paired_bootstrap_ci,
    paired_ids,
)


def arm(name, retriever, correct, qtypes=None, model="m"):
    """correct: dict qid -> bool"""
    return ArmResult(
        name=name, model=model, retriever=retriever,
        accuracy=sum(correct.values()) / max(len(correct), 1),
        read_tokens_per_query=1000.0,
        graded=dict(correct),
        qtype=qtypes or {q: "t" for q in correct},
    )


def test_pairing_uses_only_shared_questions():
    a = arm("a", "hybrid", {"q1": True, "q2": True, "q3": True})
    b = arm("b", "none", {"q1": False, "q4": False})
    assert paired_ids(a, b) == ["q1"]


def test_contingency_counts_discordant_pairs():
    sysm = arm("s", "hybrid", {"a": True, "b": True, "c": False, "d": False})
    ctl = arm("c", "none", {"a": True, "b": False, "c": True, "d": False})
    t = contingency(sysm, ctl)
    assert t == {"n": 4, "both": 1, "system_only": 1,
                 "control_only": 1, "neither": 1}


def test_mcnemar_no_discordant_pairs_is_p_one():
    same = {"a": True, "b": False}
    assert mcnemar_p(arm("s", "hybrid", same), arm("c", "none", same)) == 1.0


def test_mcnemar_matches_exact_binomial():
    # 10 discordant pairs, all favouring the system. Two-sided exact p is
    # 2 * (1/2)^10 = 0.001953125.
    sysm = arm("s", "hybrid", {f"q{i}": True for i in range(10)})
    ctl = arm("c", "none", {f"q{i}": False for i in range(10)})
    assert abs(mcnemar_p(sysm, ctl) - 2 * (0.5 ** 10)) < 1e-12

    # 5 vs 1 discordant: 2 * sum(C(6,i) for i in 0..1) / 2^6 = 2*7/64
    s = {f"a{i}": True for i in range(5)} | {"b0": False}
    c = {f"a{i}": False for i in range(5)} | {"b0": True}
    assert abs(mcnemar_p(arm("s", "hybrid", s), arm("c", "none", c))
               - 2 * 7 / 64) < 1e-12


def test_bootstrap_ci_brackets_point_estimate():
    sysm = arm("s", "hybrid", {f"q{i}": i % 2 == 0 for i in range(100)})
    ctl = arm("c", "none", {f"q{i}": i % 4 == 0 for i in range(100)})
    point, lo, hi = paired_bootstrap_ci(sysm, ctl, n_boot=2000, seed=0)
    assert abs(point - 0.25) < 1e-9  # 50% vs 25%
    assert lo < point < hi


def test_bootstrap_ci_includes_zero_when_arms_identical():
    same = {f"q{i}": i % 3 == 0 for i in range(60)}
    point, lo, hi = paired_bootstrap_ci(
        arm("s", "hybrid", same), arm("c", "none", same), n_boot=2000
    )
    assert point == 0.0 and lo == 0.0 and hi == 0.0


def test_lift_uses_strongest_control_not_the_weakest():
    """A system that beats random but loses to recency has not shown value.
    Averaging controls would hide that; taking the max must not."""
    sysm = arm("s", "hybrid", {f"q{i}": i < 60 for i in range(100)})
    weak = arm("w", "random", {f"q{i}": i < 10 for i in range(100)})
    strong = arm("r", "recency", {f"q{i}": i < 80 for i in range(100)})
    r = compute_lift(sysm, [weak, strong])
    assert r.best_control == "recency"
    assert r.lift < 0


def test_attributable_fraction_discounts_the_headline():
    # system 0.60, control 0.30 -> half the headline is attributable
    sysm = arm("s", "hybrid", {f"q{i}": i < 60 for i in range(100)})
    ctl = arm("c", "none", {f"q{i}": i < 30 for i in range(100)})
    r = compute_lift(sysm, [ctl])
    assert abs(r.system_accuracy - 0.60) < 1e-9
    assert abs(r.lift - 0.30) < 1e-9
    assert abs(r.attributable_fraction - 0.5) < 1e-9


def test_significance_requires_both_p_and_ci():
    # A single discordant pair cannot be significant however it falls.
    sysm = arm("s", "hybrid", {"a": True, "b": True})
    ctl = arm("c", "none", {"a": True, "b": False})
    assert not compute_lift(sysm, [ctl]).significant


def test_per_type_breakdown_partitions_the_questions():
    qt = {f"q{i}": ("even" if i % 2 == 0 else "odd") for i in range(20)}
    sysm = arm("s", "hybrid", {f"q{i}": i % 2 == 0 for i in range(20)}, qt)
    ctl = arm("c", "none", {f"q{i}": False for i in range(20)}, qt)
    r = compute_lift(sysm, [ctl])
    assert sum(d["n"] for d in r.per_type.values()) == 20
    assert r.per_type["even"]["lift"] == 1.0
    assert r.per_type["odd"]["lift"] == 0.0
