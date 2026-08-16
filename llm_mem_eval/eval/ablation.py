"""Memory-lift attribution: how much of a memory system's accuracy is memory?

Every memory paper reports a headline accuracy. None of them report what the
same model scores with no memory at all, or with the same number of tokens
drawn at random. Without those, the headline is an upper bound on the system's
contribution and nothing more -- a question the model can answer from world
knowledge, or one that leaks its answer in the phrasing, is credited to memory
by default.

This module computes the difference against the strongest available control:

    memory_lift = acc(system) - max(acc(closed_book), acc(matched-budget noise))

Two controls, because they fail differently. `closed_book` catches questions
answerable without any memory. `random`/`recency` at the same k catch the case
where the gain came from spending tokens rather than from spending them well.
The system has to beat both before any accuracy is attributable to retrieval.

Paired bootstrap throughout: the arms are evaluated on identical questions, so
the per-question pairing is real information and ignoring it produces intervals
roughly sqrt(2)x too wide.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

CONTROL_ARMS = ("none", "random", "recency")


@dataclass
class ArmResult:
    """One evaluated arm, keyed per question so arms can be paired."""

    name: str
    model: str
    retriever: str
    accuracy: float
    read_tokens_per_query: float
    # question_id -> True/False; questions the grader abstained on are absent
    graded: dict[str, bool] = field(default_factory=dict)
    qtype: dict[str, str] = field(default_factory=dict)

    @property
    def is_control(self) -> bool:
        return self.retriever in CONTROL_ARMS


def arm_from_payload(payload: dict, name: str | None = None) -> ArmResult:
    cfg = payload.get("config", {})
    graded, qtype = {}, {}
    for r in payload.get("records", []):
        qid = r["question_id"]
        qtype[qid] = r.get("question_type", "?")
        if r.get("deterministic") is not None:
            graded[qid] = bool(r["deterministic"])
    return ArmResult(
        name=name or cfg.get("tag") or f"{cfg.get('retriever')}",
        model=str(cfg.get("answer_backend", "?")),
        retriever=str(cfg.get("retriever", "?")),
        accuracy=float(payload.get("accuracy", 0.0)),
        read_tokens_per_query=float(payload.get("read_tokens_per_query", 0.0)),
        graded=graded,
        qtype=qtype,
    )


def paired_ids(a: ArmResult, b: ArmResult) -> list[str]:
    """Questions both arms actually graded. Comparing arms on different
    question sets is the easiest way to manufacture a lift that isn't there."""
    return sorted(set(a.graded) & set(b.graded))


def contingency(system: ArmResult, control: ArmResult) -> dict[str, int]:
    """McNemar table. `sys_only` and `ctl_only` are the discordant pairs --
    the only ones carrying information about which arm is better."""
    ids = paired_ids(system, control)
    both = sys_only = ctl_only = neither = 0
    for q in ids:
        s, c = system.graded[q], control.graded[q]
        if s and c:
            both += 1
        elif s:
            sys_only += 1
        elif c:
            ctl_only += 1
        else:
            neither += 1
    return {"n": len(ids), "both": both, "system_only": sys_only,
            "control_only": ctl_only, "neither": neither}


def mcnemar_p(system: ArmResult, control: ArmResult) -> float:
    """Exact two-sided binomial test on the discordant pairs.

    Implemented directly rather than pulled from scipy: the repo has no scipy
    dependency, and an exact test on <=100 discordant pairs is a few lines.
    """
    t = contingency(system, control)
    b, c = t["system_only"], t["control_only"]
    n = b + c
    if n == 0:
        return 1.0

    def comb(nn: int, kk: int) -> int:
        r = 1
        for i in range(kk):
            r = r * (nn - i) // (i + 1)
        return r

    tail = min(b, c)
    p = sum(comb(n, i) for i in range(tail + 1)) / (2 ** n)
    return min(1.0, 2 * p)


def paired_bootstrap_ci(
    system: ArmResult,
    control: ArmResult,
    n_boot: int = 10000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """(point estimate, lo, hi) for acc(system) - acc(control), resampling
    questions rather than arms so the pairing is preserved."""
    ids = paired_ids(system, control)
    if not ids:
        return (0.0, 0.0, 0.0)
    diffs = [int(system.graded[q]) - int(control.graded[q]) for q in ids]
    point = sum(diffs) / len(diffs)

    rng = random.Random(seed)
    n = len(diffs)
    samples = []
    for _ in range(n_boot):
        s = sum(diffs[rng.randrange(n)] for _ in range(n))
        samples.append(s / n)
    samples.sort()
    lo = samples[int((alpha / 2) * n_boot)]
    hi = samples[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return (point, lo, hi)


@dataclass
class LiftReport:
    system: str
    model: str
    system_accuracy: float
    best_control: str
    control_accuracy: float
    lift: float
    ci_lo: float
    ci_hi: float
    p_value: float
    attributable_fraction: float
    contingency: dict[str, int]
    per_type: dict[str, dict[str, float]]

    @property
    def significant(self) -> bool:
        """Lift is distinguishable from zero at the 5% level AND the interval
        excludes it. Both, because a p-value alone says nothing about size."""
        return self.p_value < 0.05 and self.ci_lo > 0


def compute_lift(
    system: ArmResult, controls: list[ArmResult], seed: int = 0
) -> LiftReport:
    """Score `system` against its strongest control.

    Strongest, not average: a memory system that beats random but loses to
    'just keep the last 10 turns' has not demonstrated retrieval value, and
    averaging the controls would hide that.
    """
    usable = [c for c in controls if paired_ids(system, c)]
    if not usable:
        raise ValueError(f"no control shares graded questions with {system.name}")

    def ctl_acc(c: ArmResult) -> float:
        ids = paired_ids(system, c)
        return sum(c.graded[q] for q in ids) / len(ids)

    best = max(usable, key=ctl_acc)
    ids = paired_ids(system, best)
    sys_acc = sum(system.graded[q] for q in ids) / len(ids)
    ctl = ctl_acc(best)
    point, lo, hi = paired_bootstrap_ci(system, best, seed=seed)

    per_type: dict[str, dict[str, float]] = {}
    for q in ids:
        t = system.qtype.get(q, "?")
        d = per_type.setdefault(t, {"n": 0, "system": 0, "control": 0})
        d["n"] += 1
        d["system"] += int(system.graded[q])
        d["control"] += int(best.graded[q])
    for t, d in per_type.items():
        d["system_acc"] = d["system"] / d["n"]
        d["control_acc"] = d["control"] / d["n"]
        d["lift"] = d["system_acc"] - d["control_acc"]

    return LiftReport(
        system=system.name,
        model=system.model,
        system_accuracy=sys_acc,
        best_control=best.retriever,
        control_accuracy=ctl,
        lift=point,
        ci_lo=lo,
        ci_hi=hi,
        p_value=mcnemar_p(system, best),
        # What share of the headline number survives the control. A system at
        # 0.58 whose closed-book control scores 0.30 is a 0.48-attributable
        # system, not a 0.58 one.
        attributable_fraction=(point / sys_acc) if sys_acc > 0 else 0.0,
        contingency=contingency(system, best),
        per_type=per_type,
    )
