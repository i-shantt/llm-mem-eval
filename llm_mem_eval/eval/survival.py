"""Answer survival: what a write path keeps, measured before any retrieval.

A memory system's write path turns a conversation into a store. Survival asks
the narrowest possible question about that store:

    is the answer still in there at all?

It ignores retrieval and generation entirely, so it is a *ceiling* on both. A
store that has dropped the answer cannot be rescued by a better retriever or a
bigger reader. This is deliberately a smaller claim than accuracy, and it is the
one thing about a write path that can be measured without a judge.

Three definitions, because they support different claims:

    survival_record   some single record contains the gold answer      PRIMARY
    survival_soft     some record contains every gold token, unordered
    survival_union    the concatenated store contains it               diagnostic

`survival_record` is primary. `survival_soft` is the variant most generous to an
extraction-based store, so any claimed survival gap should hold under both.
`survival_union` is a diagnostic only: over a store of ~10^2 records it starts to
approximate "is this string anywhere", which is a store-size proxy rather than a
memory measurement. On LongMemEval it happens to agree with `survival_record` on
every question measured here -- 0 disagreements over 2,576 scored records -- but
that is a property of this benchmark, not of the definitions. It is also not
quite universal: `scripts/audit_benchmark.py` finds exactly one gold answer
assembled across evidence turns rather than present in any single one, and it is
a temporal-reasoning question, outside the three span types scored below.

Two things this metric is NOT, both of which must be stated wherever it is used:

1. It is biased AGAINST extraction. "User is allergic to peanuts" survives;
   "User discussed dietary restrictions" does not, even where the latter might
   have been enough for the reader. `survival_soft` narrows that gap and does
   not close it. Naming the direction of the bias matters more than any control.

2. Containment does not check that the record ASSERTS the answer. A record
   saying "not Page Turners, actually Chapter One" contains "Page Turners" and
   counts as survived. That is the two-alternative false accept the grader audit
   already documents, relocated to the write path.

The chance floor is not negligible and must always be reported. A one-token gold
like "3" or "Target" appears somewhere in a 104K-token store most of the time by
accident. `placebo_null` measures that per question by re-running survival
against gold answers borrowed from *other* questions, and `chance_corrected`
reports (s - z) / (1 - z). Raw survival is never the headline.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from math import sqrt

from llm_mem_eval.cost import count_tokens
from llm_mem_eval.data.loader import Example, MemoryUnit

# `contains_answer` is imported rather than reimplemented so survival and
# end-to-end accuracy are read off the same ruler. eval/grade.py is not modified
# by this module: its measured false-accept rate is a CI gate.
from llm_mem_eval.eval.grade import contains_answer, normalize_tokens

BUCKETS = ("1", "2-3", "4+")

# The question types whose answers are spans to be kept, as opposed to values to
# be computed. scripts/audit_benchmark.py measures the split: these three have
# their gold answer verbatim in the labelled evidence 0.70-0.91 of the time,
# while temporal-reasoning and multi-session sit at 0.27 and 0.11 because their
# answers are date arithmetic and cross-session counts.
#
# Excluding those two is necessary -- survival cannot measure whether a store
# preserved an answer that was never text -- and it is also the sharpest
# limitation of this metric, because they are exactly the types where an
# extraction write path should lose the most. Stated in the README, not buried.
CLEAN_TYPES = (
    "single-session-user",
    "single-session-assistant",
    "knowledge-update",
)

# Golds of one normalised token match by chance often enough that a rate
# computed over them says more about the store's size than its contents. The
# primary analysis excludes them; see `PRIMARY_MIN_GOLD_TOKENS`.
PRIMARY_MIN_GOLD_TOKENS = 2


def gold_len_bucket(gold: str) -> str:
    n = len(normalize_tokens(gold))
    return "1" if n <= 1 else ("2-3" if n <= 3 else "4+")


def in_primary_subset(gold: str) -> bool:
    return len(normalize_tokens(gold)) >= PRIMARY_MIN_GOLD_TOKENS


def eligible_examples(examples: list[Example]) -> list[Example]:
    """The questions survival can be measured on at all.

    NOT filtered on whether the verbatim store happens to preserve the answer.
    Conditioning the subset on the control's own outcome would pin the verbatim
    arm at 1.000 by construction and make every comparison against it
    meaningless.
    """
    from llm_mem_eval.eval.grade import gold_signals_abstention, is_extractive

    return [
        ex for ex in examples
        if not ex.is_abstention
        and ex.question_type in CLEAN_TYPES
        and is_extractive(ex.answer)
        and not gold_signals_abstention(ex.answer)
    ]


def covers_gold_tokens(text: str, gold: str) -> bool:
    """Every gold token present in `text`, order and adjacency ignored.

    The paraphrase-tolerant variant. Deliberately lives here and not in
    eval/grade.py: the grader's false-accept rate is measured and asserted by
    CI, and adding a looser predicate there would move a gate.
    """
    g = set(normalize_tokens(gold))
    return bool(g) and g <= set(normalize_tokens(text))


def survives_record(units: list[MemoryUnit], gold: str,
                    question: str | None = None) -> bool:
    return any(contains_answer(u.text, gold, question) for u in units)


def survives_soft(units: list[MemoryUnit], gold: str) -> bool:
    return any(covers_gold_tokens(u.text, gold) for u in units)


def survives_union(units: list[MemoryUnit], gold: str,
                   question: str | None = None) -> bool:
    return contains_answer("\n".join(u.text for u in units), gold, question)


# --------------------------------------------------------------------------
# Chance floor
# --------------------------------------------------------------------------

def build_placebo_pool(examples: list[Example]) -> dict[tuple[str, str], list[str]]:
    """Gold answers grouped by (question_type, gold length bucket).

    Placebos are drawn from the same type and length so the null is matched on
    the two things that drive spurious containment. Drawing uniformly from all
    golds would understate the floor for short answers and overstate it for long
    ones.
    """
    pool: dict[tuple[str, str], list[str]] = defaultdict(list)
    for ex in examples:
        pool[(ex.question_type, gold_len_bucket(ex.answer))].append(str(ex.answer))
    return dict(pool)


def sample_placebos(pool: dict[tuple[str, str], list[str]], ex: Example,
                    m: int, rng: random.Random) -> list[str]:
    """`m` gold answers from other questions, matched on type and length."""
    key = (ex.question_type, gold_len_bucket(ex.answer))
    candidates = [g for g in pool.get(key, []) if g != str(ex.answer)]
    if not candidates:
        # Fall back to same type, any length, rather than returning nothing:
        # a missing null would silently read as a null of zero.
        candidates = [
            g for k, gs in pool.items() if k[0] == ex.question_type
            for g in gs if g != str(ex.answer)
        ]
    if not candidates:
        return []
    return rng.sample(candidates, min(m, len(candidates)))


# --------------------------------------------------------------------------
# Per-question outcome
# --------------------------------------------------------------------------

@dataclass
class SurvivalOutcome:
    question_id: str
    question_type: str
    gold_len_bucket: str
    in_primary: bool
    record: bool
    soft: bool
    union: bool
    null_record: float
    null_soft: float
    null_union: float
    n_records: int
    store_tokens: int

    def to_dict(self) -> dict:
        # `deterministic` is a deliberate alias for `record`: it lets
        # eval.ablation.arm_from_payload consume a survival payload unchanged,
        # so the paired statistics are the same code path as everywhere else.
        return {
            "question_id": self.question_id,
            "question_type": self.question_type,
            "deterministic": self.record,
            "gold_len_bucket": self.gold_len_bucket,
            "in_primary": self.in_primary,
            "survival_record": self.record,
            "survival_soft": self.soft,
            "survival_union": self.union,
            "null_record": self.null_record,
            "null_soft": self.null_soft,
            "null_union": self.null_union,
            "n_records": self.n_records,
            "store_tokens": self.store_tokens,
        }


def score_store(ex: Example, units: list[MemoryUnit],
                placebos: list[str]) -> SurvivalOutcome:
    """Survival plus its chance floor for one question against one store.

    The store's derived forms -- the concatenation and the per-record token sets
    -- are built once and reused across the real gold and all placebos. Rebuilt
    per gold, the union check alone would re-join a 104K-token string eleven
    times per question.
    """
    gold, q = str(ex.answer), ex.question
    joined = "\n".join(u.text for u in units)
    token_sets = [set(normalize_tokens(u.text)) for u in units]

    def _soft(g: str) -> bool:
        gt = set(normalize_tokens(g))
        return bool(gt) and any(gt <= ts for ts in token_sets)

    def _mean(vals: list[bool]) -> float:
        # No placebos available is reported as an unmeasured floor of 0.0, which
        # happens only for a type/bucket cell of size one.
        return sum(vals) / len(vals) if vals else 0.0

    return SurvivalOutcome(
        question_id=ex.question_id,
        question_type=ex.question_type,
        gold_len_bucket=gold_len_bucket(gold),
        in_primary=in_primary_subset(gold),
        record=survives_record(units, gold, q),
        soft=_soft(gold),
        union=contains_answer(joined, gold, q),
        null_record=_mean([survives_record(units, p, q) for p in placebos]),
        null_soft=_mean([_soft(p) for p in placebos]),
        null_union=_mean([contains_answer(joined, p, q) for p in placebos]),
        n_records=len(units),
        store_tokens=sum(count_tokens(u.text) for u in units),
    )


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion.

    Wilson rather than normal-approximation because these rates sit near 0 and 1
    where the normal interval runs outside [0, 1], and closed form because the
    repo deliberately carries no scipy dependency.
    """
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def chance_corrected(survival: float, null: float) -> float:
    """(s - z) / (1 - z): the share of the available headroom actually used.

    Undefined when the chance floor is 1.0 -- every placebo matched, so the test
    has no discriminating power at all. Reported as nan rather than clipped,
    because a silent 0.0 there would read as a real measurement.
    """
    if null >= 1.0:
        return float("nan")
    return (survival - null) / (1.0 - null)


def bootstrap_corrected(outcomes: list[SurvivalOutcome], variant: str = "record",
                        n_boot: int = 10000, seed: int = 0,
                        alpha: float = 0.05) -> tuple[float, float, float]:
    """CI for chance-corrected survival, resampling questions.

    The whole corrected statistic is recomputed inside each replicate. Bootstrapping
    the numerator against a fixed denominator would treat the chance floor as
    known, and it is estimated from the same questions.
    """
    s_key, z_key = variant, f"null_{variant}"
    pairs = [(float(getattr(o, s_key)), getattr(o, z_key)) for o in outcomes]
    if not pairs:
        return (float("nan"), float("nan"), float("nan"))

    def _stat(sample: list[tuple[float, float]]) -> float:
        s = sum(p[0] for p in sample) / len(sample)
        z = sum(p[1] for p in sample) / len(sample)
        return chance_corrected(s, z)

    point = _stat(pairs)
    rng = random.Random(seed)
    n = len(pairs)
    reps = []
    for _ in range(n_boot):
        draw = [pairs[rng.randrange(n)] for _ in range(n)]
        v = _stat(draw)
        if v == v:  # drop nan replicates (chance floor hit 1.0)
            reps.append(v)
    if not reps:
        return (point, float("nan"), float("nan"))
    reps.sort()
    lo = reps[int(alpha / 2 * len(reps))]
    hi = reps[min(len(reps) - 1, int((1 - alpha / 2) * len(reps)))]
    return (point, lo, hi)


def _mean(vals) -> float:
    vals = list(vals)
    return sum(vals) / len(vals) if vals else float("nan")


def summarise(outcomes: list[SurvivalOutcome], variant: str = "record") -> dict:
    """Raw survival, chance floor, corrected value and both intervals."""
    if not outcomes:
        return {"n": 0}
    s = _mean(float(getattr(o, variant)) for o in outcomes)
    z = _mean(getattr(o, f"null_{variant}") for o in outcomes)
    k = sum(bool(getattr(o, variant)) for o in outcomes)
    n = len(outcomes)
    point, lo, hi = bootstrap_corrected(outcomes, variant)
    w_lo, w_hi = wilson(k, n)
    return {
        "n": n,
        "survival": s,
        "survival_ci95": [w_lo, w_hi],
        "null": z,
        "chance_corrected": point,
        "chance_corrected_ci95": [lo, hi],
    }


def report(outcomes: list[SurvivalOutcome]) -> dict:
    """Full survival payload for one store."""
    primary = [o for o in outcomes if o.in_primary]
    by_bucket = defaultdict(list)
    by_type = defaultdict(list)
    for o in outcomes:
        by_bucket[o.gold_len_bucket].append(o)
    for o in primary:
        by_type[o.question_type].append(o)

    return {
        "n_all": len(outcomes),
        "n_primary": len(primary),
        "primary_subset": (
            f"golds with >= {PRIMARY_MIN_GOLD_TOKENS} normalized tokens; "
            "one-token golds are excluded from the headline because their "
            "chance floor is high enough to dominate the rate"
        ),
        "primary": {
            "record": summarise(primary, "record"),
            "soft": summarise(primary, "soft"),
            "union": summarise(primary, "union"),
        },
        "all_questions": {
            "record": summarise(outcomes, "record"),
            "soft": summarise(outcomes, "soft"),
        },
        "by_gold_len_bucket": {
            b: {"record": summarise(by_bucket[b], "record"),
                "soft": summarise(by_bucket[b], "soft")}
            for b in BUCKETS if by_bucket[b]
        },
        # Descriptive only: n is 30-70 per type before any conditioning, well
        # below what these intervals can resolve. Counts are printed beside
        # every rate so that is visible rather than implied.
        "by_question_type_primary": {
            t: summarise(rs, "record") for t, rs in sorted(by_type.items())
        },
        "store_stats": {
            "records_per_store_mean": _mean(o.n_records for o in outcomes),
            "tokens_per_store_mean": _mean(o.store_tokens for o in outcomes),
            "tokens_per_record_mean": _mean(
                o.store_tokens / o.n_records for o in outcomes if o.n_records
            ),
        },
    }
