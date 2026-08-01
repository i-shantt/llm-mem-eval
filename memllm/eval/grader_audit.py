"""Audit any grader using cases whose correct verdict is known by construction.

The LoCoMo audit's finding -- that its LLM judge accepted up to 63% of
intentionally wrong answers -- came from feeding it answers known to be wrong.
Nothing about that method requires human annotation: if you take a question's
gold answer and substitute a *different* question's gold answer, the result is
wrong by construction. Labels come from how the case was built.

That turns judge validation from 100 hand labels into a script. It measures the
two failure modes that matter:

  false accept -- grader says CORRECT on a known-wrong answer (inflates scores)
  false reject -- grader says INCORRECT on a known-good answer (deflates them)

Honest limit, stated in the report as well as here: constructed negatives are
*easy* negatives. A grader that passes this audit can still misjudge a model's
genuine near-miss. Passing is necessary for trusting a grader, not sufficient.
The sufficient check is human labels on the cases where two independent graders
disagree -- a far smaller set than labelling everything. See
`scripts/audit_graders.py --export-disagreements`.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

_DIGITS = re.compile(r"\d+")

# Phrasings a model actually uses when it declines. Correct for abstention
# questions, wrong for answerable ones.
REFUSALS = [
    "I don't know -- that wasn't mentioned in the conversation.",
    "I cannot find that information in the excerpts provided.",
    "The conversation does not contain any mention of that.",
    # Added after a real 1.5B output phrased it this way and was scored wrong;
    # kept here so the audit regression-tests the fix.
    "I don't have enough information to determine that.",
    # Plural subject, so "do not contain" rather than "does not contain". The
    # patterns matched only the singular, which scored a correct refusal on an
    # abstention question as wrong.
    "I'm sorry, but the excerpts provided do not contain information about that.",
]

# Rewrites that preserve meaning but change surface form. These exist because
# the easy positives (gold verbatim, gold inside a sentence) are circular for a
# containment-based grader -- it passes them by construction, so they measure
# nothing. A false-reject rate is only informative against rewrites a human
# would accept and a string matcher might not.
_NUM_WORDS = {
    "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
    "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten",
    "11": "eleven", "12": "twelve", "15": "fifteen", "20": "twenty",
    "30": "thirty", "40": "forty", "50": "fifty", "100": "hundred",
}
_MONTHS = ["January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December"]
_UNIT_EXPANSIONS = [("$", "", " dollars"), ("%", "", " percent")]
_WEEKDAYS = {"monday", "tuesday", "wednesday", "thursday", "friday",
             "saturday", "sunday"}


@dataclass
class AuditCase:
    question: str
    gold: str
    pred: str
    expected: bool  # known by construction
    kind: str       # which perturbation produced it


def _perturb_number(gold: str) -> str | None:
    """Change a number so the answer becomes factually wrong but plausible."""
    m = _DIGITS.search(gold)
    if not m:
        return None
    original = m.group()
    shifted = str(int(original) + 7 if len(original) <= 3 else int(original) + 111)
    if shifted == original:
        return None
    return gold[:m.start()] + shifted + gold[m.end():]


def _hard_positives(gold: str, question: str = "") -> list[tuple[str, str]]:
    """Meaning-preserving rewrites of a gold answer, as (kind, text).

    Each is something a human grader accepts. Whether a string matcher accepts
    them is exactly what needs measuring.
    """
    out: list[tuple[str, str]] = []

    m = _DIGITS.search(gold)
    if m and m.group() in _NUM_WORDS:
        out.append(("hard_number_word",
                    gold[:m.start()] + _NUM_WORDS[m.group()] + gold[m.end():]))

    for month in _MONTHS:
        if month.lower() in gold.lower():
            pattern = re.compile(re.escape(month), re.I)
            out.append(("hard_month_abbrev", pattern.sub(month[:3], gold)))
            break

    for symbol, repl, suffix in _UNIT_EXPANSIONS:
        if symbol in gold:
            out.append(("hard_unit_expanded",
                        gold.replace(symbol, repl).strip() + suffix))
            break

    # Casing and stray punctuation: a grader that fails these is broken outright.
    out.append(("hard_case_punct", gold.upper() + " !!"))

    # Number agreement. Found in real predictions, not by this audit: gold
    # "Friday" against a prediction of "Fridays", and gold "55-inch" against
    # "55 inches", were both scored wrong. The question fixes the referent, so
    # the plural carries no extra meaning and a human accepts it.
    # Only common count nouns qualify. Pluralising a proper noun ("Hawaiis",
    # "Rosciolis") or a number word ("fours") is not a rewrite any human would
    # accept, so scoring the grader against it would manufacture failures that
    # say nothing about the grader. Weekdays are the one capitalised class that
    # genuinely pluralises, and they are the case that started this.
    tail = gold.rstrip(".!? ").split()
    last = tail[-1] if tail else ""
    eligible = (last.isalpha() and len(last) > 3 and not last.lower().endswith("s")
                and last.lower() not in _NUM_WORDS.values()
                and (last.islower() or last.lower() in _WEEKDAYS))
    if eligible:
        plural = last + ("es" if last.lower().endswith(("ss", "x", "z", "ch", "sh"))
                         else "s")
        out.append(("hard_plural", " ".join(tail[:-1] + [plural])))

    # Set-valued answers listed in a different order. Added after inspecting
    # real predictions: a gold of "atmospheric distillation, fluid catalytic
    # cracking, alkylation, and hydrotreating" is a SET, but token-sequence
    # containment demands one specific ordering, so a model that names all four
    # in any other order is scored wrong. Every rewrite above varies surface
    # form only; this is the first that varies structure, which is where a
    # containment grader is actually weak.
    # ...unless the question asks for the ordering, in which case a reordering
    # is not a rewrite at all -- it is a different, wrong answer. That case is
    # built as a NEGATIVE in `build_audit_cases` instead.
    from memllm.eval.grade import _ORDER_QUESTION

    items = _list_items(gold)
    if len(items) >= 3 and not _ORDER_QUESTION.search(question):
        rotated = items[1:] + items[:1]
        out.append(("hard_list_reorder",
                    ", ".join(rotated[:-1]) + ", and " + rotated[-1]))
    return out


def _list_items(gold: str) -> list[str]:
    """Split a gold answer that enumerates items. Conservative: returns [] when
    the answer is prose, so ordinary answers are never treated as sets."""
    body = re.sub(r"\s*\band\b\s*", ", ", gold.strip().rstrip("."))
    items = [p.strip() for p in body.split(",") if p.strip()]
    # Multi-clause prose also contains commas; require every item to be short
    # and none to contain a verb-like trailing clause.
    if len(items) < 3 or any(len(p.split()) > 4 for p in items):
        return []
    return items


def build_audit_cases(examples, seed: int = 0, per_type: int = 60) -> list[AuditCase]:
    """Construct labelled grader-test cases from a benchmark's gold answers.

    No model outputs and no human labels are involved -- only the benchmark's
    own answer key, recombined so the correct verdict is known.
    """
    from memllm.eval.grade import (
        _ORDER_QUESTION, gold_signals_abstention, grade, is_extractive,
        normalize_tokens,
    )

    def known_wrong(pred: str, gold: str) -> bool:
        """A constructed negative is only valid if it really is wrong.

        Gold keys that list alternatives ("5 days. 6 days is also acceptable.")
        overlap between examples, so a substituted gold can be a genuinely
        correct answer. Checking against the grader itself keeps the constructed
        labels sound instead of manufacturing fake false-accepts.
        """
        return grade(pred, gold) is not True

    rng = random.Random(seed)
    answerable = [
        e for e in examples
        if not e.is_abstention
        and not gold_signals_abstention(str(e.answer))
        and is_extractive(str(e.answer))
    ]
    abstentions = [e for e in examples if e.is_abstention]

    # Pool of alternative golds, grouped by question type so substitutions stay
    # topically plausible instead of being trivially rejectable.
    by_type: dict[str, list] = {}
    for e in answerable:
        by_type.setdefault(e.question_type, []).append(e)

    cases: list[AuditCase] = []

    def add(ex, pred, expected, kind):
        cases.append(AuditCase(ex.question, str(ex.answer), pred, expected, kind))

    for ex in answerable[:per_type * 6]:
        gold = str(ex.answer)

        # --- known-correct answers a grader must accept ---
        # Easy: gold appears verbatim. Circular for containment; kept only to
        # catch a grader that is broken in an obvious way.
        add(ex, gold, True, "identity")
        add(ex, f"Based on our earlier conversation, {gold}.", True, "padded")
        add(ex, f"The answer is {gold}.", True, "sentence")
        # Hard: meaning preserved, surface form changed. This is where a string
        # matcher earns or loses its false-reject rate.
        for kind, text in _hard_positives(gold, ex.question):
            add(ex, text, True, kind)

        # The same rotation, for a question whose answer IS the ordering, is a
        # known-WRONG answer. This bucket exists because re-grading stored
        # predictions caught a real false accept the audit had no case for:
        # every list case here was a positive, so nothing tested that the
        # grader can still say no to a list.
        ordered = _list_items(gold)
        if len(ordered) >= 3 and _ORDER_QUESTION.search(ex.question):
            rotated = ordered[1:] + ordered[:1]
            add(ex, ", ".join(rotated[:-1]) + ", and " + rotated[-1],
                False, "reordered_ordered_list")

        # --- known-wrong answers a grader must reject ---
        pool = by_type.get(ex.question_type, [])
        for _ in range(6):  # a few tries to find a genuinely different gold
            other = str(rng.choice(pool).answer) if pool else ""
            if (other and normalize_tokens(other) != normalize_tokens(gold)
                    and known_wrong(other, gold)):
                add(ex, other, False, "swapped_gold")
                break

        perturbed = _perturb_number(gold)
        if perturbed and known_wrong(perturbed, gold):
            add(ex, perturbed, False, "perturbed_number")

        add(ex, rng.choice(REFUSALS), False, "refusal_to_answerable")
        add(ex, "", False, "empty")

    for ex in abstentions:
        # Inverted: declining is correct, answering confidently is not.
        for r in REFUSALS:
            add(ex, r, True, "correct_abstention")
        if answerable:
            confident = str(rng.choice(answerable).answer)
            add(ex, confident, False, "hallucinated_on_abstention")

    return cases


def audit_grader(cases: list[AuditCase], grade_fn) -> dict:
    """Score a grader against constructed cases.

    `grade_fn(case) -> bool | None`. None counts as unparseable/abstained and is
    excluded from the rates but reported, since a grader that abstains on
    everything would otherwise look flawless.
    """
    by_kind: dict[str, dict[str, int]] = {}
    n_none = 0
    fa_num = fa_den = fr_num = fr_den = 0
    hard_num = hard_den = 0

    for c in cases:
        verdict = grade_fn(c)
        k = by_kind.setdefault(c.kind, {"n": 0, "correct": 0, "none": 0})
        k["n"] += 1
        if verdict is None:
            n_none += 1
            k["none"] += 1
            continue
        if verdict == c.expected:
            k["correct"] += 1
        if c.expected is False:
            fa_den += 1
            fa_num += int(verdict is True)
        else:
            fr_den += 1
            fr_num += int(verdict is False)
            if c.kind.startswith("hard_"):
                hard_den += 1
                hard_num += int(verdict is False)

    graded = len(cases) - n_none
    return {
        "n_cases": len(cases),
        "n_graded": graded,
        "n_abstained": n_none,
        "accuracy": (
            sum(v["correct"] for v in by_kind.values()) / graded if graded else 0.0
        ),
        "false_accept_rate": fa_num / fa_den if fa_den else None,
        "false_reject_rate": fr_num / fr_den if fr_den else None,
        # The headline false-reject number is diluted by positives that contain
        # gold verbatim, which containment accepts trivially. This is the rate
        # on rewrites only, and it is the one worth quoting.
        "false_reject_rate_hard": hard_num / hard_den if hard_den else None,
        "false_accept_note": "known-wrong answers marked CORRECT; the LoCoMo "
                             "audit measured up to 0.63 for its judge",
        "by_kind": {
            k: {**v, "accuracy": v["correct"] / (v["n"] - v["none"])
                if v["n"] - v["none"] else None}
            for k, v in sorted(by_kind.items())
        },
    }


def find_disagreements(records: list[dict]) -> list[dict]:
    """Cases where two graders disagree -- the only ones a human label informs.

    Where independent graders agree, a human label almost always confirms them,
    so labelling those spends effort to learn nothing. Labelling only the
    disagreements bounds both graders' error at a fraction of the cost.
    """
    return [
        r for r in records
        if r.get("deterministic") is not None
        and r.get("judge") is not None
        and r["deterministic"] != r["judge"]
    ]
