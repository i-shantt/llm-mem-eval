"""Deterministic answer grading -- no LLM judge, no human labels.

Most of LongMemEval does not need a judge. The median gold answer is 11
characters ("Target", "$800", "February 14th"), which is short-answer QA, and
short-answer QA had a reproducible metric for a decade before LLM judges
arrived. Using a judge here trades reproducibility for nothing.

Two things make this trustworthy rather than merely convenient:

1. It abstains. Some gold answers are long rubric-style paragraphs
   ("The user would prefer responses that...") where no string metric is
   meaningful. `grade()` returns None on those instead of guessing, and the
   aggregate excludes them rather than scoring them wrong.

2. It is audited. `grader_audit.py` builds cases whose correct verdict is known
   by construction, so this grader's false-accept and false-reject rates are
   measured, not assumed. That is the same audit that found LoCoMo's judge
   accepting up to 63% of intentionally wrong answers -- run against our own
   grader first.
"""

from __future__ import annotations

import re
import string

# Gold answers longer than this are abstractive: rubrics and preference
# descriptions with no single correct surface form. Threshold rather than a
# hardcoded question-type list, so this generalises to other benchmarks.
MAX_EXTRACTIVE_GOLD_TOKENS = 12

_PUNCT = str.maketrans({c: " " for c in string.punctuation})
_ARTICLES = {"a", "an", "the"}
_ORDINAL = re.compile(r"^(\d+)(st|nd|rd|th)$")

# Both of these exist because the grader audit measured them as defects: a model
# writing "twenty" for gold "20" was scored wrong on 152 constructed cases, and
# "Feb" for "February" on 3. Number words and month names are normalised to one
# canonical form so surface choice stops affecting the score.
_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90", "hundred": "100", "thousand": "1000",
}
_MONTHS_FULL = {"january", "february", "march", "april", "may", "june", "july",
                "august", "september", "october", "november", "december"}

# Sibilant stems take "-es" rather than "-s": inch/inches, box/boxes, dish/dishes.
# A bare "s" is deliberately NOT in this list. It would read "cases" as "cas"+"es"
# and fold it to "cas" while gold "case" stays "case" -- turning a match into a
# miss. The audit caught exactly that on gold "Receiving the new phone case".
# Doubled "ss" is safe and is what makes classes/class and glasses/glass work.
_SIBILANT = ("ss", "x", "z", "ch", "sh")


def _singularise(tok: str) -> str:
    """Strip a regular plural "-s", conservatively.

    Measured defect: gold "Friday" scored a prediction of "Fridays" wrong, and
    gold "55-inch" scored "55 inches" wrong. Both are the same answer.

    Everything here is a guard against a *collision*, because two different
    words folding together is a false accept and this grader's whole claim is a
    0.000 false-accept rate. Short tokens are exempt ("was" -> "wa", "bus" ->
    "bu"), as are the "-ss"/"-us"/"-is" endings that are not plurals at all
    (class, campus, analysis).

    There is deliberately no "-ies" -> "-y" rule. It is correct for city/cities
    but wrong for movie/movies, which it would fold to "movy" while the singular
    stays "movie" -- turning a match into a miss. Plain "-s" stripping handles
    movies correctly and merely fails to help cities, and a miss is the safe
    direction to err.
    """
    if len(tok) <= 3 or not tok.endswith("s") or tok.endswith(("ss", "us", "is")):
        return tok
    if tok.endswith("es") and tok[:-2].endswith(_SIBILANT):
        return tok[:-2]
    return tok[:-1]

# A grader that accepts "I don't know" as correct inflates every score, so
# refusals are detected explicitly rather than left to fall through.
_REFUSAL_PATTERNS = [
    r"\bdo(?:n't| not)\s+know\b",
    r"\b(?:can(?:'t|not)|unable to)\s+(?:find|determine|tell|answer|recall)\b",
    r"\bno\s+(?:information|mention|record|reference|details?)\b",
    r"\bnot\s+(?:mentioned|specified|provided|discussed|stated|available|included)\b",
    r"\bis(?:n't| not)\s+(?:mentioned|specified|provided|discussed|stated)\b",
    # do/does, both contracted and not. Matching only "does not contain" left
    # "the excerpts do not contain information" undetected, so a correct
    # refusal on an abstention question was scored wrong -- and because the
    # phrasing varies between arms, that showed up as a fake +1 for whichever
    # arm happened to word it differently. Found by reading the questions a
    # conditioner appeared to fix.
    r"\bdo(?:es)?(?:n't| not)\s+(?:contain|mention|specify|say|include)\b",
    r"\bwas(?:n't| not)\s+(?:mentioned|discussed|specified)\b",
    r"\bnever\s+(?:mentioned|discussed|stated)\b",
    r"\bno\s+such\s+\w+\b",
    r"\binsufficient\s+(?:information|context|detail)\b",
    # "I don't have enough information" is one of the most common ways a model
    # declines and was missed by the patterns above, so a correct refusal was
    # being scored wrong. Found by inspecting real model output, not by the
    # constructed audit -- which is why both checks exist.
    r"\bdo(?:n't| not)\s+have\s+(?:enough|sufficient|any|access)\b",
    r"\b(?:not|isn't|is not)\s+enough\s+(?:information|context|detail)\b",
    r"\bno\s+way\s+to\s+(?:know|tell|determine)\b",
]
_REFUSAL = re.compile("|".join(_REFUSAL_PATTERNS), re.I)

# Gold answers that assert the question is unanswerable.
_GOLD_ABSTAIN = re.compile(
    r"\b(?:information provided is not enough|not enough information|"
    r"did(?:n't| not) mention|was not mentioned|cannot be answered|"
    r"no information (?:about|on|regarding))\b", re.I)

# "... is also acceptable", "... is acceptable too" -- the wrapper around an
# alternative answer, removed so the alternative itself can be matched.
_ACCEPTABLE_CLAUSE = re.compile(
    r"\s*(?:is|are|would be)?\s*(?:also\s+)?acceptable(?:\s+too)?\s*\.?", re.I)


def normalize_tokens(text: str) -> list[str]:
    """SQuAD-style normalisation, plus fixes for this benchmark's answers.

    Punctuation removal collapses "$800" -> "800" and "10%" -> "10" so a model
    that writes "800 dollars" still matches. Ordinal stripping makes
    "February 14th" match "February 14", which is otherwise a spurious miss.
    Number is stripped last, so "Fridays" matches gold "Friday".
    """
    text = str(text).lower().translate(_PUNCT)
    out = []
    for tok in text.split():
        tok = _ORDINAL.sub(r"\1", tok)
        tok = _NUMBER_WORDS.get(tok, tok)
        # Months collapse to their 3-letter prefix, so "february" == "feb".
        if tok in _MONTHS_FULL or tok == "sept":
            tok = tok[:3]
        else:
            tok = _singularise(tok)
        if tok and tok not in _ARTICLES:
            out.append(tok)
    return out


def is_refusal(pred: str) -> bool:
    return bool(_REFUSAL.search(str(pred)))


def gold_signals_abstention(gold: str) -> bool:
    """Some gold answers are themselves statements that the question is
    unanswerable ("You did not mention this information...").

    LongMemEval flags most of these with an `_abs` question id, but not all --
    one of 31 slips through, and grading it by string match marks a correct
    refusal as wrong. Reading the answer key is more reliable than the id.
    """
    return bool(_GOLD_ABSTAIN.search(str(gold)))


def gold_alternatives(gold: str) -> list[str]:
    """Split a gold answer into all separately-acceptable answers.

    LongMemEval's temporal questions spell alternatives out in prose --
    "1 day. 2 days (including the last day) is also acceptable." -- across 38
    examples, 29% of the temporal-reasoning slice. Matching the whole string
    rejects a model that answered "2 days", which the answer key permits. That
    is not a scoring detail; it makes every system look bad at temporal
    reasoning for a reason that has nothing to do with the system.
    """
    text = _ACCEPTABLE_CLAUSE.sub("", str(gold))
    parts: list[str] = []
    for sentence in re.split(r"[.;]\s+", text):
        for clause in re.split(r"\s+\bor\b\s+", sentence):
            clause = clause.strip().strip(".;,")
            if not clause:
                continue
            parts.append(clause)
            # Parentheticals are qualifications, so the answer without them is
            # equally acceptable: "2 days (including the last day)" -> "2 days".
            stripped = re.sub(r"\([^)]*\)", " ", clause).strip()
            if stripped and stripped != clause:
                parts.append(stripped)
    return parts or [str(gold)]


def _contains_span(pred: str, gold: str) -> bool:
    g, p = normalize_tokens(gold), normalize_tokens(pred)
    if not g:
        return False
    return any(p[i:i + len(g)] == g for i in range(len(p) - len(g) + 1))


# Questions whose answer IS the ordering. Set comparison is invalid for these:
# it accepts the right items in the wrong sequence, which is the wrong answer.
# Kept narrow on purpose -- "first", "before" and "after" appear in all sorts of
# questions, and a false trigger here only falls back to strict span matching,
# whereas a missed trigger is a false accept.
_ORDER_QUESTION = re.compile(
    r"\b(order|sequence|chronological|earliest|latest)\b", re.I)


def _set_items(gold: str) -> list[str]:
    """Items of an enumerated gold answer, or [] if it is prose.

    Deliberately narrow: 3+ comma/and-separated fragments, each at most four
    words. Prose sentences contain commas too, and treating one as a set would
    accept any answer that happened to mention its clauses.
    """
    body = re.sub(r"\s*\band\b\s*", ", ", str(gold).strip().rstrip("."))
    items = [p.strip() for p in body.split(",") if p.strip()]
    if len(items) < 3 or any(len(p.split()) > 4 for p in items):
        return []
    return items


def contains_answer(pred: str, gold: str, question: str | None = None) -> bool:
    """True if pred contains any answer the gold key accepts.

    Token-sequence containment rather than substring containment: a raw
    substring test scores gold "20" as found inside pred "120 pages", which is
    wrong and would silently inflate every numeric answer.

    Enumerated answers are compared as sets. The grader audit measured this
    one at 0/2 before the special case existed: a gold listing four refining
    processes demanded one specific ordering, so naming all four in any other
    order scored wrong. ALL items must appear, so this cannot accept a partial
    answer -- the audit's false-accept rate stays at zero.

    `question` gates that set comparison off when the question asks for an
    ordering, and omitting it keeps the old, more permissive behaviour. Passing
    it matters: on "What is the order of airlines I flew with from earliest to
    latest", gold "JetBlue, Delta, United, American Airlines", a 14B model
    answered "JetBlue, Delta, American Airlines, and then United" -- the right
    four airlines in the wrong order, which is simply the wrong answer -- and
    set comparison accepted it. Found by re-grading stored predictions, not by
    the constructed audit, which had no reordering case for an ordered gold.
    """
    if any(_contains_span(pred, alt) for alt in gold_alternatives(gold)):
        return True
    if question is not None and _ORDER_QUESTION.search(question):
        return False
    items = _set_items(gold)
    return bool(items) and all(_contains_span(pred, it) for it in items)


def token_f1(pred: str, gold: str) -> float:
    """Soft overlap, reported alongside the strict metric as a sanity check."""
    g, p = normalize_tokens(gold), normalize_tokens(pred)
    if not g or not p:
        return float(g == p)
    common = 0
    remaining = list(g)
    for tok in p:
        if tok in remaining:
            remaining.remove(tok)
            common += 1
    if common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(g)
    return 2 * precision * recall / (precision + recall)


def is_extractive(gold: str) -> bool:
    """Whether a gold answer has a checkable surface form at all."""
    return len(normalize_tokens(gold)) <= MAX_EXTRACTIVE_GOLD_TOKENS


def grade(pred: str, gold: str, is_abstention: bool = False,
          question: str | None = None) -> bool | None:
    """True=correct, False=incorrect, None=not deterministically gradable.

    None is a real answer, not a failure. Scoring an abstractive question with
    string matching produces a number that looks like accuracy and isn't.

    `question` is optional so existing callers keep working, but callers that
    have it should pass it -- see `contains_answer` for the false accept it
    prevents on ordering questions.
    """
    if is_abstention or gold_signals_abstention(gold):
        # The task is to decline; the gold string is not the target.
        return is_refusal(pred)
    if not is_extractive(gold):
        return None
    if is_refusal(pred):
        return False
    return contains_answer(pred, gold, question)
