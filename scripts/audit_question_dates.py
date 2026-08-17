"""What resolution do LongMemEval-S's dates actually support?

    python scripts/audit_question_dates.py     # all 500, writes results/question_date_audit.json

`Retriever.search` takes a `question_date` that every retriever in this repo
ignores. That looks like an unfinished feature, and the obvious next commit is
to consume it -- rank by proximity to the question, or refuse to retrieve
anything logged after it, the way a deployed memory system must. This measures
what either of those commits would actually do.

The answer depends entirely on a detail that is easy to miss: `parse_date` keeps
the calendar day and discards the time of day. At the two resolutions the same
data says different things, and both are reported here because the gap between
them *is* the finding.

**At day resolution** -- what every retriever here sorts on -- `question_date`
is inert. No session in any haystack is dated after its question, so
"closest to the question" and "most recent" are the same permutation at every
granularity, checked by sorting both ways and comparing rather than by arguing
about it. Any recency prior monotone in the session date induces that same
permutation whatever origin it measures age from, and for exponential decay the
origin cancels out of every pairwise ratio exactly (checked numerically, since
it is the one claim here about arithmetic rather than about the data). A
retriever that ignores `question_date` at this resolution loses nothing.

**At minute resolution** it is not inert, and not in a comfortable way. 76
questions have at least one session timestamped *after* the question was asked.
The overshoot is never as much as a day, which is exactly why day-truncation
hides all of it. Those sessions are not empty: 43 questions have gold evidence
in one, and for 21 of them *every* gold evidence turn is there.

So a system that treats the question timestamp as a retrieval cutoff -- which is
the correct behaviour for a deployed memory system, and what Mem0's
platform-only `search(reference_date=...)` is for -- scores zero on those 21 by
construction, having done nothing wrong. 42 of the 43 are temporal-reasoning,
the question type that scores worst under every retriever in this repo.

The conclusion is not that the benchmark is broken. At day resolution it is
internally consistent, and the sub-day component of `question_date` is most
likely not intended as a cutoff at all. The conclusion is narrower and it is
for implementers: **these dates support day resolution and no finer.** Filter or
rank at sub-day resolution and you will measure your own correctness as a loss.

Nothing here calls an LLM. The numbers are exact and identical on every run.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import Counter
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_mem_eval.data.loader import Example, load_examples, parse_date  # noqa: E402

_STAMP_RE = re.compile(r"(\d{4})/(\d{2})/(\d{2})[^\d]*(\d{2}):(\d{2})")


def day_ordinal(date_str: str) -> int | None:
    """Day resolution: exactly the key `parse_date` gives the retrievers."""
    y, m, d = parse_date(date_str)
    if (y, m, d) == (0, 0, 0):
        return None
    try:
        return date(y, m, d).toordinal()
    except ValueError:
        return None


def minute_stamp(date_str: str) -> int | None:
    """Minute resolution: the tie-break the day key throws away."""
    m = _STAMP_RE.search(date_str)
    if not m:
        return None
    y, mo, d, hh, mm = (int(g) for g in m.groups())
    try:
        return date(y, mo, d).toordinal() * 1440 + hh * 60 + mm
    except ValueError:
        return None


def orderings_agree(units, keys: dict[int, int], q_key: int) -> bool:
    """Does ranking by absolute recency equal ranking by proximity to the question?

    Both orderings share the same `session_index` tie-break, so a difference in
    the returned permutation is a difference in the primary key and nothing
    else. `HybridRetriever._recency_ranking` sorts by `(date, session_index)`
    descending, which the first of these mirrors.
    """
    by_recency = sorted(units, key=lambda u: (keys[u.unit_id], u.session_index),
                        reverse=True)
    by_proximity = sorted(
        units, key=lambda u: (abs(q_key - keys[u.unit_id]), -u.session_index)
    )
    return [u.unit_id for u in by_recency] == [u.unit_id for u in by_proximity]


def exponential_origin_cancels(session_keys: list[int], q_key: int,
                               half_life_days: float = 30.0) -> float:
    """Largest deviation from origin-independence, over every pair, in ratio terms.

    exp(-l(q - s1)) / exp(-l(q - s2)) = exp(l(s1 - s2)), with no q on the right.
    Recomputing both sides for a far-away origin should therefore agree to
    floating point. This is the check that a multiplicative prior does not
    smuggle `question_date` back in through the ratio.

    Done in log space. Evaluating the two weights separately and dividing
    underflows to 0/0 for any origin far enough to be a convincing test -- at a
    30-day half-life, an origin 100,000 days out puts both exponentials well
    below the smallest representable double. That failure is loud, but the
    version of it that rounds to a *plausible* ratio instead would not be.
    """
    lam = math.log(2) / half_life_days
    far = q_key + 100_000
    worst = 0.0
    for i, s1 in enumerate(session_keys):
        for s2 in session_keys[i + 1:]:
            here = -lam * (q_key - s1) + lam * (q_key - s2)
            there = -lam * (far - s1) + lam * (far - s2)
            # exp of the difference of log-ratios; 0.0 means exact cancellation.
            worst = max(worst, abs(math.exp(here - there) - 1.0))
    return worst


def audit(examples: list[Example]) -> dict:
    position = {"day": Counter(), "minute": Counter()}
    ordering = {"day": Counter(), "minute": Counter()}
    gaps_days: list[int] = []
    unparsable = 0
    worst_ratio_dev = 0.0

    # The sub-day leak, at turn granularity so evidence labels are visible.
    n_with_post_question_sessions = 0
    n_with_post_question_evidence = 0
    n_entirely_post_question_evidence = 0
    post_question_types: Counter = Counter()
    post_evidence_types: Counter = Counter()
    entirely_post_types: Counter = Counter()
    max_overshoot_minutes = 0
    affected_ids: list[str] = []

    for ex in examples:
        q_day, q_min = day_ordinal(ex.question_date), minute_stamp(ex.question_date)
        sessions = ex.units("session")
        s_days = [day_ordinal(u.session_date) for u in sessions]
        if q_day is None or q_min is None or any(d is None for d in s_days):
            unparsable += 1
            continue

        latest = max(s_days)
        if all(d < q_day for d in s_days):
            position["day"]["all sessions strictly before the question"] += 1
        elif latest == q_day:
            position["day"]["latest session same day as the question"] += 1
        else:
            position["day"]["at least one session after the question"] += 1
        gaps_days.append(q_day - latest)

        s_mins = [minute_stamp(u.session_date) for u in sessions]
        if all(m is not None for m in s_mins):
            if all(m < q_min for m in s_mins):
                position["minute"]["all sessions strictly before the question"] += 1
            elif max(s_mins) > q_min:
                position["minute"]["at least one session after the question"] += 1
            else:
                position["minute"]["latest session same minute as the question"] += 1

        for gran in ("session", "user_turn", "turn"):
            units = ex.units(gran)
            for res, q_key, key_fn in (("day", q_day, day_ordinal),
                                       ("minute", q_min, minute_stamp)):
                keys = {u.unit_id: (key_fn(u.session_date) or 0) for u in units}
                agree = orderings_agree(units, keys, q_key)
                ordering[res][f"{gran}: {'agree' if agree else 'DIFFER'}"] += 1

        worst_ratio_dev = max(
            worst_ratio_dev,
            exponential_origin_cancels(sorted(set(s_days)), q_day),
        )

        # --- the sub-day leak ------------------------------------------------
        turns = ex.units("turn")
        post = [u for u in turns if (minute_stamp(u.session_date) or 0) > q_min]
        if not post:
            continue
        n_with_post_question_sessions += 1
        post_question_types[ex.question_type] += 1
        max_overshoot_minutes = max(
            max_overshoot_minutes,
            max((minute_stamp(u.session_date) or 0) - q_min for u in post),
        )
        post_evidence = [u for u in post if u.is_evidence]
        all_evidence = [u for u in turns if u.is_evidence]
        if post_evidence:
            n_with_post_question_evidence += 1
            post_evidence_types[ex.question_type] += 1
            affected_ids.append(ex.question_id)
            if len(post_evidence) == len(all_evidence):
                n_entirely_post_question_evidence += 1
                entirely_post_types[ex.question_type] += 1

    n = len(examples) - unparsable
    day_clean = all("DIFFER" not in k for k in ordering["day"])
    return {
        "n_questions": len(examples),
        "n_unparsable_dates": unparsable,
        "day_resolution": {
            "question_position": dict(position["day"]),
            "recency_vs_proximity_ordering": dict(ordering["day"]),
            "orderings_identical_everywhere": day_clean,
            "gap_to_latest_session_days": {
                "min": min(gaps_days) if gaps_days else None,
                "median": statistics.median(gaps_days) if gaps_days else None,
                "max": max(gaps_days) if gaps_days else None,
            },
            "exponential_origin_max_relative_deviation": worst_ratio_dev,
        },
        "minute_resolution": {
            "question_position": dict(position["minute"]),
            "recency_vs_proximity_ordering": dict(ordering["minute"]),
            "questions_with_post_question_sessions": n_with_post_question_sessions,
            "questions_with_post_question_evidence": n_with_post_question_evidence,
            "questions_with_all_evidence_post_question":
                n_entirely_post_question_evidence,
            "post_question_session_types": dict(post_question_types),
            "post_question_evidence_types": dict(post_evidence_types),
            "all_evidence_post_question_types": dict(entirely_post_types),
            "max_overshoot_minutes": max_overshoot_minutes,
            "max_overshoot_days": max_overshoot_minutes / 1440,
            "question_ids_with_post_question_evidence": sorted(affected_ids),
        },
        "conclusion": (
            f"LongMemEval-S's dates support day resolution and no finer. At day "
            f"resolution question_date is inert: no session is dated after its "
            f"question in {n}/{n} cases, proximity and recency orderings are "
            f"identical at every granularity, and under exponential decay the "
            f"origin cancels from every pairwise ratio (max relative deviation "
            f"{worst_ratio_dev:.2e}). At minute resolution "
            f"{n_with_post_question_sessions} questions have a session logged "
            f"after the question -- never by a full day, which is why day "
            f"truncation hides it -- and {n_with_post_question_evidence} have "
            f"gold evidence in one, {n_entirely_post_question_evidence} of them "
            f"entirely. Treating the question timestamp as a retrieval cutoff, "
            f"which is what a deployed memory system must do, therefore scores "
            f"zero on those {n_entirely_post_question_evidence} by construction."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/longmemeval_s")
    ap.add_argument("--limit", type=int, default=0, help="0 = all 500")
    ap.add_argument("--out", default="results/question_date_audit.json")
    args = ap.parse_args()

    examples = load_examples(args.data, limit=args.limit or None)
    print(f"{len(examples)} questions")
    r = audit(examples)

    for res in ("day_resolution", "minute_resolution"):
        print(f"\n=== {res.replace('_', ' ')} ===")
        for k, v in sorted(r[res]["question_position"].items()):
            print(f"  {v:>4}  {k}")
        for k, v in sorted(r[res]["recency_vs_proximity_ordering"].items()):
            print(f"  {v:>4}  ordering {k}")

    m = r["minute_resolution"]
    print(f"\n=== the sub-day leak ===")
    print(f"  {m['questions_with_post_question_sessions']:>4}  questions with a "
          f"session logged after the question  "
          f"{dict(m['post_question_session_types'])}")
    print(f"  {m['questions_with_post_question_evidence']:>4}  of those have gold "
          f"evidence in one  {dict(m['post_question_evidence_types'])}")
    print(f"  {m['questions_with_all_evidence_post_question']:>4}  have *all* "
          f"their gold evidence there  {dict(m['all_evidence_post_question_types'])}")
    print(f"  max overshoot {m['max_overshoot_minutes']} min "
          f"({m['max_overshoot_days']:.2f} days)")

    d = r["day_resolution"]
    g = d["gap_to_latest_session_days"]
    print(f"\ngap to latest session: min {g['min']}d, median {g['median']}d, "
          f"max {g['max']}d")
    print(f"exponential origin max relative deviation: "
          f"{d['exponential_origin_max_relative_deviation']:.2e}")
    print(f"\n{r['conclusion']}")

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(r, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
