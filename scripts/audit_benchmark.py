"""Audit how much of LongMemEval is answerable by finding a span.

    python scripts/audit_benchmark.py          # all 500, writes results/benchmark_audit.json

Every retrieval metric on this benchmark -- ours included -- is reported as one
aggregate over 500 questions. That aggregate is only meaningful for questions
whose answer is actually *present* in the turns the benchmark labels as evidence.
This script measures what fraction those are, per question type.

It asks three separate questions, because they support different claims:

  in_evidence      the gold answer's token sequence appears in a `has_answer`
                   turn. A retriever that returns exactly the gold evidence
                   hands the reader a span it can copy.
  in_haystack      it appears in *some* turn, labelled or not.
  outside_evidence in_haystack and not in_evidence. This separates two very
                   different explanations for a low in_evidence rate:
                   the benchmark requires synthesis (answer is nowhere verbatim)
                   versus the annotation is coarse (answer is verbatim in a turn
                   nobody labelled). Only the first is a property of the task.

Answers of one normalised token are reported separately throughout. They match
by chance -- see scripts/run_survival_eval.py, which measures that chance rate
directly -- so a high in_evidence rate in that bucket means much less than the
same rate in the 4+ bucket.

Nothing here calls an LLM. The numbers are exact and identical on every run.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_mem_eval.data.loader import Example, load_examples  # noqa: E402
from llm_mem_eval.eval.grade import (  # noqa: E402
    contains_answer,
    gold_signals_abstention,
    is_extractive,
    normalize_tokens,
)

# Buckets, not a continuum: one-token golds behave categorically differently
# from the rest under any containment test.
BUCKETS = ("1", "2-3", "4+")


def gold_len_bucket(gold: str) -> str:
    n = len(normalize_tokens(gold))
    return "1" if n <= 1 else ("2-3" if n <= 3 else "4+")


def is_numeral_gold(gold: str) -> bool:
    """A bare count like '3'. These are aggregates, not spans to be retrieved."""
    return gold.strip().replace(".", "", 1).isdigit()


def audit_example(ex: Example) -> dict:
    """Per-question containment facts. No aggregation, so callers can re-slice."""
    gold, q = str(ex.answer), ex.question
    turn_texts = [t.content for t in ex.turns]
    ev_texts = [t.content for t in ex.turns if t.has_answer]

    # Per-turn `any` is the same test the survival metric applies to a store
    # record; the union is the more generous "somewhere in the evidence" test.
    # They can differ when an answer is split across two evidence turns, so
    # both are reported rather than picking one.
    in_ev_any = any(contains_answer(t, gold, q) for t in ev_texts)
    in_ev_union = contains_answer("\n".join(ev_texts), gold, q)
    in_hay_any = any(contains_answer(t, gold, q) for t in turn_texts)

    return {
        "question_id": ex.question_id,
        "question_type": ex.question_type,
        "gold": gold,
        "gold_len_bucket": gold_len_bucket(gold),
        "gold_is_numeral": is_numeral_gold(gold),
        "n_evidence_turns": ex.n_evidence_turns,
        "in_evidence_any_turn": in_ev_any,
        "in_evidence_union": in_ev_union,
        "in_haystack_any_turn": in_hay_any,
        # The load-bearing one: verbatim somewhere, but not where the benchmark
        # says the evidence is.
        "outside_evidence_only": in_hay_any and not in_ev_any,
    }


def _rate(rows: list[dict], key: str) -> float:
    return sum(r[key] for r in rows) / len(rows) if rows else 0.0


def _group(rows: list[dict], by: str) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r[by]].append(r)
    return groups


def summarise(rows: list[dict], by: str) -> dict:
    groups = _group(rows, by)
    return {
        g: {
            "n": len(rs),
            "in_evidence_any_turn": _rate(rs, "in_evidence_any_turn"),
            "in_evidence_union": _rate(rs, "in_evidence_union"),
            "in_haystack_any_turn": _rate(rs, "in_haystack_any_turn"),
            "outside_evidence_only": _rate(rs, "outside_evidence_only"),
            "frac_numeral_gold": _rate(rs, "gold_is_numeral"),
        }
        for g, rs in sorted(groups.items(), key=lambda kv: -len(kv[1]))
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/longmemeval_s")
    ap.add_argument("--limit", type=int, default=0,
                    help="0 = all questions; the audit is a property of the "
                         "benchmark, so it should normally run on all of them")
    ap.add_argument("--out", default="results/benchmark_audit.json")
    args = ap.parse_args()

    examples = load_examples(args.data)
    if args.limit:
        examples = examples[: args.limit]

    # Excluded for stated reasons, not silently. Abstention questions have no
    # span to find by construction; abstractive golds have no checkable surface
    # form, which is the same rule `grade()` uses to return None.
    n_abstention = sum(1 for ex in examples if ex.is_abstention)
    n_abstractive = sum(
        1 for ex in examples
        if not ex.is_abstention
        and (not is_extractive(ex.answer) or gold_signals_abstention(ex.answer))
    )
    scored = [
        ex for ex in examples
        if not ex.is_abstention
        and is_extractive(ex.answer)
        and not gold_signals_abstention(ex.answer)
    ]

    rows = [audit_example(ex) for ex in scored]

    payload = {
        "data": args.data,
        "n_questions_in_split": len(examples),
        "n_excluded_abstention": n_abstention,
        "n_excluded_abstractive_gold": n_abstractive,
        "n_scored": len(rows),
        "overall": {
            "in_evidence_any_turn": _rate(rows, "in_evidence_any_turn"),
            "in_evidence_union": _rate(rows, "in_evidence_union"),
            "in_haystack_any_turn": _rate(rows, "in_haystack_any_turn"),
            "outside_evidence_only": _rate(rows, "outside_evidence_only"),
        },
        "by_question_type": summarise(rows, "question_type"),
        "by_gold_len_bucket": summarise(rows, "gold_len_bucket"),
        "by_type_and_bucket": {
            f"{t}|{b}": {
                "n": n,
                "in_evidence_any_turn": _rate(
                    [r for r in rows
                     if r["question_type"] == t and r["gold_len_bucket"] == b],
                    "in_evidence_any_turn"),
            }
            for (t, b), n in sorted(Counter(
                (r["question_type"], r["gold_len_bucket"]) for r in rows
            ).items())
        },
        # The headline slice. One-token golds are dropped because they match by
        # chance often enough to dominate any rate computed over them.
        "by_question_type_gold_ge2_tokens": summarise(
            [r for r in rows if r["gold_len_bucket"] != "1"], "question_type"
        ),
        # Three mutually exclusive fates for a gold answer, which is the whole
        # point of the audit: only the third is a property of the task.
        "answer_location_gold_ge2_tokens": {
            t: {
                "n": len(rs),
                "in_labelled_evidence": _rate(rs, "in_evidence_any_turn"),
                "verbatim_but_unlabelled": _rate(rs, "outside_evidence_only"),
                "nowhere_verbatim": 1.0 - _rate(rs, "in_haystack_any_turn"),
            }
            for t, rs in sorted(
                _group([r for r in rows if r["gold_len_bucket"] != "1"],
                       "question_type").items(),
                key=lambda kv: -len(kv[1]),
            )
        },
        # One on this benchmark, which is worth recording either way. The single
        # case is `gpt4_f420262c`, gold "JetBlue, Delta, United, American
        # Airlines": the four airlines appear in the right order across the
        # evidence turns but never all within one of them. So per-record and
        # whole-store containment are equivalent for 415 of 416 questions, and
        # the one exception is temporal-reasoning -- outside the three span
        # types the survival metric scores, which is why survival's union and
        # record definitions still agree on every question it measures.
        "n_union_vs_any_turn_disagreements": sum(
            1 for r in rows
            if r["in_evidence_union"] != r["in_evidence_any_turn"]
        ),
        "reading": [
            "in_evidence_* is the fraction of questions whose gold answer "
            "appears verbatim in the turns the benchmark labels has_answer. It "
            "is NOT an accuracy: a question can be answered correctly by "
            "reasoning over evidence that does not contain the answer string.",
            "outside_evidence_only distinguishes 'the task needs synthesis' "
            "from 'the annotation is coarse'. A high value means the answer "
            "text exists in an unlabelled turn.",
            "One-token golds match by chance; treat that bucket separately.",
        ],
        "records": rows,
    }

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))

    print(f"{len(examples)} questions; scored {len(rows)} "
          f"(excluded {n_abstention} abstention, "
          f"{n_abstractive} abstractive gold)\n")
    hdr = f"{'question_type':<26}{'n':>5}{'in_evid':>9}{'in_hay':>8}{'outside':>9}{'numeral':>9}"
    print(hdr)
    print("-" * len(hdr))
    for t, v in payload["by_question_type"].items():
        print(f"{t:<26}{v['n']:>5}{v['in_evidence_any_turn']:>9.3f}"
              f"{v['in_haystack_any_turn']:>8.3f}"
              f"{v['outside_evidence_only']:>9.3f}{v['frac_numeral_gold']:>9.3f}")
    o = payload["overall"]
    print(f"{'ALL':<26}{len(rows):>5}{o['in_evidence_any_turn']:>9.3f}"
          f"{o['in_haystack_any_turn']:>8.3f}{o['outside_evidence_only']:>9.3f}")

    print("\nWhere the gold answer actually is (golds >= 2 tokens, so chance "
          "matches do not dominate):")
    hdr2 = (f"{'question_type':<26}{'n':>5}{'labelled':>10}{'unlabelled':>12}"
            f"{'nowhere':>10}")
    print(hdr2)
    print("-" * len(hdr2))
    for t, v in payload["answer_location_gold_ge2_tokens"].items():
        print(f"{t:<26}{v['n']:>5}{v['in_labelled_evidence']:>10.3f}"
              f"{v['verbatim_but_unlabelled']:>12.3f}{v['nowhere_verbatim']:>10.3f}")
    print(f"\nanswers split across evidence turns: "
          f"{payload['n_union_vs_any_turn_disagreements']}/{len(rows)}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
