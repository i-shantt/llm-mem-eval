"""Reprint the temporal-reasoning questions behind data/tr_miss_labels.json.

    python scripts/audit_benchmark.py               # must run first
    python scripts/sample_tr_misses.py --n 20 --seed 0

The audit measures that only 0.232 of temporal-reasoning questions (golds of two
or more tokens) have their answer present in the labelled evidence turns. That
number does not say whether the cause is the task or the string matcher, so the
misses were sampled and hand-labelled. This script regenerates the exact sample
those labels refer to, with the evidence quoted, so any label can be checked.

Sampling is `random.Random(seed).shuffle` over the candidates in dataset order,
so the same seed always yields the same questions.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memllm.data.loader import load_examples  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/longmemeval_s")
    ap.add_argument("--audit", default="results/benchmark_audit.json")
    ap.add_argument("--labels", default="data/tr_miss_labels.json")
    ap.add_argument("--question-type", default="temporal-reasoning")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    audit_path = Path(args.audit)
    if not audit_path.exists():
        raise SystemExit(
            f"{audit_path} not found. Run scripts/audit_benchmark.py first."
        )
    audit = {r["question_id"]: r
             for r in json.loads(audit_path.read_text())["records"]}

    candidates = [
        ex for ex in load_examples(args.data)
        if (a := audit.get(ex.question_id))
        and a["question_type"] == args.question_type
        and not a["in_evidence_any_turn"]
        and a["gold_len_bucket"] != "1"
    ]
    random.Random(args.seed).shuffle(candidates)
    sample = candidates[: args.n]

    labels = {}
    lp = Path(args.labels)
    if lp.exists():
        labels = {r["question_id"]: r
                  for r in json.loads(lp.read_text())["labels"]}

    print(f"{len(candidates)} candidates; showing {len(sample)} "
          f"(seed={args.seed})\n")
    for i, ex in enumerate(sample, 1):
        lab = labels.get(ex.question_id, {})
        ev = [t.content for t in ex.turns if t.has_answer]
        print("=" * 92)
        print(f"{i:2d}. [{ex.question_id}]  {lab.get('category', 'UNLABELLED')}")
        print(f"    Q:    {ex.question}")
        print(f"    GOLD: {ex.answer!r}")
        for j, e in enumerate(ev[:2], 1):
            print(f"    EV{j}:  {e[:300]}")
        if lab.get("note"):
            print(f"    NOTE: {lab['note']}")

    if labels:
        covered = sum(1 for ex in sample if ex.question_id in labels)
        counts = Counter(labels[ex.question_id]["category"]
                         for ex in sample if ex.question_id in labels)
        print(f"\n{covered}/{len(sample)} labelled: "
              + ", ".join(f"{c} {n}" for c, n in counts.most_common()))
        if covered != len(sample):
            print("WARNING: the sample and the label file disagree. Either the "
                  "seed/n changed or the audit was regenerated from different "
                  "data; the labels no longer describe this sample.")


if __name__ == "__main__":
    main()
