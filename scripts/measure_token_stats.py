"""Measure token statistics of the benchmark so the cost figure uses measured
inputs rather than assumed ones.

    python scripts/measure_token_stats.py --limit 40 > /dev/null
    # writes results/token_stats.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memllm.cost import count_tokens  # noqa: E402
from memllm.data.loader import load_examples, stratified_subset  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/longmemeval_s")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--granularity", default="turn")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/token_stats.json")
    args = ap.parse_args()

    examples = stratified_subset(
        load_examples(args.data), args.limit, seed=args.seed
    )

    per_unit: list[int] = []
    per_example: list[int] = []
    for ex in examples:
        toks = [count_tokens(u.text) for u in ex.units(args.granularity)]
        per_unit.extend(toks)
        per_example.append(sum(toks))

    stats = {
        "data": args.data,
        "granularity": args.granularity,
        "n_examples": len(examples),
        "n_units": len(per_unit),
        "tokens_per_unit": {
            "mean": statistics.mean(per_unit),
            "median": statistics.median(per_unit),
            "p90": sorted(per_unit)[int(0.9 * len(per_unit))],
            "max": max(per_unit),
        },
        "tokens_per_example_full_context": {
            "mean": statistics.mean(per_example),
            "median": statistics.median(per_example),
        },
        "encoding": "tiktoken cl100k_base",
    }

    Path(args.out).parent.mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
