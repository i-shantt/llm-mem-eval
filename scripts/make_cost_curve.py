"""The headline figure: total cost vs number of queries against one memory.

Write cost is paid once per conversation. Read cost is paid per query. So the
honest comparison is a line, not a point:

    total(n) = write_cost + n * read_cost_per_query

A system with an expensive LLM-driven write path starts high and only amortises
after many queries hit the same memory. Read-path-only accounting -- what the
literature reports -- is the n -> infinity limit, which flatters exactly the
systems that spend the most up front.

    python scripts/make_cost_curve.py --results results/sweep_turn_n100.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Price we attribute to the answering model's prompt tokens. Retrieval in this
# repo is local and free; what costs money is the context we hand to the LLM.
ANSWER_MODEL = "gpt-4o-mini"
PROMPT_PRICE_PER_1M = 0.15


def tokens_for_context(n_units: int, avg_tokens_per_unit: float) -> float:
    return n_units * avg_tokens_per_unit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/sweep_turn_n100.json")
    ap.add_argument("--k", type=int, default=10,
                   help="how many retrieved units get sent to the LLM")
    ap.add_argument("--token-stats", default="results/token_stats.json",
                   help="produced by scripts/measure_token_stats.py")
    ap.add_argument("--avg-unit-tokens", type=float, default=None,
                   help="override the measured mean tokens per retrieved turn")
    ap.add_argument("--full-context-tokens", type=float, default=None,
                   help="override the measured full-context token count")
    ap.add_argument("--max-queries", type=int, default=1000)
    ap.add_argument("--out", default="results/cost_curve.png")
    args = ap.parse_args()

    runs = json.loads(Path(args.results).read_text())

    # Prefer measured token statistics over any hardcoded guess.
    stats_path = Path(args.token_stats)
    if stats_path.exists():
        ts = json.loads(stats_path.read_text())
        measured_unit = ts["tokens_per_unit"]["mean"]
        measured_full = ts["tokens_per_example_full_context"]["mean"]
        print(f"using measured token stats from {stats_path} "
              f"({ts['n_units']} units over {ts['n_examples']} examples)")
    else:
        raise SystemExit(
            f"{stats_path} not found -- run scripts/measure_token_stats.py "
            "first so the figure uses measured inputs, not assumptions."
        )
    if args.avg_unit_tokens is not None:
        measured_unit = args.avg_unit_tokens
    if args.full_context_tokens is not None:
        measured_full = args.full_context_tokens
    args.avg_unit_tokens = measured_unit
    args.full_context_tokens = measured_full

    # Our system: zero LLM calls on the write path. The only per-query LLM cost
    # is the retrieved context we prepend.
    retrieved_tokens = args.k * args.avg_unit_tokens
    ours_read_per_query = retrieved_tokens * PROMPT_PRICE_PER_1M / 1e6
    ours_write = 0.0  # no LLM calls, no API dollars -- local compute only

    full_read_per_query = args.full_context_tokens * PROMPT_PRICE_PER_1M / 1e6

    published = json.loads(Path("data/published_costs.json").read_text())
    # Read path from Mem0's own Table 2; construction tokens from RecMem's
    # Table 8, because Mem0's paper reports no write-path cost at all.
    read_block = published["locomo_read_path"]
    build_block = published["locomo_construction_tokens"]
    mem0_read_tokens = next(
        s["tokens_per_query"] for s in read_block["systems"] if s["name"] == "Mem0"
    )
    mem0_build_tokens = next(
        s["construction_tokens"] for s in build_block["systems"]
        if s["name"] == "Mem0"
    )
    mem0_write = mem0_build_tokens * PROMPT_PRICE_PER_1M / 1e6
    mem0_read_per_query = mem0_read_tokens * PROMPT_PRICE_PER_1M / 1e6

    ns = list(range(1, args.max_queries + 1))
    series = {
        f"memllm hybrid (top-{args.k}) — measured": [
            ours_write + n * ours_read_per_query for n in ns
        ],
        "full context — measured": [n * full_read_per_query for n in ns],
        "Mem0 — reported, LoCoMo (not comparable)": [
            mem0_write + n * mem0_read_per_query for n in ns
        ],
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    styles = {
        f"memllm hybrid (top-{args.k}) — measured": dict(lw=2.4, color="#1b7f5f"),
        "full context — measured": dict(lw=2.0, color="#7a4fb8"),
        "Mem0 — reported, LoCoMo (not comparable)": dict(
            lw=2.0, color="#b8853f", ls="--"
        ),
    }
    for label, ys in series.items():
        ax.plot(ns, ys, label=label, **styles[label])

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("queries against one conversation's memory")
    ax.set_ylabel(f"cumulative cost (USD, {ANSWER_MODEL} prompt tokens)")
    ax.set_title("Memory cost is a line, not a point\n"
                 "write path paid once + read path paid per query",
                 fontsize=11)
    ax.grid(alpha=0.25, which="both", lw=0.5)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)

    note = (
        "Mem0 line uses REPORTED LoCoMo numbers (unverified) and a different\n"
        "benchmark; shown for shape, not for head-to-head comparison."
    )
    ax.text(0.98, 0.03, note, transform=ax.transAxes, fontsize=7,
            ha="right", va="bottom", color="#555")

    fig.tight_layout()
    Path(args.out).parent.mkdir(exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print(f"wrote {args.out}")

    # Crossover: how many queries before an LLM-heavy write path pays off?
    print("\nBreak-even analysis (USD):")
    print(f"  memllm:       write ${ours_write:.4f} + "
          f"${ours_read_per_query:.5f}/query  ({retrieved_tokens:.0f} tok/query)")
    print(f"  full context: write $0 + "
          f"${full_read_per_query:.5f}/query  "
          f"({args.full_context_tokens:.0f} tok/query)")
    print(f"  Mem0 (rep.):  write ${mem0_write:.4f} + "
          f"${mem0_read_per_query:.5f}/query")

    ratio = full_read_per_query / ours_read_per_query
    print(f"\n  memllm read path is {ratio:.1f}x cheaper per query "
          f"than full context.")
    if mem0_read_per_query < ours_read_per_query:
        n_even = mem0_write / (ours_read_per_query - mem0_read_per_query)
        print(f"  Mem0's cheaper read path only repays its write path after "
              f"{n_even:.0f} queries on the same conversation.")
    else:
        print("  Mem0's reported read path is not cheaper than ours, so its "
              "write path never repays.")


if __name__ == "__main__":
    main()
