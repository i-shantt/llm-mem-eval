"""Measure answer survival for one or more write policies.

    python scripts/run_survival_eval.py --policies verbatim_turn \
        truncated_recency_50 truncated_recency_25 leadk_25

Writes results/survival/survival_<policy>.json per policy, plus a combined
results/survival/summary.json with the paired comparisons.

Survival asks only "is the answer still in the store at all?", which makes it a
ceiling on retrieval and on accuracy, and the one property of a write path that
can be measured without a judge. See memllm/eval/survival.py for the three
definitions, the declared bias, and why the chance floor is reported beside
every rate.

Output goes in results/survival/, a subdirectory, deliberately:
scripts/make_report.py globs results/*.json non-recursively and folds any value
dict containing a "metrics" key into the retrieval table, so a top-level file
here would silently appear as a retriever row.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memllm.cost import CostLedger  # noqa: E402
from memllm.data.loader import load_examples  # noqa: E402
from memllm.eval.ablation import (  # noqa: E402
    ArmResult,
    contingency,
    mcnemar_p,
    paired_bootstrap_ci,
)
from memllm.eval.survival import (  # noqa: E402
    build_placebo_pool,
    eligible_examples,
    report,
    sample_placebos,
    score_store,
)
from memllm.write import build_policy, check_store  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/longmemeval_s")
    ap.add_argument("--policies", nargs="+", default=["verbatim_turn"])
    ap.add_argument("--n-placebo", type=int, default=10,
                    help="placebo golds per question, for the chance floor")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0,
                    help="0 = the whole eligible subset")
    ap.add_argument("--outdir", default="results/survival")
    ap.add_argument("--baseline", default=None,
                    help="policy to compare the others against, paired. "
                         "Defaults to the first --policies entry.")
    args = ap.parse_args()

    examples = eligible_examples(load_examples(args.data))
    if args.limit:
        examples = examples[: args.limit]
    print(f"eligible subset: {len(examples)} questions "
          f"({', '.join(sorted({e.question_type for e in examples}))})")

    # The placebo pool is drawn from the eligible subset only, so a borrowed
    # gold is always a plausible answer to a question of the same kind.
    pool = build_placebo_pool(examples)
    rng = random.Random(args.seed)
    placebos = {ex.question_id: sample_placebos(pool, ex, args.n_placebo, rng)
                for ex in examples}

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    graded: dict[str, dict[str, bool]] = {}
    summaries: dict[str, dict] = {}

    for spec in args.policies:
        policy = build_policy(spec)
        ledger = CostLedger()
        outcomes = []
        for i, ex in enumerate(examples, 1):
            units = policy.build(ex, ledger)
            check_store(units)
            outcomes.append(score_store(ex, units, placebos[ex.question_id]))
            if i % 50 == 0 or i == len(examples):
                print(f"  [{policy.name}] {i}/{len(examples)}", flush=True)

        rep = report(outcomes)
        payload = {
            "store_id": policy.name,
            "policy_config": policy.config(),
            "n_placebo_per_question": args.n_placebo,
            "seed": args.seed,
            # `config` and records keyed `deterministic` let
            # eval.ablation.arm_from_payload read this file unchanged.
            "config": {"tag": policy.name, "retriever": "store",
                       "answer_backend": policy.config().get("policy", "?")},
            "survival": rep,
            "cost_total": ledger.to_dict(),
            "records": [o.to_dict() for o in outcomes],
        }
        out = outdir / f"survival_{policy.name}.json"
        out.write_text(json.dumps(payload, indent=2))

        summaries[policy.name] = rep
        # Built explicitly rather than via arm_from_payload so the restriction
        # to the primary subset is visible here: one-token golds are excluded
        # from every paired test, not just from the headline rate.
        graded[policy.name] = ArmResult(
            name=policy.name,
            model=policy.config().get("policy", "?"),
            retriever="store",
            accuracy=rep["primary"]["record"]["survival"],
            read_tokens_per_query=0.0,
            graded={o.question_id: o.record for o in outcomes if o.in_primary},
            qtype={o.question_id: o.question_type for o in outcomes},
        )

        p = rep["primary"]["record"]
        s = rep["primary"]["soft"]
        print(f"\n=== {policy.name} ===")
        print(f"  n primary {rep['n_primary']}/{rep['n_all']} "
              f"(golds of 2+ tokens)")
        print(f"  record  survival {p['survival']:.3f}  null {p['null']:.3f}  "
              f"corrected {p['chance_corrected']:.3f} "
              f"[{p['chance_corrected_ci95'][0]:.3f}, "
              f"{p['chance_corrected_ci95'][1]:.3f}]")
        print(f"  soft    survival {s['survival']:.3f}  null {s['null']:.3f}  "
              f"corrected {s['chance_corrected']:.3f}")
        st = rep["store_stats"]
        print(f"  store   {st['records_per_store_mean']:.0f} records, "
              f"{st['tokens_per_store_mean']:,.0f} tokens "
              f"({st['tokens_per_record_mean']:.0f}/record)")
        print(f"  wrote {out}")

    baseline = args.baseline or args.policies[0]
    comparisons = {}
    if baseline in graded:
        for name, g in graded.items():
            if name == baseline:
                continue
            # McNemar and the paired bootstrap are reused directly from
            # eval.ablation; compute_lift is deliberately NOT, because it picks
            # the strongest control by accuracy and would compare against the
            # verbatim ceiling instead of the intended budget control.
            comparisons[f"{name}_vs_{baseline}"] = {
                "contingency": contingency(g, graded[baseline]),
                "mcnemar_p": mcnemar_p(g, graded[baseline]),
                "paired_diff_ci95": paired_bootstrap_ci(g, graded[baseline]),
            }

    summary_path = outdir / "summary.json"
    summary_path.write_text(json.dumps({
        "n_eligible": len(examples),
        "baseline": baseline,
        "policies": list(summaries),
        "survival_primary_record": {
            k: v["primary"]["record"] for k, v in summaries.items()
        },
        "store_stats": {k: v["store_stats"] for k, v in summaries.items()},
        "paired_vs_baseline": comparisons,
    }, indent=2))

    if comparisons:
        print(f"\npaired against {baseline} (survival_record, primary subset):")
        for k, v in comparisons.items():
            pt, lo, hi = v["paired_diff_ci95"]
            print(f"  {k:<44} diff {pt:+.3f} [{lo:+.3f}, {hi:+.3f}]  "
                  f"p={v['mcnemar_p']:.2e}")
    print(f"\nwrote {summary_path}")


if __name__ == "__main__":
    main()
