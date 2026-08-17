"""Attribute a memory system's accuracy: how much of it is actually memory?

    python scripts/run_ablation.py --results results

Reads e2e result files, groups arms by model, and scores every real retriever
against the strongest control that shares its question set. Reports the lift,
a paired bootstrap CI, an exact McNemar p-value, and the fraction of the
headline accuracy that survives the control.

Run the controls first -- they are cheap, and without them the rest of the
table cannot be interpreted:

    for R in none random recency; do
      python scripts/run_e2e_eval.py --limit 100 --retriever $R \
        --answer-backend ollama:qwen2.5:7b-instruct --k 10 \
        --tag e2e_7b_${R}_turn_k10_n100
    done
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_mem_eval.eval.ablation import (  # noqa: E402
    CONTROL_ARMS,
    arm_from_payload,
    compute_lift,
)


def load_arms(results_dir: Path) -> list:
    arms = []
    for f in sorted(results_dir.glob("e2e_*.json")):
        try:
            payload = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        if "records" not in payload or "accuracy" not in payload:
            continue
        # A clamped arm reports one identical prompt length for every question;
        # its accuracy is an artefact and must not become somebody's control.
        pt = {r.get("prompt_tokens") for r in payload["records"]}
        if len(payload["records"]) > 5 and len(pt) == 1 and pt != {0}:
            print(f"  skipping {f.name}: prompt length identical on every "
                  f"question ({pt.pop()} tok) -- context was clamped",
                  file=sys.stderr)
            continue
        arms.append(arm_from_payload(payload, name=f.stem))
    return arms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/ablation.json")
    args = ap.parse_args()

    arms = load_arms(Path(args.results))
    if not arms:
        sys.exit(f"no usable e2e results in {args.results}/")

    by_model: dict[str, list] = {}
    for a in arms:
        by_model.setdefault(a.model, []).append(a)

    reports = []
    for model, group in sorted(by_model.items()):
        controls = [a for a in group if a.is_control]
        systems = [a for a in group if not a.is_control]
        short = model.split(":")[-1] if ":" in model else model

        print(f"\n{'=' * 78}\n{short}\n{'=' * 78}")
        if not controls:
            print("  NO CONTROL ARMS. Every number below is an upper bound on")
            print("  this system's contribution, not a measurement of it.")
            print(f"  Run: --retriever none  (and random, recency) for {short}")
            for s in sorted(systems, key=lambda x: -x.accuracy):
                print(f"    {s.retriever:<10} accuracy {s.accuracy:.3f}  "
                      f"(unattributable)")
            continue

        print(f"  controls: " + ", ".join(
            f"{c.retriever}={c.accuracy:.3f}" for c in
            sorted(controls, key=lambda x: -x.accuracy)))
        print(f"\n  {'system':<10} {'acc':>6} {'ctl':>6} {'lift':>7} "
              f"{'95% CI':>16} {'p':>8} {'attrib':>8}")
        computed = []
        for s in sorted(systems, key=lambda x: -x.accuracy):
            try:
                r = compute_lift(s, controls, seed=args.seed)
            except ValueError as e:
                print(f"  {s.retriever:<10} skipped: {e}")
                continue
            computed.append(r)
            star = "*" if r.significant else " "
            print(f"  {s.retriever:<10} {r.system_accuracy:>6.3f} "
                  f"{r.control_accuracy:>6.3f} {r.lift:>+7.3f}{star} "
                  f"[{r.ci_lo:>+.3f},{r.ci_hi:>+.3f}] {r.p_value:>8.4f} "
                  f"{r.attributable_fraction:>7.1%}")
            reports.append({"model": model, **r.__dict__})

        # Where the lift lives. A system can post a healthy overall lift while
        # contributing nothing on the question types it was bought for.
        #
        # Reuses the reports computed above rather than recomputing them. The
        # old version re-ran compute_lift over `systems` in a generator with no
        # except clause, so a system that shares no graded question with any
        # control -- printed as "skipped" two lines up -- raised ValueError here
        # and killed the script after the table had already looked fine. It also
        # paid for a second 10,000-resample bootstrap per system.
        best = max(computed, key=lambda r: r.lift, default=None)
        if best is not None:
            print(f"\n  best system ({best.system.split('_')[1] if '_' in best.system else best.system}) by question type:")
            print(f"    {'type':<28} {'n':>4} {'system':>8} {'control':>8} {'lift':>8}")
            for t, d in sorted(best.per_type.items(), key=lambda x: -x[1]["lift"]):
                print(f"    {t:<28} {int(d['n']):>4} {d['system_acc']:>8.3f} "
                      f"{d['control_acc']:>8.3f} {d['lift']:>+8.3f}")

    if reports:
        out = Path(args.out)
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(reports, indent=2, default=str))
        print(f"\nwrote {out}")
    print("\n* = lift is significant at p<0.05 with a CI excluding zero")
    print(f"controls: {', '.join(CONTROL_ARMS)}")


if __name__ == "__main__":
    main()
