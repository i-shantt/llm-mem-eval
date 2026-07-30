"""Compute judge/human agreement from a hand-labelled worksheet.

OPTIONAL. The main eval path grades deterministically and needs no labels at
all -- see memllm/eval/grade.py and scripts/audit_graders.py. This script exists
for the case where you want human ground truth on the LLM judge specifically.

If you do, label the *disagreements* file that run_e2e_eval.py writes when
--judge-backend is set, not the full worksheet. Where the deterministic grader
and the judge already agree, a human label almost always confirms both, so those
labels cost effort and buy nothing.

    python scripts/validate_judge.py \
        --results results/e2e_hybrid_k10_n100.json \
        --worksheet results/e2e_hybrid_k10_n100_disagreements.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memllm.eval.judge import JudgedAnswer, judge_agreement  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--worksheet", required=True)
    args = ap.parse_args()

    payload = json.loads(Path(args.results).read_text())
    records = payload.get("records") or payload.get("answers") or []
    judged = [
        JudgedAnswer(
            question_id=r["question_id"], question=r["question"], gold=r["gold"],
            pred=r["pred"], is_abstention=r["is_abstention"],
            verdict=r.get("verdict", r.get("judge")),
            raw_verdict=r.get("raw_verdict", ""),
        )
        for r in records
    ]
    if not any(j.verdict is not None for j in judged):
        raise SystemExit(
            f"{args.results} has no LLM judge verdicts. Re-run "
            "scripts/run_e2e_eval.py with --judge-backend, or just use the "
            "deterministic grade (scripts/audit_graders.py reports its error "
            "rates and needs no labels)."
        )

    stats = judge_agreement(judged, args.worksheet)
    print(json.dumps(stats, indent=2))

    if stats.get("n_comparable", 0) == 0:
        raise SystemExit(
            "\nNo labelled examples yet. Open the worksheet and set "
            "`human_label` to true or false on each line."
        )

    payload["judge_validation"] = stats
    payload.pop("judged_accuracy_note", None)
    Path(args.results).write_text(json.dumps(payload, indent=2))
    print(f"\nmerged into {args.results}")

    kappa = stats["cohens_kappa"]
    fa = stats.get("false_accept_rate")
    print(f"\nCohen's kappa: {kappa:.3f}", end="  ")
    if kappa >= 0.8:
        print("(strong -- judged accuracy is reportable)")
    elif kappa >= 0.6:
        print("(moderate -- report the kappa alongside every judged number)")
    else:
        print("(weak -- do NOT report judged accuracy; fix the judge first)")
    if fa is not None:
        print(f"False-accept rate: {fa:.3f} "
              f"(LoCoMo's judge was audited at up to 0.63)")


if __name__ == "__main__":
    main()
