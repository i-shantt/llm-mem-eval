"""Compute judge/human agreement from a hand-labelled worksheet.

    python scripts/validate_judge.py \
        --results results/e2e_hybrid_k10_n50.json \
        --worksheet results/e2e_hybrid_k10_n50_worksheet.jsonl
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
    judged = [
        JudgedAnswer(
            question_id=a["question_id"], question=a["question"], gold=a["gold"],
            pred=a["pred"], is_abstention=a["is_abstention"],
            verdict=a["verdict"], raw_verdict=a["raw_verdict"],
        )
        for a in payload["answers"]
    ]

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
