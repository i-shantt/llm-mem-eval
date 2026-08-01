"""Re-grade stored e2e arms from their saved predictions. No model, no GPU.

Every arm keeps the model's full `pred` text, so a change to the grader can be
applied to every result ever produced without re-running anything. That is the
whole reason `pred` is stored: a grader is a hypothesis about what counts as a
correct answer, and hypotheses get revised.

Prints what moved and why. `--write` applies it; the default is a dry run,
because silently rewriting measured results is how a repo starts lying.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memllm.eval.grade import grade, token_f1  # noqa: E402


def regrade(payload: dict) -> tuple[dict, list[dict]]:
    """Return (new_payload, changed_records). Does not mutate the input."""
    records = [dict(r) for r in payload["records"]]
    changed = []

    n_correct = n_graded = n_not_gradable = 0
    f1_sum = 0.0
    for r in records:
        before = r["deterministic"]
        after = grade(r["pred"], r["gold"], r["is_abstention"], r.get("question"))
        if after != before:
            changed.append({"question_id": r["question_id"],
                            "question_type": r["question_type"],
                            "gold": r["gold"], "pred": r["pred"],
                            "before": before, "after": after})
        r["deterministic"] = after
        if after is None:
            n_not_gradable += 1
        else:
            n_graded += 1
            n_correct += int(after)
        f1_sum += token_f1(r["pred"], r["gold"])

    by_type: dict[str, dict] = {}
    for r in records:
        if r["deterministic"] is None:
            continue
        b = by_type.setdefault(r["question_type"], {"n": 0, "correct": 0})
        b["n"] += 1
        b["correct"] += int(r["deterministic"])

    new = dict(payload)
    new["records"] = records
    new["accuracy"] = n_correct / max(n_graded, 1)
    new["n_graded"] = n_graded
    new["n_not_gradable"] = n_not_gradable
    new["token_f1_mean"] = f1_sum / payload["n_examples"]
    new["accuracy_by_question_type"] = {
        t: {**v, "accuracy": v["correct"] / v["n"]} for t, v in sorted(by_type.items())
    }
    return new, changed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--write", action="store_true",
                    help="apply the re-grade; default is a dry run")
    args = ap.parse_args()

    files = sorted(Path(args.results).glob("e2e_*.json"))
    if not files:
        sys.exit(f"no e2e arms under {args.results}/")

    all_changed: list[dict] = []
    moved = 0
    print(f"{'arm':44s} {'accuracy':>18s}  {'graded':>7s}")
    for f in files:
        payload = json.loads(f.read_text())
        if "records" not in payload:
            continue
        new, changed = regrade(payload)
        before, after = payload["accuracy"], new["accuracy"]
        flag = ""
        if changed:
            moved += 1
            flag = f"   {before:.4f} -> {after:.4f}  ({len(changed)} verdicts)"
            all_changed.extend({**c, "arm": f.stem} for c in changed)
        print(f"{f.stem:44s} {after:18.4f}  {new['n_graded']:7d}{flag}")
        if args.write:
            f.write_text(json.dumps(new, indent=2))

    print(f"\n{moved}/{len(files)} arms moved; "
          f"{len(all_changed)} individual verdicts changed")
    if all_changed:
        print("\nEvery changed verdict, so none of this is taken on trust:")
        for c in all_changed:
            print(f"\n  {c['arm']}  [{c['question_type']}]  "
                  f"{c['before']} -> {c['after']}")
            print(f"    gold: {c['gold']!r}")
            print(f"    pred: {c['pred'][:160]!r}")
    if not args.write:
        print("\n(dry run -- re-run with --write to apply)")


if __name__ == "__main__":
    main()
