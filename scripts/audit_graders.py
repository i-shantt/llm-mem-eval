"""Audit the deterministic grader (and optionally an LLM judge) with zero labels.

    # deterministic grader only -- instant, no model needed
    python scripts/audit_graders.py

    # include an LLM judge, to compare the two on identical cases
    python scripts/audit_graders.py --judge-backend ollama:llama3.1:8b-instruct

Verdicts here are known by construction, so no hand-labelling is involved. See
memllm/eval/grader_audit.py for why that is legitimate and where it stops being
sufficient.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memllm.data.loader import load_examples  # noqa: E402
from memllm.eval.grade import grade  # noqa: E402
from memllm.eval.grader_audit import audit_grader, build_audit_cases  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/longmemeval_s")
    ap.add_argument("--per-type", type=int, default=60)
    ap.add_argument("--judge-backend", default=None,
                   help="optional; audits an LLM judge on the same cases")
    ap.add_argument("--judge-limit", type=int, default=400,
                   help="LLM judging is slow; cap the cases sent to it")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/grader_audit.json")
    args = ap.parse_args()

    examples = load_examples(args.data)
    cases = build_audit_cases(examples, seed=args.seed, per_type=args.per_type)
    abst = {c.question for c in cases if c.kind.endswith("abstention")}
    print(f"{len(cases)} constructed cases from {len(examples)} examples "
          f"({len(abst)} abstention questions)\n")

    # The grader needs to know whether a case is an abstention question; kind
    # encodes that, since that is how the case was built.
    def det(c):
        return grade(c.pred, c.gold, is_abstention=c.kind.endswith("abstention"),
                     question=c.question)

    report = {"n_examples": len(examples), "graders": {}}
    report["graders"]["deterministic"] = audit_grader(cases, det)

    if args.judge_backend:
        from memllm.eval.judge import judge_answer
        from memllm.generate.backends import build_backend

        backend = build_backend(args.judge_backend)
        subset = cases[:args.judge_limit]
        print(f"auditing judge {backend.name} on {len(subset)} cases...")
        memo: dict[tuple, bool | None] = {}

        def judge(c):
            key = (c.question, c.gold, c.pred)
            if key not in memo:
                verdict, _ = judge_answer(
                    backend, c.question, c.gold, c.pred,
                    is_abstention=c.kind.endswith("abstention"),
                )
                memo[key] = verdict
                if len(memo) % 25 == 0:
                    print(f"  {len(memo)}/{len(subset)}", flush=True)
            return memo[key]

        report["graders"][f"judge:{backend.name}"] = audit_grader(subset, judge)
        report["judge_audit_n_cases"] = len(subset)

    Path(args.out).parent.mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))

    def fmt(x):
        return "n/a" if x is None else f"{x:.3f}"

    for name, r in report["graders"].items():
        print(f"\n=== {name} ===")
        print(f"  cases graded      {r['n_graded']}/{r['n_cases']} "
              f"({r['n_abstained']} abstained)")
        print(f"  accuracy          {r['accuracy']:.3f}")
        print(f"  false accept      {fmt(r['false_accept_rate'])}"
              "   <- known-wrong answers accepted")
        print(f"  false reject      {fmt(r['false_reject_rate'])}"
              "   <- all known-good answers")
        print(f"  false reject HARD {fmt(r['false_reject_rate_hard'])}"
              "   <- rewrites only; the honest number")
        print("  by case kind:")
        for kind, v in r["by_kind"].items():
            acc = v["accuracy"]
            print(f"    {kind:<28} {'n/a' if acc is None else f'{acc:.3f}'}  "
                  f"(n={v['n']})")

    print(f"\nwrote {args.out}")

    det_r = report["graders"]["deterministic"]
    if (det_r["false_accept_rate"] or 0) > 0.05:
        print("\nWARNING: deterministic false-accept rate above 0.05 -- the "
              "grader is too lenient to report accuracy with.")


if __name__ == "__main__":
    main()
