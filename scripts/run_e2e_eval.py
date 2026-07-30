"""End-to-end evaluation: retrieve, answer with a local LLM, grade.

    python scripts/run_e2e_eval.py --limit 100 --retriever hybrid \
        --answer-backend ollama:qwen2.5:7b-instruct

Grading is deterministic by default -- no LLM judge, no API key, no hand labels,
and identical numbers on every re-run. Its false-accept and false-reject rates
are measured by scripts/audit_graders.py, so the grader is characterised rather
than trusted.

`--judge-backend` additionally runs an LLM judge. Its only real use is
cross-checking: where the two graders disagree is where a human label would
actually teach you something, and that set is a fraction of the dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memllm.cost import CostLedger  # noqa: E402
from memllm.data.loader import load_examples, stratified_subset  # noqa: E402
from memllm.eval.grade import grade, is_extractive, token_f1  # noqa: E402
from memllm.eval.judge import (  # noqa: E402
    JudgedAnswer,
    export_labelling_worksheet,
    judge_answer,
)
from memllm.generate.backends import build_backend  # noqa: E402

ANSWER_PROMPT = """Here are excerpts from earlier conversations with the user, \
most relevant first.

{context}

Today's date is {date}.

Using only the excerpts above, answer the user's question. If the excerpts do \
not contain the answer, say you don't know.

Question: {question}
Answer concisely."""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/longmemeval_s")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--retriever", default="hybrid")
    ap.add_argument("--granularity", default="turn")
    ap.add_argument("--k", type=int, default=10,
                   help="retrieved units placed in the prompt")
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--recency-weight", type=float, default=0.0)
    ap.add_argument("--answer-backend", default="hf:Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--judge-backend", default=None,
                   help="optional LLM judge, for cross-checking the "
                        "deterministic grader; off by default")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    # Reuse the retrieval eval's builder so both paths stay identical.
    from run_retrieval_eval import build_retriever  # noqa: E402
    from memllm.retrieval.embed_cache import EmbeddingCache  # noqa: E402

    examples = stratified_subset(
        load_examples(args.data), args.limit, seed=args.seed
    )
    cache = EmbeddingCache(enabled=True)
    retriever = build_retriever(args.retriever, args, cache=cache)
    if hasattr(retriever, "warmup"):
        retriever.warmup()

    answerer = build_backend(args.answer_backend, max_new_tokens=args.max_new_tokens)
    judger = build_backend(args.judge_backend) if args.judge_backend else None
    print(f"answer backend: {answerer.name}")
    print(f"judge backend:  {judger.name if judger else 'none (deterministic grading)'}")
    print(f"{len(examples)} examples, retriever={args.retriever}, k={args.k}")

    ledger = CostLedger()
    judged: list[JudgedAnswer] = []
    records: list[dict] = []
    n_det_correct = n_det_graded = n_not_gradable = 0
    n_correct = n_parseable = 0
    f1_sum = 0.0
    t0 = time.perf_counter()

    for i, ex in enumerate(examples, 1):
        units = ex.units(args.granularity)
        by_id = {u.unit_id: u for u in units}
        retriever.index(units, ledger, f"{ex.question_id}|{args.granularity}")
        hits = retriever.search(ex.question, args.k, ledger, ex.question_date)

        context = "\n\n".join(
            f"[{by_id[uid].session_date}] {by_id[uid].text}"
            for uid, _ in hits if uid in by_id
        )
        prompt = ANSWER_PROMPT.format(
            context=context, date=ex.question_date, question=ex.question
        )
        gen = answerer.generate(prompt)
        ledger.add_llm("read", gen.prompt_tokens, gen.completion_tokens)

        gold = str(ex.answer)
        det = grade(gen.text, gold, ex.is_abstention)
        if det is None:
            n_not_gradable += 1
        else:
            n_det_graded += 1
            n_det_correct += int(det)
        f1_sum += token_f1(gen.text, gold)

        verdict, raw = (None, "")
        if judger is not None:
            verdict, raw = judge_answer(
                judger, ex.question, gold, gen.text, ex.is_abstention
            )
            judged.append(JudgedAnswer(
                question_id=ex.question_id, question=ex.question, gold=gold,
                pred=gen.text, is_abstention=ex.is_abstention,
                verdict=verdict, raw_verdict=raw,
            ))
            if verdict is not None:
                n_parseable += 1
                n_correct += int(verdict)

        records.append({
            "question_id": ex.question_id, "question": ex.question,
            "question_type": ex.question_type, "gold": gold, "pred": gen.text,
            "is_abstention": ex.is_abstention,
            "extractive": is_extractive(gold),
            "deterministic": det, "judge": verdict, "raw_verdict": raw,
        })

        if i % 5 == 0 or i == len(examples):
            acc = n_det_correct / max(n_det_graded, 1)
            print(f"  {i}/{len(examples)}  acc={acc:.3f}  "
                  f"{time.perf_counter()-t0:.0f}s", flush=True)

    tag = args.tag or f"e2e_{args.retriever}_k{args.k}_n{len(examples)}"
    n = len(examples)

    # Per-question-type accuracy, so a weak overall number can be attributed.
    by_type: dict[str, dict[str, int]] = {}
    for r in records:
        if r["deterministic"] is None:
            continue
        b = by_type.setdefault(r["question_type"], {"n": 0, "correct": 0})
        b["n"] += 1
        b["correct"] += int(r["deterministic"])

    payload = {
        "config": vars(args) | {"answer_backend_name": answerer.name,
                                "judge_backend_name": judger.name if judger else None},
        "n_examples": n,
        "grader": "deterministic (memllm.eval.grade); audit in "
                  "results/grader_audit.json",
        "accuracy": n_det_correct / max(n_det_graded, 1),
        "n_graded": n_det_graded,
        "n_not_gradable": n_not_gradable,
        "not_gradable_note": "abstractive gold answers with no checkable surface "
                             "form; excluded rather than scored as wrong",
        "token_f1_mean": f1_sum / n,
        "accuracy_by_question_type": {
            t: {**v, "accuracy": v["correct"] / v["n"]}
            for t, v in sorted(by_type.items())
        },
        "cost_total": ledger.to_dict(),
        "read_tokens_per_query": (
            ledger.read.llm_prompt_tokens + ledger.read.llm_completion_tokens
        ) / n,
        "records": records,
    }
    if judger is not None:
        from memllm.eval.grader_audit import find_disagreements
        disagreements = find_disagreements(records)
        payload["llm_judge"] = {
            "backend": judger.name,
            "n_parseable_verdicts": n_parseable,
            "accuracy": n_correct / max(n_parseable, 1),
            "n_disagreements_with_deterministic": len(disagreements),
            "disagreement_rate": len(disagreements) / max(n_det_graded, 1),
        }

    out = Path("results") / f"{tag}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))

    print(f"\naccuracy: {payload['accuracy']:.3f} "
          f"({n_det_graded} graded, {n_not_gradable} not gradable)")
    print(f"token-F1: {payload['token_f1_mean']:.3f}")
    print(f"read tokens/query: {payload['read_tokens_per_query']:.0f}")
    for t, v in payload["accuracy_by_question_type"].items():
        print(f"  {t:<28} {v['accuracy']:.3f}  (n={v['n']})")
    print(f"\nwrote {out}")

    if judger is not None:
        lj = payload["llm_judge"]
        print(f"\nLLM judge ({lj['backend']}): {lj['accuracy']:.3f}, disagrees "
              f"with the deterministic grader on {lj['n_disagreements_with_deterministic']} "
              f"of {n_det_graded} ({lj['disagreement_rate']:.1%})")
        ws = export_labelling_worksheet(
            [j for j in judged
             if any(d["question_id"] == j.question_id for d in disagreements)],
            f"results/{tag}_disagreements.jsonl",
        )
        print(f"Only the disagreements are worth a human label: {ws}")
    else:
        print("Grading was deterministic -- reproducible, and no labels needed.")


if __name__ == "__main__":
    main()
