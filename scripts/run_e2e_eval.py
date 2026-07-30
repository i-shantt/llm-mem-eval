"""End-to-end evaluation: retrieve, answer with a local LLM, judge.

    python scripts/run_e2e_eval.py --limit 50 --retriever hybrid \
        --answer-backend ollama:qwen2.5:7b-instruct \
        --judge-backend ollama:qwen2.5:7b-instruct

Also emits a hand-labelling worksheet so the judge can be validated. No judged
number in this repo is reported without its judge/human agreement.
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
                   help="defaults to the answer backend")
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
    judger = build_backend(args.judge_backend or args.answer_backend)
    print(f"answer backend: {answerer.name}\njudge backend:  {judger.name}")
    print(f"{len(examples)} examples, retriever={args.retriever}, k={args.k}")

    ledger = CostLedger()
    judged: list[JudgedAnswer] = []
    n_correct = 0
    n_parseable = 0
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

        verdict, raw = judge_answer(
            judger, ex.question, ex.answer, gen.text, ex.is_abstention
        )
        judged.append(JudgedAnswer(
            question_id=ex.question_id, question=ex.question, gold=ex.answer,
            pred=gen.text, is_abstention=ex.is_abstention,
            verdict=verdict, raw_verdict=raw,
        ))
        if verdict is not None:
            n_parseable += 1
            n_correct += int(verdict)

        if i % 5 == 0 or i == len(examples):
            acc = n_correct / max(n_parseable, 1)
            print(f"  {i}/{len(examples)}  judged_acc={acc:.3f}  "
                  f"{time.perf_counter()-t0:.0f}s", flush=True)

    tag = args.tag or f"e2e_{args.retriever}_k{args.k}_n{len(examples)}"
    n = len(examples)
    payload = {
        "config": vars(args) | {"answer_backend_name": answerer.name,
                                "judge_backend_name": judger.name},
        "n_examples": n,
        "n_parseable_verdicts": n_parseable,
        "judged_accuracy": n_correct / max(n_parseable, 1),
        "judged_accuracy_note": "NOT VALID until judge/human agreement is "
                                "computed via scripts/validate_judge.py",
        "cost_total": ledger.to_dict(),
        "read_tokens_per_query": (
            ledger.read.llm_prompt_tokens + ledger.read.llm_completion_tokens
        ) / n,
        "answers": [
            {"question_id": j.question_id, "question": j.question,
             "gold": j.gold, "pred": j.pred, "is_abstention": j.is_abstention,
             "verdict": j.verdict, "raw_verdict": j.raw_verdict}
            for j in judged
        ],
    }
    out = Path("results") / f"{tag}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))

    ws = export_labelling_worksheet(judged, f"results/{tag}_worksheet.jsonl")

    print(f"\njudged accuracy: {payload['judged_accuracy']:.3f} "
          f"({n_parseable}/{n} verdicts parseable)")
    print(f"read tokens/query: {payload['read_tokens_per_query']:.0f}")
    print(f"wrote {out}")
    print(f"\nNEXT: hand-label {ws} (set human_label true/false), then run")
    print(f"  python scripts/validate_judge.py --results {out} --worksheet {ws}")
    print("The accuracy above is not reportable until that agreement is known.")


if __name__ == "__main__":
    main()
