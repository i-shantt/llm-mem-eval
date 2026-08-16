"""Run judge-free retrieval evaluation over LongMemEval.

    python scripts/run_retrieval_eval.py --limit 50 --retrievers bm25 \
        --granularity turn

Every run writes results/<tag>.json with metrics plus full write/read cost.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memllm.cost import CostLedger  # noqa: E402
from memllm.data.loader import (  # noqa: E402
    load_examples,
    stratified_subset,
)
from memllm.eval.retrieval_metrics import aggregate, score_example  # noqa: E402


def build_retriever(name: str, args, cache=None):
    if name == "bm25":
        from memllm.retrieval.bm25 import BM25Retriever

        return BM25Retriever()
    if name == "dense":
        from memllm.retrieval.dense import DenseRetriever

        return DenseRetriever(
            model_name=args.embed_model, device=args.device, cache=cache
        )
    if name == "hybrid":
        from memllm.retrieval.hybrid import HybridRetriever

        return HybridRetriever(
            model_name=args.embed_model,
            device=args.device,
            recency_weight=args.recency_weight,
            cache=cache,
        )
    if name in ("oracle", "recency", "random", "none"):
        from memllm.retrieval import baselines

        cls = {
            "oracle": baselines.OracleRetriever,
            "recency": baselines.RecencyRetriever,
            "random": baselines.RandomRetriever,
            "none": baselines.NoMemoryRetriever,
        }[name]
        return cls()
    raise ValueError(f"unknown retriever: {name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/longmemeval_s")
    ap.add_argument("--limit", type=int, default=50,
                    help="stratified subset size; 0 = all 500")
    ap.add_argument("--retrievers", nargs="+", default=["bm25"])
    ap.add_argument("--granularity", default="turn",
                    choices=["turn", "user_turn", "session"])
    ap.add_argument("--k", type=int, default=20, help="depth to retrieve")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10, 20],
                    help="depths to report. Granularities are only comparable "
                         "at a matched read-token budget, and units differ in "
                         "size by ~40x, so that means a different k for each: "
                         "~2600 tokens is 12 turns, 48 user turns, or 1 session")
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--recency-weight", type=float, default=0.0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-cache", action="store_true",
                    help="disable the embedding cache; use for authoritative "
                         "wall-clock timing runs")
    args = ap.parse_args()

    examples = load_examples(args.data)
    if args.limit:
        examples = stratified_subset(examples, args.limit, seed=args.seed)
    print(f"{len(examples)} examples, granularity={args.granularity}, k={args.k}")

    from memllm.retrieval.embed_cache import EmbeddingCache

    cache = EmbeddingCache(enabled=not args.no_cache)

    all_results = {}
    for rname in args.retrievers:
        retriever = build_retriever(rname, args, cache=cache)
        # Model load must not be billed to the write or read path.
        if hasattr(retriever, "warmup"):
            retriever.warmup()
        ledger = CostLedger()
        results = []
        t0 = time.perf_counter()

        for i, ex in enumerate(examples, 1):
            units = ex.units(args.granularity)
            cache_key = f"{ex.question_id}|{args.granularity}"
            # Write path: indexing the haystack.
            retriever.index(units, ledger, cache_key)
            # Read path: one query.
            ledger.read.llm_calls += 0  # retrieval-only arm makes no LLM calls
            hits = retriever.search(
                ex.question, args.k, ledger, question_date=ex.question_date
            )
            results.append(
                score_example(
                    ex.question_id, ex.question_type, ex.is_abstention, units, hits
                )
            )
            if i % 10 == 0 or i == len(examples):
                el = time.perf_counter() - t0
                print(f"  [{rname}] {i}/{len(examples)}  {el:.0f}s", flush=True)

        metrics = aggregate(results, ks=tuple(args.ks))
        n = len(examples)
        payload = {
            "retriever": rname,
            "config": {
                "granularity": args.granularity,
                "k": args.k,
                "limit": args.limit,
                "embed_model": args.embed_model if rname != "bm25" else None,
                "recency_weight": args.recency_weight,
                "seed": args.seed,
            },
            "metrics": metrics,
            # Aggregates cannot tell a retrieval failure from a generation
            # failure on any single question. These can: an end-to-end miss at
            # recall@10 == 1.0 is the model's, not the retriever's.
            "per_question": [
                {
                    "question_id": r.question_id,
                    "question_type": r.question_type,
                    "n_evidence": r.n_evidence,
                    "k": args.k,
                    "recall@k": r.recall_at(args.k),
                    "any_hit@k": r.any_hit_at(args.k),
                    # kept under the old names too so existing joins still work
                    "recall@10": r.recall_at(10),
                    "any_hit@10": r.any_hit_at(10),
                    "first_evidence_rank": (
                        min(r.evidence_ranks) if r.evidence_ranks else None
                    ),
                }
                for r in results
            ],
            "embed_cache": cache.stats(),
            "timing_is_authoritative": not cache.used_replayed_timings,
            "cost_total": ledger.to_dict(),
            "cost_per_example": {
                "write_wall_clock_s": ledger.write.wall_clock_s / n,
                "read_wall_clock_s": ledger.read.wall_clock_s / n,
                "write_llm_calls": ledger.write.llm_calls / n,
                "write_llm_tokens": (
                    ledger.write.llm_prompt_tokens
                    + ledger.write.llm_completion_tokens
                ) / n,
                "write_embed_tokens": ledger.write.embed_tokens / n,
            },
        }
        all_results[rname] = payload

        print(f"\n=== {rname} ({args.granularity}) ===")
        print(f"  scorable {metrics['n_scorable']}/{metrics['n_total']} "
              f"(zero-evidence excluded: {metrics['n_zero_evidence']})")
        for k in (1, 5, 10, 20):
            if f"any_hit@{k}" in metrics:
                print(f"  any_hit@{k:<3} {metrics[f'any_hit@{k}']:.3f}   "
                      f"recall@{k:<3} {metrics[f'recall@{k}']:.3f}")
        print(f"  MRR {metrics['mrr']:.3f}")
        print(f"  write: {ledger.write.wall_clock_s/n*1000:.0f} ms/ex, "
              f"{ledger.write.llm_calls/n:.1f} LLM calls/ex")
        print(f"  read:  {ledger.read.wall_clock_s/n*1000:.0f} ms/query")

    tag = args.tag or f"{'_'.join(args.retrievers)}_{args.granularity}_n{len(examples)}"
    out = Path("results") / f"{tag}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(all_results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
