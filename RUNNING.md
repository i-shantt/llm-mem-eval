# What is committed but not run

This repo distinguishes between code that has produced the numbers in the README
and code that has not. Everything in this file is the second kind. No artifact in
`results/` was produced by any of it, and no claim in the README depends on it.

Keeping unrun code is a deliberate choice over deleting it: the design decisions
are the reviewable part, and the alternative — running it badly against a
deadline and publishing the output — produces a worse artifact than saying what
was not done.

---

## 1. The Mem0 arm — `memllm/write/mem0_adapter.py`

**Status: written, unit-tested against a stub, never executed against mem0.**

### Why it was not run

Running Mem0's open-source SDK with a 7B extraction model and a 33M-parameter
embedder, on free GPU, would produce a number. It would also be a number for a
configuration Mem0 has already said in writing is not the one their published
results describe:

> "Scores reflect Mem0's managed platform, which includes proprietary
> optimizations not available in the open-source SDK."
>
> "Open-source users should expect directionally similar gains but not identical
> numbers."
>
> — [The Token-Efficient Memory Algorithm](https://mem0.ai/blog/mem0-the-token-efficient-memory-algorithm)

And on why small benchmarks move so much with configuration, from the same post:
scores on LoCoMo and LongMemEval "can be materially improved by aggressive
retrieval strategies, larger context windows, or frontier models."

That describes the setup available here almost exactly — a shallower retrieval
budget, a lighter extraction model, and a smaller embedder than their stack. A
low survival number from it would be evidence about *this configuration*, and it
would be read as evidence about Mem0. Publishing it against a deadline, without
the matched-configuration run that would separate the two, is not a measurement
worth making.

The same post notes the framework is open source "so anyone can reproduce the
numbers independently," which is the right way to do this and is what the
procedure below sets up.

### What to match when it is run

Their documented OSS stack, from
[`mem0ai/memory-benchmarks`](https://github.com/mem0ai/memory-benchmarks) — their
extraction-model ablation holds all of this fixed and varies only the extractor:

| component | their setting | what this adapter currently defaults to |
|---|---|---|
| embedder | Qwen 600M | `bge-small-en-v1.5` (384d) — **below their stack; fix before running** |
| vector store | Qdrant | Qdrant ✓ |
| answerer + judge | GPT-5 | n/a — survival needs no answerer |
| retrieval budget | top-k 200 | n/a for survival; matters for any end-to-end arm |
| extraction model | `gpt-5-mini` (OSS default) | whatever `MEM0_LLM_MODEL` names |

The embedder gap does not affect **survival**, which is measured over the store
with no retrieval at all. It does affect any retrieval or end-to-end arm built on
the store, and it should be closed rather than caveated.

### Procedure

```bash
pip install "mem0ai==2.0.18" "fastembed>=0.7" "spacy>=3.8"
python -m spacy download en_core_web_sm            # needs network

# mem0's vllm/openai providers are HTTP clients, not in-process: serve first.
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --served-model-name Qwen/Qwen2.5-7B-Instruct \
    --gpu-memory-utilization 0.85 --max-model-len 8192 --port 8000 &

export MEM0_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
export MEM0_BATCH=session

python - <<'PY'
from memllm.write.mem0_adapter import Mem0OssPolicy
r = Mem0OssPolicy.from_spec("mem0_oss_v3_qwen7b").preflight()
print(r.explain()); assert r.ok
PY

python scripts/run_survival_eval.py \
    --policies verbatim_turn truncated_recency_25 leadk_25 mem0_oss_v3_qwen7b
```

**Run the preflight.** All four failure modes it checks are silent in mem0 — the
run completes either way and produces a worse store:

1. a vector store without `keyword_search` drops BM25 for the whole session
   (chroma and faiss do this; qdrant does not);
2. missing `fastembed` drops BM25 *even on qdrant*;
3. a collection predating v3 has no sparse slot;
4. a missing spaCy model turns the entity boost off.

It also checks the one that corrupts the store's contents rather than its
ranking: `add(timestamp=...)` raises in OSS, so the adapter passes the
conversation date through `metadata["created_at"]` *and* patches
`generate_additive_extraction_prompt`. Without both, the extractor resolves
"last week" against today's date — the failure in
[mem0 issue #3944](https://github.com/mem0ai/mem0/issues/3944).

### Cost

At `MEM0_BATCH=session`, ~48 `add()` calls per conversation × 184 conversations
≈ 8,800 LLM calls. `scripts/model_write_cost.py` models ~0.61M prompt tokens per
conversation at that granularity. Pilot on 10 conversations before committing to
the full set.

### Naming

The store is `mem0_oss_v3_qwen7b`, never `mem0` — in code, in payloads, and in
prose, in the success case as much as the failure case. It is the open-source
SDK, not the managed platform that scores 94.4, and an open 7B extractor, not
`gpt-5-mini`.

---

## 2. Other deferred work

**Three-stage decomposition (survival → retrieval → reading).** Survival is a
ceiling on retrieval and accuracy, so the natural next measurement is the
conditional pass rate at each stage over one store. It needs an answering model
and it needs the leakage row — `P(correct | not retrieved)` — published beside
it, or the funnel is decorative. The `none` control arm already measures the
closed-book floor it would be read against.

**Resumable checkpointing.** `results/stores/<id>/records.jsonl` with an
fsync-per-example append and a torn-tail truncation on resume. Not written,
because nothing shipping here runs long enough to need it; it is a prerequisite
for the Mem0 arm, not for anything already done.

**n=500 refresh of the existing retrieval and end-to-end arms.** They are
n=100 stratified. The benchmark audit is n=500. The two are therefore different
question sets, which the README states wherever both appear.

**A recency signal at retrieval time.** Mem0's OSS ranker provably has none:
`created_at`/`updated_at` are stored on every memory and are not inputs to
`score_and_rank`, and `search(reference_date=...)` raises "Platform-only". This
repo's `search()` already takes a `question_date` that every retriever ignores,
and `HybridRetriever`'s `recency_weight` is implemented but quantised to integer
rank-list repetitions and exercised by no committed result. Worth doing properly;
it is a second thread, and splitting focus was the larger risk.

**LexRank / TextRank selection.** `ExtractiveSelectionPolicy` implements lead-k
only. A centrality-based selector would give a third point on the
survival-versus-store-size curve that is neither truncation nor lead-k.
