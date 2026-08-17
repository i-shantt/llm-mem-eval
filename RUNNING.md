# What is committed but not run

This repo distinguishes between code that has produced the numbers in the README
and code that has not. Everything in this file is the second kind. No artifact in
`results/` was produced by any of it, and no claim in the README depends on it.

Keeping unrun code is a deliberate choice over deleting it: the design decisions
are the reviewable part, and the alternative — running it badly against a
deadline and publishing the output — produces a worse artifact than saying what
was not done.

---

## 1. The Mem0 arm — `llm_mem_eval/write/mem0_adapter.py`

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

# Cheapest useful thing the extra buys, and it does not need a server: the
# write-cost model pins mem0's 7,671-token extraction prompt and asserts the
# wheel is the v3 rewrite. This re-derives both from the installed package.
python scripts/verify_write_cost_inputs.py

# mem0's vllm/openai providers are HTTP clients, not in-process: serve first.
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --served-model-name Qwen/Qwen2.5-7B-Instruct \
    --gpu-memory-utilization 0.85 --max-model-len 8192 --port 8000 &

export MEM0_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
export MEM0_BATCH=session

python - <<'PY'
from llm_mem_eval.write.mem0_adapter import Mem0OssPolicy
r = Mem0OssPolicy.from_spec("mem0_oss_v3_qwen7b").preflight()
print(r.explain()); assert r.ok
PY

python scripts/run_survival_eval.py \
    --policies verbatim_turn truncated_recency_25 leadk_25 mem0_oss_v3_qwen7b
```

**Run the preflight.** Every failure it checks for is silent in mem0 — the run
completes either way and produces a worse store. Four of them degrade retrieval:

1. a vector store without `keyword_search` drops BM25 for the whole session.
   `chroma.py` and `faiss.py` never override it, so they inherit the base
   class's "returns None if not supported"; `qdrant.py` does override it. Also
   checkable: qdrant is the only entry under `mem0/configs/vector_stores/` whose
   config accepts a local `path`, so it is the only store this arm can use
   without standing up a server — mem0's milvus config takes a server `url`
   rather than a milvus-lite file;
2. missing `fastembed` drops BM25 *even on qdrant*;
3. a collection predating v3 has no sparse slot;
4. a missing spaCy model turns the entity boost off.

Two more checks cover the failure that corrupts the store's *contents* rather
than its ranking. `add(timestamp=...)` is hard-rejected in OSS with a "Platform-only"
error, so the adapter has to pin the conversation date in **three** places:

1. `metadata["created_at"]`, which survives `_strip_identity_keys` and overrides
   the stored row's `datetime.now()` default;
2. `generate_additive_extraction_prompt(timestamp=...)`, the prompt's
   **Observation Date**;
3. `generate_additive_extraction_prompt(current_date=...)`, the prompt's
   **Current Date**.

The third was missing at first, and the omission is instructive. `_resolve_dates`
defaults each date independently, so binding only `timestamp` produced a prompt
reading *"Observation Date 2023-05-20 / Current Date 2026-08-17"* — a three-year
gap invented by the harness, with wall-clock today still sitting in the prompt as
the anchor a model resolves "last week" against. It looks like the fix and is not
one. During ingestion the conversation's date *is* the present, so both have to be
it. The preflight now also asserts that both keyword names still exist, because
`functools.partial` binds a renamed kwarg silently.

Without all three, the extractor resolves relative time against today and every
relative reference in the store is wrong — the failure in
[mem0 issue #3944](https://github.com/mem0ai/mem0/issues/3944).

**One more that reading the code caught, and that no preflight would have.**
`Memory.get_all(top_k=...)` **defaults to 20**. `build()` originally called it
without a `top_k`, so it would have read back 20 memories from a conversation of
~490 turns and measured answer survival over a fraction of the store. Nothing
would have raised; the arm would simply have reported that Mem0's extraction
loses most answers, which is both false and the single most misleading result this
repo could publish. It now passes an explicit `top_k` and raises if the result
saturates it.

That is the argument for reviewing unrun code rather than trusting that running it
would have surfaced the problem. Neither of these two defects announces itself at
runtime: one produces a plausible store with the wrong dates, the other a
plausible store with most of it missing.

### Cost

At `MEM0_BATCH=session`, ~48 `add()` calls per conversation × 184 conversations
≈ 8,800 LLM calls. `scripts/model_write_cost.py` models ~0.61M prompt tokens per
conversation at that granularity. Pilot on 10 conversations before committing to
the full set.

Priced for whatever extractor you point it at, from
[`results/write_cost_model.json`](results/write_cost_model.json). Per
conversation at `session` batching:

| | tokens |
|---|---|
| prompt | 592,877 |
| of which the 7,671-token system prompt, re-sent every call | 366,198 |
| completion (assumed 300/call — the one estimated term) | 14,321 |

So the bill is `592,877 × input_price + 14,321 × output_price` per conversation,
times however many conversations you run. Two consequences worth planning around:

- **Prompt caching roughly halves it**, because **62% of the prompt tokens** are a
  byte-identical prefix on every one of the ~48 calls. (The README's batching table
  says 60% for the same quantity; that one is a share of prompt *plus* completion,
  which is what `share_system_prompt` in the artifact means.) Whether caching
  applies depends on the provider and on whether mem0's adapter for it marks the
  prefix cacheable — check before assuming it.
- **The extractor's tier moves the total by more than the batching does at fixed
  batching.** A frontier extractor is the wrong choice on the merits, not just the
  cost: mem0's OSS default is `gpt-5-mini`, so a frontier model measures a
  configuration nobody deploys, and the deviation from their documented stack is
  larger, not smaller. Match the tier, and name whatever you used in the arm name.

**Deliberately not priced here in dollars.** The one figure in this repo that
cannot be traced to a primary source is a vendor's list price, and
`model_write_cost.py` already says so about the `gpt-5-mini` rates it uses for the
README table. Adding a second vendor's prices would compound an input the repo
cannot verify. The token counts above are exact except the completion term; the
arithmetic is yours.

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

**n=500 refresh of the end-to-end arms.** No longer deferred for retrieval: the
retrieval sweep and the cost figure's token inputs are now measured on all 500
questions, which is what the benchmark audit was always measured on. The
end-to-end arms are still a stratified n=100, and that is a compute limit rather
than a choice — the retrieval sweep is CPU-only and takes about an hour per
granularity, whereas the end-to-end ladder is four model sizes through an
answering model. So the retrieval table and the end-to-end table remain different
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
