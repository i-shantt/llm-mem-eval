# memllm — what does LLM memory actually cost?

Every LLM memory system advertises token savings measured on the **read path**:
tokens per query. Almost none report the **write path** — the cost of building
the memory in the first place. That is where the money is.

For a single benchmark instance, published numbers put verbatim retrieval at
roughly **$0.01 and zero LLM calls**, and Mem0 at **~$0.50 with 1000+ LLM
calls** (~1.5M tokens) to construct its memory. Yet the headline comparison is
usually "1,764 tokens per query vs 26,031 for full context — 90% savings."

Those two facts belong in the same table. This repo puts them there.

## The headline

![cost curve](results/cost_curve.png)

Read-path-only accounting is the `n → ∞` limit of this plot, which flatters
exactly the systems that spend the most up front. Measured on LongMemEval
(n=100, turn granularity, top-10 retrieved):

> **Mem0's cheaper read path only repays its write path after ~4,630 queries
> against the same conversation.** No personal assistant ever reaches that.
> Meanwhile a zero-LLM-write-path retriever is **49.7× cheaper per query than
> full context** — at $0.00031/query against $0.01561.

## What the retrieval numbers say

Full tables in [RESULTS.md](RESULTS.md) — regenerated from run artifacts, never
hand-typed. The short version, at n=100:

| system | any_hit@1 | any_hit@10 | MRR | write cost | LLM calls |
|---|---|---|---|---|---|
| random | 0.021 | 0.041 | 0.027 | 0 ms | 0 |
| recency (last-N turns) | 0.000 | 0.062 | 0.012 | 0 ms | 0 |
| **BM25** | **0.546** | 0.825 | 0.634 | **16 ms** | **0** |
| dense (bge-small, 33M) | 0.464 | 0.897 | 0.616 | 9,028 ms | 0 |
| hybrid (BM25+dense, RRF) | 0.536 | **0.907** | **0.649** | 9,043 ms | 0 |
| oracle (ceiling) | 1.000 | 1.000 | 1.000 | — | — |

Three findings worth stating plainly:

1. **BM25 beats a neural embedding model on top-1 precision and MRR** (0.546 vs
   0.464; 0.634 vs 0.616), for **564× less write cost**. The embedding model
   buys deeper recall (`any_hit@10` 0.897 vs 0.825), not better ranking.
2. **BM25 is perfect on knowledge-update questions** (`any_hit@10` = 1.000, vs
   dense 0.938) — the slice the memory-conflict literature treats as the open
   problem.
3. **"Just keep the last N turns" does not work.** Recency scores 0.062, barely
   above random's 0.041. Any system whose gains could come from a recency prior
   needs to prove otherwise.

The weakest slice for every retriever is `single-session-preference` (BM25 MRR
0.158), which is consistent with the constraint-adherence literature: preference
questions are not lexically similar to the turns that answer them.

## Why the evaluation here is judge-free

LongMemEval labels **every turn** with `has_answer`. That is a gold retrieval
label, so retrieval quality is measurable with no LLM judge and no API spend:

- `any_hit@k` — was any evidence turn retrieved in the top k? (the precondition
  for answering at all)
- `recall@k` — what fraction of evidence turns were retrieved?
- `MRR` — how highly was the first evidence turn ranked?

This matters because the field's judges are not trustworthy. An audit of LoCoMo
— the benchmark most memory systems report on — found **6.4% of its answer key
is wrong** and its LLM judge **accepts up to 63% of intentionally wrong
answers**. This project uses LongMemEval instead (it scores knowledge updates
and abstention, which LoCoMo does not).

Questions with no evidence turns (the abstention cases) are **excluded** from
recall rather than scored as 0 or 1, and reported separately.

### End-to-end answers are graded without a judge too

The median gold answer in LongMemEval is **11 characters** (`Target`, `$800`,
`February 14th`). That is short-answer QA, which had a reproducible metric long
before LLM judges existed. `memllm/eval/grade.py` scores answers by normalised
token-span containment: no API key, no GPU for grading, and byte-identical
numbers on every re-run.

Two things make that trustworthy rather than merely cheap.

**It abstains instead of guessing.** Some gold answers are rubric paragraphs
(`"The user would prefer responses that..."`) with no checkable surface form.
Those return `None` and are excluded from the denominator rather than scored
wrong.

**Its error rates are measured, not assumed.** `scripts/audit_graders.py`
rebuilds the LoCoMo audit's method as a script: take a gold answer, substitute a
different question's gold, perturb its numbers, replace it with a refusal — each
result's correct verdict is known *by construction*, so no annotation is needed.
Current numbers over 3,050 constructed cases:

| | rate |
|---|---|
| false accept (known-wrong answers marked correct) | **0.000** |
| false reject on meaning-preserving rewrites | **0.004** |

For comparison, the audited LoCoMo judge's false-accept rate reached **0.63**.

The audit is not decoration — it found and fixed four real defects, two of which
were costing 29 accuracy points on identical model outputs. See
[Honest limits](#honest-limits) for where it stops being sufficient.

## Layout

```
memllm/
  cost.py                    write/read cost ledger + amortisation
  data/loader.py             LongMemEval loading, turn/session granularity
  retrieval/
    bm25.py                  lexical baseline, zero LLM calls
    dense.py                 bge-small-en-v1.5 (33M params), local
    hybrid.py                BM25 + dense via reciprocal rank fusion
    baselines.py             oracle ceiling, recency floor, random floor
  eval/
    retrieval_metrics.py     judge-free metrics from has_answer labels
    grade.py                 deterministic answer grading, no judge or labels
    grader_audit.py          grader error rates from constructed cases
    judge.py                 optional LLM judge, for cross-checking only
  generate/backends.py       local answer generation (transformers / ollama)
scripts/
  run_retrieval_eval.py      run any set of retrievers, emit results/*.json
  run_e2e_eval.py            retrieve -> answer -> grade, with cost accounting
  audit_graders.py           measure grader false-accept / false-reject rates
  make_report.py             regenerate RESULTS.md from run artifacts
  make_cost_curve.py         the write+read amortisation figure
tests/test_harness.py        smoke tests for label integrity + cost accounting
```

## Reproducing

```bash
python -m venv .venv && ./.venv/bin/pip install -r requirements.txt

# ~278MB; 500 questions, each with its own ~123K-token, ~490-turn haystack
./.venv/bin/python -c "from huggingface_hub import hf_hub_download; \
  hf_hub_download('xiaowu0162/longmemeval','longmemeval_s', \
  repo_type='dataset',local_dir='data/raw')"

./.venv/bin/python tests/test_harness.py
./.venv/bin/python scripts/run_retrieval_eval.py \
    --limit 100 --retrievers random recency bm25 dense hybrid oracle
./.venv/bin/python scripts/make_report.py > RESULTS.md
```

Runs on CPU. Uses MPS or CUDA automatically if present. No API key required for
any retrieval result in this repo.

`--limit` takes a **stratified** subset that preserves the question-type
distribution, so the knowledge-update and abstention slices stay intact. Every
reported number carries its own `n`.

## Honest limits

- Comparisons against Mem0, Zep, A-Mem and similar use **their published
  numbers**, not our re-runs of their code. Tables label each number as
  *reported* or *measured*. Reimplementing those systems is out of scope.
- Retrieval quality is not answer quality. A system can retrieve the right turn
  and still answer wrong. That is what the end-to-end arm is for.
- `has_answer` is LongMemEval's own labelling, inheriting whatever errors it
  contains. It is a better label than an LLM judge's opinion, not a perfect one.
- **The grader audit's negatives are easy negatives.** Substituted and perturbed
  answers are wrong in obvious ways; a real model's near-miss can be subtler.
  A 0.000 false-accept rate over constructed cases is *necessary* for trusting
  the grader, not *sufficient*. Two of the four defects the grader had were found
  by reading real model output, not by the audit — so the repo does both.
- Deterministic grading is stricter than a human on genuine paraphrase, so
  absolute accuracies here are a **lower bound**. Every system is graded by the
  identical rule, so the comparisons between systems — which is what this project
  claims — are unaffected by that bias.
- `--judge-backend` adds an LLM judge and reports where the two graders disagree.
  Those disagreements are the only cases a human label would inform; agreement
  cases teach you nothing. If you want human validation, label those (a few per
  hundred) rather than the whole set.
