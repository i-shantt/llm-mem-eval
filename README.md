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

### What happened when we labelled the disagreements

The design says: run both graders, and hand-label only where they disagree.
We did that. On one arm (`qwen2.5:7b-instruct` answering, hybrid retrieval,
k=10, n=100, judged by `llama3.1:8b-instruct`) the two graders disagreed on
**36 answers**. All 36 were labelled by hand — about fifteen minutes of work,
against the ~100 labels a from-scratch validation would have cost.

**The deterministic grader was right on 27 of 36. The LLM judge was right on 9.**

Of the 36, 27 answers were unambiguously wrong. **The judge accepted 26 of
them:**

| question | gold | model answered | judge |
|---|---|---|---|
| pre-1920 coins in my collection | 38 | 37 | CORRECT |
| what time do I go to the gym | 6:00 pm | 7:00 pm | CORRECT |
| Saturday wake-up time | 7:30 am | 7:45 am | CORRECT |
| total online courses completed | 20 | 12 | CORRECT |
| total views on YouTube + TikTok | 1,998 | 2,018 | CORRECT |
| who gave me the jewelry | my aunt | *"I don't know"* | CORRECT |
| how much faster was my 5K | 10 minutes | *"I don't have enough details"* | CORRECT |
| what did I save on the handbag | $300 | *"I can't determine"* | CORRECT |

Ten of the 26 are the judge marking a **flat refusal** correct on a question
that has a real gold answer. That is not leniency, it is a broken grader. The
audited LoCoMo judge accepted 63% of wrong answers; this one accepted 26 of 27.

State the caveat with the number: the disagreement set is *selected* for
disagreement, so 26/27 bounds the judge's false-accept rate on hard cases, not
over the whole run. Bounding it over the whole run needs labels on the
agreement set too. It is disqualifying either way — **no judged accuracy from
an 8B judge appears anywhere in this repo.**

### The one false accept the constructed audit could not have caught

The deterministic grader was not clean either. It was wrong on 9 of the 36:
eight false rejects (all gold-surface-form handling — `55-inch` vs "55 inches",
`Friday` vs "Fridays", `University of California, Los Angeles (UCLA)` vs
"UCLA"), and one **false accept**:

> **Q:** Which group did I join first, 'Page Turners' or 'Marketing Professionals'?
> **Gold:** Page Turners
> **Model:** "You joined the **'Marketing Professionals'** group first, as you
> asked about marketing resources […] before mentioning the 'Page Turners' book
> club group."
> **Deterministic grader: CORRECT.** ❌

The gold string is present, so containment accepts it — while the answer
asserts the opposite. On a two-alternative question both candidates appear in
the retrieved context and in most answers, which makes containment nearly
uninformative for that question shape.

This is exactly the limitation the constructed audit is documented as unable to
find: substituted and perturbed golds are *easy* negatives, and this is a hard
one. The 0.000 false-accept rate over 3,050 constructed cases is still true and
still worth measuring. It just is not sufficient, which is why this repo labels
real disagreements as well — and reports what that found rather than only the
number that looks good.

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
  by reading real model output, not by the audit — so the repo does both. This
  is not hypothetical: hand-labelling the disagreements turned up a real false
  accept the constructed audit missed, on a "which of A or B" question where the
  gold string appears inside an answer that asserts the opposite.
- **The deterministic grader has a known false-reject class**: gold answers whose
  surface form differs from a correct answer's (singular/plural, a parenthetical
  alias, a gold that is a full sentence rather than a span, a gold that is a
  superset like `University of Melbourne in Australia`). Measured at 8 of 36
  labelled disagreements on one arm. It depresses absolute accuracy and, because
  the same rule is applied to every system, leaves comparisons intact.
- **LLM judges at 8B are not usable for this task.** Measured here, not assumed:
  `llama3.1:8b-instruct` accepted 26 of 27 wrong answers, including ten flat
  refusals to questions that had gold answers. `--judge-backend` exists to
  *characterise* a judge and to surface disagreements, not to produce a headline
  number.
- Deterministic grading is stricter than a human on genuine paraphrase, so
  absolute accuracies here are a **lower bound**. Every system is graded by the
  identical rule, so the comparisons between systems — which is what this project
  claims — are unaffected by that bias.
- `--judge-backend` adds an LLM judge and reports where the two graders disagree.
  Those disagreements are the only cases a human label would inform; agreement
  cases teach you nothing. If you want human validation, label those (a few per
  hundred) rather than the whole set.
