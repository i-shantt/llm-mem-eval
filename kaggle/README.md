# Running the end-to-end arm on Kaggle

The retrieval results run fine on a laptop. The end-to-end arm does not: a 1.5B
model on Apple MPS takes ~26s per question, so 100 questions with a 7B model is
a Kaggle job.

**Notebook settings:** Accelerator `GPU T4 x2` (or P100), and **Internet must be
ON** — Ollama and the dataset both need it. Kaggle gives ~30h/week and 12h per
session, which is plenty; a 100-question run with a 7B takes roughly 30-50 min.

## Cell 1 — install and start Ollama

```python
!curl -fsSL https://ollama.com/install.sh | sh > /dev/null 2>&1
import subprocess, time, urllib.request
subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL,
                 stderr=subprocess.DEVNULL)

for _ in range(60):                      # wait for the server, don't assume
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        print("ollama up"); break
    except Exception:
        time.sleep(1)
else:
    raise RuntimeError("ollama did not start -- is Internet enabled?")
```

## Cell 2 — pull the model

```python
!ollama pull qwen2.5:7b-instruct     # ~4.7GB, fits a single T4
!ollama list
```

Use `qwen2.5:7b-instruct` as the answering model. For the judge, a **different**
model is better practice — a model grading its own output is biased toward it.
`llama3.1:8b-instruct` is a reasonable second pull if time allows.

## Cell 3 — get the code and data

```python
!git clone https://github.com/<your-username>/memllm.git /kaggle/working/memllm
%cd /kaggle/working/memllm
!pip install -q rank_bm25 sentence-transformers tiktoken

from huggingface_hub import hf_hub_download
hf_hub_download("xiaowu0162/longmemeval", "longmemeval_s",
                repo_type="dataset", local_dir="data/raw")
```

## Cell 4 — verify before spending GPU hours

```python
!python tests/test_harness.py
!python scripts/measure_token_stats.py --limit 40 > /dev/null
!python scripts/run_e2e_eval.py --limit 3 --retriever bm25 --k 5 \
    --answer-backend ollama:qwen2.5:7b-instruct --tag kaggle_smoke
```

## Cell 5 — the real runs

```python
# judge-free retrieval sweep (fast, no LLM)
!python scripts/run_retrieval_eval.py --limit 100 --no-cache \
    --retrievers random recency bm25 dense hybrid oracle \
    --granularity turn --k 20 --tag sweep_turn_n100

# end-to-end, one arm per retriever so retrieval quality can be separated
# from answer quality
for r in ["bm25", "hybrid", "oracle"]:
    !python scripts/run_e2e_eval.py --limit 100 --retriever {r} --k 10 \
        --answer-backend ollama:qwen2.5:7b-instruct \
        --judge-backend ollama:llama3.1:8b-instruct \
        --tag e2e_{r}_k10_n100
```

The `oracle` arm matters: it separates "we retrieved the wrong turn" from "we
retrieved the right turn and the model still answered wrong." Without it, a low
end-to-end score is uninterpretable.

## Cell 6 — download results, then hand-label

```python
!python scripts/make_report.py > RESULTS.md
!python scripts/make_cost_curve.py --results results/sweep_turn_n100.json --k 10
!tar czf /kaggle/working/results.tar.gz results RESULTS.md
```

Download `results.tar.gz`. Then hand-label a worksheet locally and run
`scripts/validate_judge.py`. **Until that agreement number exists, the
end-to-end accuracies are not reportable** — that is the whole point of the
critique this project makes, so it applies to our own numbers first.

## Notes

- `--no-cache` on the retrieval sweep forces authoritative wall-clock timing.
  Kaggle's GPU differs from a laptop, so report the hardware with the timings.
- Kaggle sessions die at 12h and lose `/kaggle/working` on some restarts. Tar
  and download results at the end of every session.
- Ollama's `prompt_eval_count` / `eval_count` are what feed the read-path
  ledger, so token accounting stays real rather than estimated.
