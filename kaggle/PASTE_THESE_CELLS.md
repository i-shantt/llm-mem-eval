# Kaggle cells to copy and paste

No Kaggle API token needed. Create a notebook at
[kaggle.com/code](https://www.kaggle.com/code) → **New Notebook**, then in the
right sidebar:

- **Accelerator:** `GPU T4 x2` — **required** for the 32B arm. A single P100
  (16 GB) cannot hold a 32B model; it will fall back to CPU and take all day.
- **Internet:** `On` ← required; Ollama and the dataset are both fetched at runtime

Internet requires a phone-verified account (Settings → Phone Verification).

Paste the cells below and **Run All**. Budget **4–6 hours**. Every arm prints
its own results the moment it finishes and every arm is skipped if its results
file already exists, so a session that dies partway can be resumed by re-running
Cell 4 — and a partial run is still usable.

---

## What this run is for

The previous run left three questions open. Each cell below answers one.

1. **Is the low accuracy the model or the system?** The oracle arm hands the
   model the gold evidence, so `1 − oracle_accuracy` is model-and-grader loss
   with retrieval removed. At 7B, oracle was **0.549** — 45 points of loss with
   nothing left to retrieve. Re-running oracle at 32B is the single most
   decisive number here.
2. **Was the last run silently truncated?** `num_ctx` was never set, so Ollama
   used its 2048 default while prompts averaged 2,232–2,893 tokens. Accuracy
   was inversely ordered with prompt length across arms, which is exactly what
   truncation produces. The 7B control arms below rerun the identical config
   with an 8192 window and settle it.
3. **Does session granularity pay off end to end?** Retrieval `recall@10` goes
   **0.811 → 0.981** from turn to session granularity. That was measured on the
   retrieval path only; the `hybrid@session` arm tests whether it converts into
   answers.

---

## Cell 1 — Ollama and the models (~20 min, mostly the 32B download)

```python
import subprocess, time, urllib.request, textwrap

subprocess.run("curl -fsSL https://ollama.com/install.sh | sh",
               shell=True, check=True)
subprocess.Popen(["ollama", "serve"],
                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

for i in range(90):
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        print(f"ollama up after {i}s"); break
    except Exception:
        time.sleep(1)
else:
    raise RuntimeError("ollama did not start -- is Internet enabled in the sidebar?")

# ~20GB. If this OOMs later, swap to qwen2.5:14b-instruct and change BIG below.
for m in ["qwen2.5:7b-instruct", "qwen2.5:32b-instruct"]:
    print(f"pulling {m} ...", flush=True)
    subprocess.run(["ollama", "pull", m], check=True)

print(subprocess.run(["ollama", "list"], capture_output=True, text=True).stdout)
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                      "--format=csv"], capture_output=True, text=True).stdout)
```

## Cell 2 — code, data, and pre-flight checks (~3 min)

```python
import subprocess, os, json

subprocess.run(["git", "clone", "--depth", "1",
                "https://github.com/i-shantt/memllm.git",
                "/kaggle/working/memllm"], check=True)
os.chdir("/kaggle/working/memllm")
subprocess.run(["pip", "install", "-q", "rank_bm25", "sentence-transformers",
                "tiktoken"], check=True)

# The last run silently used a 2048-token context because this flag was not in
# the pushed code. Fail in three minutes rather than after three hours.
help_text = subprocess.run(["python", "scripts/run_e2e_eval.py", "--help"],
                           capture_output=True, text=True).stdout
for flag in ["--num-ctx", "--gen-timeout", "--max-new-tokens"]:
    assert flag in help_text, (
        f"{flag} missing -- the clone is stale. Run `git push origin main` "
        f"locally, then re-run this cell.")
print("pre-flight: all required flags present")

from huggingface_hub import hf_hub_download
hf_hub_download("xiaowu0162/longmemeval", "longmemeval_s",
                repo_type="dataset", local_dir="data/raw")

for cmd in (["python", "tests/test_harness.py"],
            ["python", "scripts/audit_graders.py"]):
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-2500:], r.stderr[-1500:])
    assert r.returncode == 0, f"{cmd} failed -- fix before the real runs"

audit = json.load(open("results/grader_audit.json"))["graders"]["deterministic"]
assert audit["false_accept_rate"] == 0.0, "grader accepts known-wrong answers"
print("grader false-accept:", audit["false_accept_rate"],
      "| false-reject (rewrites):", round(audit["false_reject_rate_hard"], 4))
```

## Cell 3 — retrieval sweep, CPU only (~15 min, no GPU contention)

```python
import subprocess
N = "100"

def run(cmd, label):
    print(f"\n===== {label} =====", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-4000:], flush=True)
    if r.returncode != 0:
        print("STDERR:", r.stderr[-2000:], flush=True)
    return r.returncode == 0

run(["python", "scripts/measure_token_stats.py", "--limit", N], "token stats")

# Now emits per-question recall@10, which is what separates a retrieval failure
# from a generation failure on any individual question.
for gran in ["turn", "user_turn", "session"]:
    run(["python", "scripts/run_retrieval_eval.py", "--limit", N, "--no-cache",
         "--retrievers", "random", "recency", "bm25", "dense", "hybrid", "oracle",
         "--granularity", gran, "--k", "20", "--tag", f"sweep_{gran}_n{N}"],
        f"retrieval sweep: {gran}")
```

## Cell 4 — the end-to-end arms (~4–5 hours, resumable)

```python
import subprocess, pathlib, json
N = "100"
SMALL = "ollama:qwen2.5:7b-instruct"
BIG   = "ollama:qwen2.5:32b-instruct"    # -> qwen2.5:14b-instruct if OOM

def arm(model, retriever, gran, tag, ctx="8192"):
    """One end-to-end arm. Skipped if it already has a results file."""
    out = pathlib.Path(f"results/{tag}.json")
    if out.exists():
        print(f"[skip] {tag} already done"); return
    print(f"\n===== {tag} =====", flush=True)
    r = subprocess.run(
        ["python", "scripts/run_e2e_eval.py", "--limit", N,
         "--retriever", retriever, "--granularity", gran, "--k", "10",
         "--num-ctx", ctx,
         # 64 truncated at least five answers mid-sentence last time, and a 32B
         # is more verbose than a 7B. A cut-off answer grades as wrong.
         "--max-new-tokens", "256",
         # 300s is not enough for a 32B on two T4s.
         "--gen-timeout", "900",
         "--answer-backend", model,
         # No --judge-backend. The 8B judge accepted 26 of 27 wrong answers,
         # including ten flat refusals; it cannot produce a usable number.
         "--tag", tag],
        capture_output=True, text=True)
    print(r.stdout[-4000:], flush=True)
    if r.returncode != 0:
        print("STDERR:", r.stderr[-3000:], flush=True)

# --- 7B controls, at the settings the last run should have used (~45 min) ---
# Same model, same arms, correct context window. Any gain over the previous
# numbers (bm25 0.385 / hybrid 0.429 / oracle 0.549) is truncation, not skill,
# and the earlier retrieval attribution has to be revised down by that much.
arm(SMALL, "bm25",   "turn", f"e2e_7b_bm25_turn_k10_n{N}")
arm(SMALL, "hybrid", "turn", f"e2e_7b_hybrid_turn_k10_n{N}")
arm(SMALL, "oracle", "turn", f"e2e_7b_oracle_turn_k10_n{N}")

# --- 32B, ordered most-decisive-first so a timeout loses the least (~3-4 h) ---
arm(BIG, "oracle", "turn",    f"e2e_32b_oracle_turn_k10_n{N}")
arm(BIG, "hybrid", "turn",    f"e2e_32b_hybrid_turn_k10_n{N}")
arm(BIG, "hybrid", "session", f"e2e_32b_hybrid_session_k10_n{N}")

# --- optional, only if there is time left (~1 h) ---
arm(SMALL, "hybrid", "session", f"e2e_7b_hybrid_session_k10_n{N}")
```

## Cell 5 — summary, truncation audit, and download (~2 min)

```python
import json, pathlib, subprocess, shutil

rows = []
for p in sorted(pathlib.Path("results").glob("e2e_*.json")):
    d = json.load(open(p))
    c = d["config"]
    rows.append({
        "arm": p.stem,
        "model": c.get("answer_backend_name"),
        "retriever": c["retriever"], "granularity": c.get("granularity"),
        "accuracy": round(d["accuracy"], 3),
        "token_f1": round(d["token_f1_mean"], 3),
        "n_graded": d["n_graded"], "n_not_gradable": d["n_not_gradable"],
        "read_tok/q": round(d["read_tokens_per_query"]),
        "max_prompt": d.get("prompt_tokens_max"),
        "num_ctx": c.get("num_ctx"),
        "truncated": d.get("n_prompts_truncated", 0),
        "hit_tok_cap": d.get("n_hit_token_cap"),
        "by_type": {t: round(v["accuracy"], 2)
                    for t, v in d["accuracy_by_question_type"].items()},
    })

print(f"{'arm':<38}{'acc':>7}{'f1':>7}{'read':>7}{'maxP':>7}{'ctx':>7}{'trunc':>7}{'cap':>6}")
for r in rows:
    print(f"{r['arm']:<38}{r['accuracy']:>7}{r['token_f1']:>7}"
          f"{r['read_tok/q']:>7}{r['max_prompt']:>7}{r['num_ctx']:>7}"
          f"{r['truncated']:>7}{r['hit_tok_cap']:>6}")

print("\nby question type:")
for r in rows:
    print(f"  {r['arm']}\n    {r['by_type']}")

# Every wrong answer, tagged with whether retrieval had actually delivered the
# evidence. recall@10 == 1.0 on a wrong answer means the model failed, not the
# retriever -- this is the table the whole attribution rests on.
def failure_table(e2e_tag, sweep_tag="sweep_turn_n100", retr="hybrid"):
    e2e = json.load(open(f"results/{e2e_tag}.json"))
    sweep = json.load(open(f"results/{sweep_tag}.json"))[retr]
    rec = {q["question_id"]: q for q in sweep["per_question"]}
    print(f"\n===== wrong answers: {e2e_tag} =====")
    for r in e2e["records"]:
        if r["deterministic"] is not False:
            continue
        q = rec.get(r["question_id"], {})
        print(f"  recall@10={q.get('recall@10')} n_ev={q.get('n_evidence')} "
              f"[{r['question_type']}] cap={r['hit_token_cap']}")
        print(f"    Q:    {r['question'][:110]}")
        print(f"    GOLD: {r['gold'][:110]}")
        print(f"    PRED: {r['pred'][:220]}")

for tag in [f"e2e_32b_hybrid_turn_k10_n100", f"e2e_7b_hybrid_turn_k10_n100"]:
    if pathlib.Path(f"results/{tag}.json").exists():
        failure_table(tag)

open("RESULTS.md", "w").write(
    subprocess.run(["python", "scripts/make_report.py"],
                   capture_output=True, text=True).stdout)
subprocess.run(["python", "scripts/make_cost_curve.py",
                "--results", "results/sweep_turn_n100.json", "--k", "10"])

shutil.make_archive("/kaggle/working/memllm_results", "gztar", ".", "results")
shutil.copy("RESULTS.md", "/kaggle/working/RESULTS.md")
json.dump(rows, open("/kaggle/working/arm_summary.json", "w"), indent=2)
for p in sorted(pathlib.Path("/kaggle/working").glob("*")):
    if p.is_file():
        print(" ", p.name, f"{p.stat().st_size/1024:.0f} KB")

print("\n" + "=" * 70 + "\nRESULTS.md\n" + "=" * 70)
print(open("RESULTS.md").read())
```

---

## What each number will mean

Predictions written down **before** the run, so the writeup reports a checked
prediction rather than a rationalised outcome.

| measurement | 7B result | prediction at 32B | what it decides |
|---|---|---|---|
| oracle @ turn | 0.549 | **0.70–0.78** | how much of the loss was the model |
| hybrid @ turn | 0.429 | **0.58–0.65** | end-to-end headline |
| oracle − hybrid | 0.120 | **stays ≈ 0.12** | retrieval cost is model-independent |
| temporal-reasoning @ oracle | 0.400 | **0.65+** | the arithmetic failures |
| hybrid @ session − hybrid @ turn | — | **+0.05–0.10** | does recall 0.811→0.981 convert |

The one that matters most is the third. If the oracle−hybrid gap holds near 12
points across a 4.5× change in model size, the claim becomes "retrieval costs a
fixed ~12 accuracy points independent of the answering model" — which is a far
stronger result than any single accuracy figure.

And on the 7B controls: if `e2e_7b_*` comes back materially above 0.385 / 0.429
/ 0.549, the previous run was truncating, and the retrieval share of the loss
was overstated.

## Sending results back

Either works:

1. **Download** `memllm_results.tar.gz` from the notebook's **Output** tab and
   drop it in `/Users/ishant/memllm/`. Most complete.
2. **Copy the text** Cell 5 prints — the arm table, the by-type block, and the
   wrong-answer tables. Enough to do the attribution.

## If something fails

- **Cell 1 raises "ollama did not start"** — Internet is off in the sidebar.
- **Cell 2 assertion on a missing flag** — the GitHub clone is behind local.
  Run `git push origin main` on the laptop and re-run Cell 2.
- **32B is unbearably slow, or OOM** — set `BIG = "ollama:qwen2.5:14b-instruct"`
  and re-run Cell 4. 14B still answers the model-vs-system question; it just
  makes a weaker version of the point.
- **`truncated` non-zero in Cell 5** — raise `--num-ctx` to 16384 for that arm
  and rerun. Any arm with truncation is not comparable to the others.
- **`hit_tok_cap` large** — answers are being cut off; raise `--max-new-tokens`.
- **Session dies** — Cell 4 skips completed arms, so just re-run Cells 1, 2, 4.
  Download the tarball before closing the tab; `/kaggle/working` is not durable.
