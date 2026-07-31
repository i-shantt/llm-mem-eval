# Kaggle cells to copy and paste

No Kaggle API token needed. Create a notebook at
[kaggle.com/code](https://www.kaggle.com/code) → **New Notebook**, then in the
right sidebar:

- **Accelerator:** `GPU T4 x2` (or `P100`). Nothing here needs more than one
  card — the largest model is 14B, ~9 GB of weights against 15 GB of VRAM.
- **Internet:** `On` ← required; Ollama and the dataset are both fetched at runtime

Internet requires a phone-verified account (Settings → Phone Verification).

Paste the cells below and **Run All**. Budget **3–4 hours**. Every arm prints
its results the moment it finishes and is skipped if its results file already
exists, so a session that dies partway resumes by re-running Cell 4 — and a
partial run is still usable.

---

## What this run is for

The scaling-ladder run settled the truncation question and broke two
assumptions. This run follows up on what it found.

1. **Granularity, compared honestly.** The last run compared `k=10` across
   granularities. Ten sessions is ~22K tokens against an 8192 window, so
   llama.cpp discarded half the context and every prompt came back clamped to
   exactly 4098 tokens; both session arms scored 0.03–0.04 and measured the
   clamp. Units differ ~40× in size, so `k` is now set **per granularity** to
   hold read cost at ~2,600 tokens: 12 turns, 48 user turns, or 1 session. That
   is the only comparison that means anything for a project about cost.
2. **Oracle is not a ceiling, so stop treating it as one.** It retrieves only
   `has_answer`-labelled turns, which is a *smaller* prompt than a retriever
   builds on 59 of 100 questions. Hybrid beat oracle at 1.5B and 14B. The arm
   still measures something useful, but `1 − oracle` is not model loss alone.
3. **Answer length is confounding the model comparison.** Capping stored
   answers at their first N words, 14B's 14-point lead over 7B decays to zero
   (+0.142 full → +0.044 at 15 words → 0.000 at 8). Token-F1 puts the same gap
   at +0.011. `make_report.py` now prints that decay for every arm, so it is
   visible rather than assumed.

### Why not 32B

32B at q4_K_M is ~20 GB of weights plus ~2 GB of KV cache at 8K context. It
fits in 32 GB of T4 x2 only if Ollama actually spreads across both cards, and
Kaggle's root filesystem often cannot hold the download in the first place.
Cell 1 diagnoses both. If 32B does load cleanly, add it as a fifth rung — but
the ladder does not depend on it.

| model (q4_K_M) | weights | + KV @ 8K | one T4 (15 GB)? |
|---|---|---|---|
| qwen2.5:1.5b-instruct | ~1.0 GB | ~1.6 GB | yes |
| qwen2.5:3b-instruct | ~1.9 GB | ~2.8 GB | yes |
| qwen2.5:7b-instruct | ~4.7 GB | ~6.3 GB | yes |
| qwen2.5:14b-instruct | ~9.0 GB | ~10.6 GB | yes |
| qwen2.5:32b-instruct | ~19.9 GB | ~22 GB | no — needs both cards |

---

## Cell 1 — Ollama, disk/GPU diagnostics, and the model ladder (~15 min)

```python
import subprocess, time, urllib.request, os, json

# Ollama defaults to /root/.ollama, which is small on Kaggle. Putting the model
# store on the working volume is what makes multi-GB pulls survive.
os.environ["OLLAMA_MODELS"] = "/kaggle/working/.ollama/models"
os.environ["OLLAMA_SCHED_SPREAD"] = "1"   # use every visible GPU, not just one
os.makedirs(os.environ["OLLAMA_MODELS"], exist_ok=True)

print(subprocess.run(["df", "-h", "/root", "/kaggle/working"],
                     capture_output=True, text=True).stdout)
print(subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total",
                      "--format=csv"], capture_output=True, text=True).stdout)

subprocess.run("curl -fsSL https://ollama.com/install.sh | sh",
               shell=True, check=True)
subprocess.Popen(["ollama", "serve"], env=os.environ,
                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

for i in range(90):
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        print(f"ollama up after {i}s"); break
    except Exception:
        time.sleep(1)
else:
    raise RuntimeError("ollama did not start -- is Internet enabled in the sidebar?")

# The whole ladder is ~17 GB on disk; each rung loads into one card on its own.
LADDER = ["qwen2.5:1.5b-instruct", "qwen2.5:3b-instruct",
          "qwen2.5:7b-instruct", "qwen2.5:14b-instruct"]
for m in LADDER:
    print(f"pulling {m} ...", flush=True)
    subprocess.run(["ollama", "pull", m], check=True)

# Load the largest rung and confirm it is fully on GPU. "100% GPU" is the only
# acceptable answer -- any CPU share means it did not fit and will crawl.
subprocess.run(["ollama", "run", "qwen2.5:14b-instruct", "hi"],
               capture_output=True, text=True, timeout=600)
print(subprocess.run(["ollama", "ps"], capture_output=True, text=True).stdout)
print(subprocess.run(["df", "-h", "/kaggle/working"],
                     capture_output=True, text=True).stdout)
```

**Read `ollama ps` before going further.** If the `PROCESSOR` column says
anything other than `100% GPU` for 14B, drop the top rung to `qwen2.5:7b` and
report what it said — that, not the arm results, is the thing to fix first.

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

run(["python", "scripts/measure_token_stats.py", "--limit", N], "token stats")

# Now emits per-question recall@10, which is what separates a retrieval failure
# from a generation failure on any individual question.
# k is per granularity, not shared. Units differ ~40x in size (turn ~213 tok,
# user_turn ~54, session ~2187), so a shared k compares a 2.6K-token prompt
# against a 22K-token one. These three k values are ~2600 read tokens each.
BUDGET_K = {"turn": 12, "user_turn": 48, "session": 1}
for gran, k in BUDGET_K.items():
    run(["python", "scripts/run_retrieval_eval.py", "--limit", N, "--no-cache",
         "--retrievers", "random", "recency", "bm25", "dense", "hybrid", "oracle",
         "--granularity", gran, "--k", str(max(k, 20)),
         "--ks", "1", "3", "5", "10", "20", str(k),
         "--tag", f"sweep_{gran}_n{N}"], f"retrieval sweep: {gran} (budget k={k})")
```

## Cell 4 — the scaling ladder (~2.5–3 hours, resumable)

```python
import subprocess, pathlib
N = "100"
LADDER = ["1.5b", "3b", "7b", "14b"]   # drop "14b" if Cell 1 showed CPU offload

def arm(size, retriever, gran, k="10", ctx="8192"):
    """One end-to-end arm. Skipped if it already has a results file."""
    tag = f"e2e_{size}_{retriever}_{gran}_k{k}_n{N}"
    out = pathlib.Path(f"results/{tag}.json")
    if out.exists():
        print(f"[skip] {tag} already done"); return
    print(f"\n===== {tag} =====", flush=True)
    r = subprocess.run(
        ["python", "scripts/run_e2e_eval.py", "--limit", N,
         "--retriever", retriever, "--granularity", gran, "--k", k,
         "--num-ctx", ctx,
         # 64 truncated at least five answers mid-sentence last time, and a
         # bigger model is more verbose. A cut-off answer grades as wrong.
         "--max-new-tokens", "256",
         "--gen-timeout", "900",
         "--answer-backend", f"ollama:qwen2.5:{size}-instruct",
         # No --judge-backend. The 8B judge accepted 26 of 27 wrong answers,
         # including ten flat refusals; it cannot produce a usable number.
         "--tag", tag],
        capture_output=True, text=True)
    print(r.stdout[-4000:], flush=True)
    if r.returncode != 0:
        print("STDERR:", r.stderr[-3000:], flush=True)

# --- the ladder: oracle first, so a timeout still leaves a complete curve ---
# Oracle removes retrieval entirely, so this curve is pure model capacity.
for size in LADDER:
    arm(size, "oracle", "turn")

# Same ladder with real retrieval. The gap between the two curves at each rung
# is the cost of retrieval at that model size.
for size in LADDER:
    arm(size, "hybrid", "turn")

# --- 7B controls for the truncation question (~25 min) ---
# Identical config to the previous run except the context window. Any gain over
# bm25 0.385 / hybrid 0.429 / oracle 0.549 is truncation, not skill.
arm("7b", "bm25", "turn")

# --- granularity at a MATCHED read-token budget ---
# The previous run compared k=10 across granularities. Ten sessions is ~22K
# tokens against an 8192 window, so llama.cpp discarded half the context and
# every prompt came back clamped to 4098 tokens -- the arms scored 0.03-0.04
# and measured the clamp, not the system. k is now set per granularity so all
# three arms read ~2600 tokens, which is the only comparison that means
# anything for a project about cost.
for size in LADDER[-2:]:
    arm(size, "hybrid", "turn", k="12")
    arm(size, "hybrid", "user_turn", k="48")
    arm(size, "hybrid", "session", k="1")
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

print(f"{'arm':<42}{'acc':>7}{'f1':>7}{'read':>7}{'maxP':>7}{'ctx':>7}{'trunc':>7}{'cap':>6}")
for r in rows:
    print(f"{r['arm']:<42}{r['accuracy']:>7}{r['token_f1']:>7}"
          f"{r['read_tok/q']:>7}{r['max_prompt']:>7}{r['num_ctx']:>7}"
          f"{r['truncated']:>7}{r['hit_tok_cap']:>6}")

# The scaling curve, and the retrieval gap at each rung.
print("\nsize      oracle   hybrid   gap")
acc = {(r["model"], r["retriever"], r["granularity"]): r["accuracy"] for r in rows}
for size in ["1.5b", "3b", "7b", "14b", "32b"]:
    m = f"ollama:qwen2.5:{size}-instruct"
    o, h = acc.get((m, "oracle", "turn")), acc.get((m, "hybrid", "turn"))
    if o is None and h is None:
        continue
    gap = round(o - h, 3) if (o is not None and h is not None) else None
    print(f"{size:<10}{str(o):>7}{str(h):>9}{str(gap):>7}")

print("\nby question type:")
for r in rows:
    print(f"  {r['arm']}\n    {r['by_type']}")

# Every wrong answer, tagged with whether retrieval had actually delivered the
# evidence. recall@10 == 1.0 on a wrong answer means the model failed, not the
# retriever -- this is the table the whole attribution rests on.
def failure_table(tag, sweep_tag="sweep_turn_n100", retr="hybrid"):
    e2e = json.load(open(f"results/{tag}.json"))
    sweep = json.load(open(f"results/{sweep_tag}.json"))[retr]
    rec = {q["question_id"]: q for q in sweep["per_question"]}
    print(f"\n===== wrong answers: {tag} =====")
    for r in e2e["records"]:
        if r["deterministic"] is not False:
            continue
        q = rec.get(r["question_id"], {})
        print(f"  recall@10={q.get('recall@10')} n_ev={q.get('n_evidence')} "
              f"[{r['question_type']}] cap={r['hit_token_cap']}")
        print(f"    Q:    {r['question'][:110]}")
        print(f"    GOLD: {r['gold'][:110]}")
        print(f"    PRED: {r['pred'][:220]}")

for size in ["14b", "7b"]:
    tag = f"e2e_{size}_hybrid_turn_k10_n100"
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
prediction rather than a rationalised outcome. Only 7B is measured today.

| rung | oracle @ turn | hybrid @ turn | gap |
|---|---|---|---|
| 1.5B | 0.341 | 0.352 | −0.011 |
| 3B | 0.407 | 0.308 | +0.099 |
| 7B | 0.571 | 0.440 | +0.131 |
| 14B | 0.549 | 0.582 | −0.033 |

Those are measured, not predicted. What is still open is granularity at matched
budget, and the prediction there is that **session at k=1 beats turn at k=12**
on the same token budget — one whole session carries its own context, where
twelve scattered turns do not. If that holds, the finding is that *what* you
retrieve matters more than *how well* you rank it, at equal cost.

Three things this can show, in order of how much they are worth:

1. **The gap column stays flat.** If oracle − hybrid holds near 0.12 across ~10×
   of parameters, the claim becomes *"retrieval costs a fixed ~12 accuracy
   points, independent of the answering model."* That is a far stronger result
   than any single accuracy figure, and it is the one worth putting in the
   README.
2. **The oracle curve climbs steeply.** That is the direct evidence that the
   remaining loss is model capacity rather than the memory system — the point
   under dispute. If it flattens instead, the ceiling is the benchmark or the
   grader, not the model, and that is a more interesting finding still.
3. **`temporal-reasoning` at oracle** (0.400 at 7B). Every recall-1.00 failure
   was arithmetic over correctly-retrieved facts — date subtraction, summation,
   counting. If scale fixes anything, it fixes this slice first.

On the truncation question: if `e2e_7b_*` comes back materially above 0.385 /
0.429 / 0.549, the previous run was truncating and the retrieval share of the
loss was overstated.

## Sending results back

Either works:

1. **Download** `memllm_results.tar.gz` from the notebook's **Output** tab and
   drop it in `/Users/ishant/memllm/`. Most complete.
2. **Copy the text** Cell 5 prints — the arm table, the scaling curve, the
   by-type block, and the wrong-answer tables. Enough to do the attribution.

## If something fails

- **Cell 1 raises "ollama did not start"** — Internet is off in the sidebar.
- **`ollama ps` shows a CPU share for 14B** — it did not fit. Drop `"14b"` from
  `LADDER`; the 1.5B/3B/7B curve still answers the scaling question.
- **`no space left on device` during a pull** — `OLLAMA_MODELS` did not take.
  Confirm Cell 1 set it *before* `ollama serve` started; restarting the kernel
  and re-running Cell 1 is the fix.
- **Cell 2 assertion on a missing flag** — the GitHub clone is behind local.
  Run `git push origin main` on the laptop and re-run Cell 2.
- **`truncated` non-zero in Cell 5** — raise `--num-ctx` to 16384 for that arm
  and rerun. Any arm with truncation is not comparable to the others.
- **`hit_tok_cap` large** — answers are being cut off; raise `--max-new-tokens`.
- **Session dies** — Cell 4 skips completed arms, so re-run Cells 1, 2, 4.
  Download the tarball before closing the tab; `/kaggle/working` is not durable.
