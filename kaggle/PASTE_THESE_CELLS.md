# Kaggle cells to copy and paste

No Kaggle API token needed. Create a notebook at
[kaggle.com/code](https://www.kaggle.com/code) → **New Notebook**, then in the
right sidebar:

- **Accelerator:** `GPU T4 x2` (or `GPU P100`)
- **Internet:** `On` ← required; Ollama and the dataset are both fetched at runtime

Internet requires a phone-verified account (Settings → Phone Verification).

Paste the three cells below, then **Run All**. Expect **1.5–2.5 hours**. Cell 3
prints the full results at the end, so you can copy the text back even without
downloading anything.

---

## Cell 1 — Ollama and the model (~5 min)

```python
import subprocess, time, urllib.request

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

subprocess.run(["ollama", "pull", "qwen2.5:7b-instruct"], check=True)
print(subprocess.run(["ollama", "list"], capture_output=True, text=True).stdout)
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

from huggingface_hub import hf_hub_download
hf_hub_download("xiaowu0162/longmemeval", "longmemeval_s",
                repo_type="dataset", local_dir="data/raw")

# Fail before spending GPU time, not after.
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

## Cell 3 — the runs (~1.5–2 hours)

```python
import subprocess, shutil, pathlib
N = "100"

def run(cmd, label):
    print(f"\n===== {label} =====", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-3500:], flush=True)
    if r.returncode != 0:
        print("STDERR:", r.stderr[-2000:], flush=True)

# Measured token counts; the cost curve refuses to run without these.
run(["python", "scripts/measure_token_stats.py", "--limit", N], "token stats")

# Judge-free retrieval sweep at three memory granularities. --no-cache forces
# real wall-clock timings, so write-path cost is measured on this hardware.
for gran in ["turn", "user_turn", "session"]:
    run(["python", "scripts/run_retrieval_eval.py", "--limit", N, "--no-cache",
         "--retrievers", "random", "recency", "bm25", "dense", "hybrid", "oracle",
         "--granularity", gran, "--k", "20", "--tag", f"sweep_{gran}_n{N}"],
        f"retrieval sweep: {gran}")

# End to end. Grading is deterministic, so no judge model is loaded here.
# The oracle arm separates retrieval failure from generation failure.
for retr in ["bm25", "hybrid", "oracle"]:
    run(["python", "scripts/run_e2e_eval.py", "--limit", N, "--retriever", retr,
         "--k", "10", "--num-ctx", "8192",
         "--answer-backend", "ollama:qwen2.5:7b-instruct",
         "--tag", f"e2e_{retr}_k10_n{N}"], f"e2e: {retr}")

# One arm re-graded by a different model, to measure how often an LLM judge
# disagrees with the deterministic grader. Optional -- delete if short on time.
subprocess.run(["ollama", "pull", "llama3.1:8b-instruct"], check=False)
run(["python", "scripts/run_e2e_eval.py", "--limit", N, "--retriever", "hybrid",
     "--k", "10", "--num-ctx", "8192",
     "--answer-backend", "ollama:qwen2.5:7b-instruct",
     "--judge-backend", "ollama:llama3.1:8b-instruct",
     "--tag", f"e2e_hybrid_k10_n{N}_judged"], "e2e: hybrid + LLM judge")

# Report, figure, and one tarball to download.
open("RESULTS.md", "w").write(
    subprocess.run(["python", "scripts/make_report.py"],
                   capture_output=True, text=True).stdout)
run(["python", "scripts/make_cost_curve.py",
     "--results", f"results/sweep_turn_n{N}.json", "--k", "10"], "cost curve")

shutil.make_archive("/kaggle/working/memllm_results", "gztar",
                    ".", "results")
shutil.copy("RESULTS.md", "/kaggle/working/RESULTS.md")
print("\nfiles to download from the Output tab:")
for p in sorted(pathlib.Path("/kaggle/working").glob("*")):
    if p.is_file():
        print(" ", p.name, f"{p.stat().st_size/1024:.0f} KB")

print("\n" + "=" * 70 + "\nRESULTS.md\n" + "=" * 70)
print(open("RESULTS.md").read())
```

---

## Sending results back

Either works:

1. **Download** `memllm_results.tar.gz` from the notebook's **Output** tab (right
   sidebar, or the Data pane at the bottom) and drop it in `/Users/ishant/memllm/`.
2. **Copy the text** that Cell 3 prints after the `RESULTS.md` banner and paste it
   into the chat. Less complete than the tarball but enough to write up.

## If something fails

- **Cell 1 raises "ollama did not start"** — Internet is off in the sidebar.
- **Cell 2 assertion fails** — send the printed output; the harness caught a real
  problem and the run should not proceed.
- **`truncation_warning` in the output** — prompts overflowed the context window;
  raise `--num-ctx` above 8192 and rerun that arm.
- **Session dies at 12h** — `/kaggle/working` is lost on some restarts, so
  download the tarball before closing the tab.
