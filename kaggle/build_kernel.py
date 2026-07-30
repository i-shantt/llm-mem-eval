"""Generate a Kaggle notebook + metadata for a non-interactive GPU run.

    python kaggle/build_kernel.py            # writes kaggle/kernel/
    ./.venv/bin/kaggle kernels push -p kaggle/kernel
    ./.venv/bin/kaggle kernels status <user>/memllm-eval
    ./.venv/bin/kaggle kernels output <user>/memllm-eval -p results/kaggle

Kaggle runs the notebook top to bottom and keeps whatever lands in
/kaggle/working, so the run needs no interaction. GPU and Internet must both be
enabled -- Internet because Ollama and the dataset are both fetched at runtime.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = "https://github.com/i-shantt/memllm.git"
ANSWER_MODEL = "qwen2.5:7b-instruct"
N = 100

CELLS: list[str] = [
    # --- 1: Ollama ---
    f'''# Install and start Ollama. Internet must be enabled on this notebook.
import subprocess, time, urllib.request
subprocess.run("curl -fsSL https://ollama.com/install.sh | sh",
               shell=True, check=True, capture_output=True)
subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL,
                 stderr=subprocess.DEVNULL)

for _ in range(90):
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        print("ollama up"); break
    except Exception:
        time.sleep(1)
else:
    raise RuntimeError("ollama did not start -- is Internet enabled?")

subprocess.run(["ollama", "pull", "{ANSWER_MODEL}"], check=True)
print(subprocess.run(["ollama", "list"], capture_output=True, text=True).stdout)''',

    # --- 2: code + data ---
    f'''# Clone the repo and fetch the benchmark (~278MB).
import subprocess, os
subprocess.run(["git", "clone", "--depth", "1", "{REPO}",
                "/kaggle/working/memllm"], check=True)
os.chdir("/kaggle/working/memllm")
subprocess.run(["pip", "install", "-q", "rank_bm25", "sentence-transformers",
                "tiktoken"], check=True)

from huggingface_hub import hf_hub_download
hf_hub_download("xiaowu0162/longmemeval", "longmemeval_s",
                repo_type="dataset", local_dir="data/raw")
print(os.listdir("data/raw"))''',

    # --- 3: verify before spending GPU time ---
    '''# Fail fast: label integrity, cost accounting, and grader error rates.
import subprocess
for cmd in (["python", "tests/test_harness.py"],
            ["python", "scripts/audit_graders.py"]):
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-2000:])
    assert r.returncode == 0, f"{cmd} failed -- stopping before the real runs"

import json
audit = json.load(open("results/grader_audit.json"))["graders"]["deterministic"]
assert audit["false_accept_rate"] == 0.0, "grader accepts wrong answers"
print("grader false-accept", audit["false_accept_rate"],
      "| false-reject(hard)", round(audit["false_reject_rate_hard"], 4))''',

    # --- 4: measured token stats (needed by the cost curve) ---
    f'''# Real tiktoken counts; make_cost_curve.py refuses to run without these.
import subprocess
print(subprocess.run(["python", "scripts/measure_token_stats.py",
                      "--limit", "{N}"],
                     capture_output=True, text=True).stdout[-2000:])''',

    # --- 5: retrieval sweep, authoritative timings, three granularities ---
    f'''# Judge-free retrieval sweep. --no-cache forces real wall-clock timing,
# so the write-path costs in the cost curve are measured on this hardware.
import subprocess
for gran in ["turn", "user_turn", "session"]:
    print(f"===== granularity={{gran}} =====", flush=True)
    r = subprocess.run(
        ["python", "scripts/run_retrieval_eval.py", "--limit", "{N}",
         "--no-cache", "--retrievers", "random", "recency", "bm25", "dense",
         "hybrid", "oracle", "--granularity", gran, "--k", "20",
         "--tag", f"sweep_{{gran}}_n{N}"],
        capture_output=True, text=True)
    print(r.stdout[-4000:], r.stderr[-1500:], flush=True)''',

    # --- 6: end-to-end arms ---
    f'''# Retrieve -> answer -> deterministic grade. No judge model is loaded.
# The oracle arm separates retrieval failure from generation failure.
import subprocess
for retr in ["bm25", "hybrid", "oracle"]:
    print(f"===== e2e {{retr}} =====", flush=True)
    r = subprocess.run(
        ["python", "scripts/run_e2e_eval.py", "--limit", "{N}",
         "--retriever", retr, "--k", "10", "--num-ctx", "8192",
         "--answer-backend", "ollama:{ANSWER_MODEL}",
         "--tag", f"e2e_{{retr}}_k10_n{N}"],
        capture_output=True, text=True)
    print(r.stdout[-4000:], r.stderr[-1500:], flush=True)''',

    # --- 7: cross-check the deterministic grader against an LLM judge ---
    f'''# One arm re-graded by an LLM judge, to report how often the two differ.
# A different model from the answerer: self-grading is biased toward itself.
import subprocess
subprocess.run(["ollama", "pull", "llama3.1:8b-instruct"], check=False)
r = subprocess.run(
    ["python", "scripts/run_e2e_eval.py", "--limit", "{N}",
     "--retriever", "hybrid", "--k", "10", "--num-ctx", "8192",
     "--answer-backend", "ollama:{ANSWER_MODEL}",
     "--judge-backend", "ollama:llama3.1:8b-instruct",
     "--tag", f"e2e_hybrid_k10_n{N}_judged"],
    capture_output=True, text=True)
print(r.stdout[-4000:], r.stderr[-1500:])''',

    # --- 8: report + collect ---
    f'''# Regenerate the report and figure, then copy everything to /kaggle/working
# so `kaggle kernels output` retrieves it.
import subprocess, shutil, pathlib
open("RESULTS.md", "w").write(
    subprocess.run(["python", "scripts/make_report.py"],
                   capture_output=True, text=True).stdout)
print(subprocess.run(["python", "scripts/make_cost_curve.py", "--results",
                      f"results/sweep_turn_n{N}.json", "--k", "10"],
                     capture_output=True, text=True).stdout[-2000:])

out = pathlib.Path("/kaggle/working/out"); out.mkdir(exist_ok=True)
shutil.copytree("results", out / "results", dirs_exist_ok=True)
shutil.copy("RESULTS.md", out / "RESULTS.md")
print(sorted(p.name for p in (out / "results").iterdir()))
print(open("RESULTS.md").read()[:4000])''',
]


def build(out_dir: Path, username: str) -> None:
    nb = {
        "cells": [
            {"cell_type": "code", "metadata": {}, "source": src.splitlines(True),
             "execution_count": None, "outputs": []}
            for src in CELLS
        ],
        "metadata": {
            "kernelspec": {"language": "python", "display_name": "Python 3",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "memllm_eval.ipynb").write_text(json.dumps(nb, indent=1))
    (out_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{username}/memllm-eval",
        "title": "memllm eval",
        "code_file": "memllm_eval.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }, indent=1))
    print(f"wrote {out_dir}/memllm_eval.ipynb ({len(CELLS)} cells)")
    print(f"wrote {out_dir}/kernel-metadata.json for {username}/memllm-eval")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="kaggle/kernel")
    ap.add_argument("--username", default=None,
                   help="defaults to the username in ~/.kaggle/kaggle.json")
    args = ap.parse_args()

    username = args.username
    if not username:
        creds = Path.home() / ".kaggle" / "kaggle.json"
        if not creds.exists():
            raise SystemExit(
                f"{creds} not found. Get it from kaggle.com/settings -> API -> "
                "Create New Token, then: mkdir -p ~/.kaggle && mv "
                "~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 "
                "~/.kaggle/kaggle.json"
            )
        username = json.loads(creds.read_text())["username"]

    build(Path(args.out), username)
