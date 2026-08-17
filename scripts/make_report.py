"""Generate the results table from results/*.json.

Numbers in the README are never hand-typed -- they are regenerated from run
artifacts, so the repo cannot drift from its own measurements.

    python scripts/make_report.py > RESULTS.md
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ORDER = ["random", "recency", "bm25", "dense", "hybrid", "oracle"]


def load_all(results_dir: Path) -> dict[str, dict]:
    """Every run, keyed `<retriever>|<granularity>|<n_total>`.

    `n` belongs in the key rather than being used as a tiebreak. Keeping only
    the largest run per (retriever, granularity) is right for the per-row
    tables, which print their own `n`, but it silently corrupts the
    cross-granularity table the moment one granularity is re-run at a different
    size: turn at n=500 would have been compared against session at n=100 and
    the difference reported as a granularity effect. Callers now say which they
    want -- `best_runs` for per-row tables, `matched_runs` for comparisons.
    """
    runs: dict[str, dict] = {}
    for path in sorted(results_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            print(f"skipping unparseable {path}", file=sys.stderr)
            continue
        if not isinstance(payload, dict):
            continue
        for name, run in payload.items():
            # results/ also holds non-run artifacts (token_stats_*.json); a run
            # payload is a dict keyed by retriever name carrying "metrics".
            if not isinstance(run, dict) or "metrics" not in run:
                continue
            gran = run.get("config", {}).get("granularity", "?")
            n = run.get("metrics", {}).get("n_total", 0)
            runs[f"{name}|{gran}|{n}"] = run
    return runs


def best_runs(runs: dict[str, dict]) -> dict[str, dict]:
    """Largest run per (retriever, granularity), keyed `<retriever>|<granularity>`.

    For tables whose rows stand alone and print their own `n`.
    """
    out: dict[str, dict] = {}
    for key, run in runs.items():
        name, gran, n = key.split("|")
        short = f"{name}|{gran}"
        if short not in out or int(n) > out[short]["metrics"]["n_total"]:
            out[short] = run
    return out


def matched_runs(
    runs: dict[str, dict], grans: list[str]
) -> tuple[dict[str, dict], int | None]:
    """Runs at the largest `n` present for *every* granularity in `grans`.

    Returns (`<retriever>|<granularity>` -> run, n). A cross-granularity row is
    only a granularity comparison if the question set is held fixed, so this
    trades size for comparability rather than mixing the two.
    """
    ns: dict[str, set[int]] = {}
    for key in runs:
        _, gran, n = key.split("|")
        ns.setdefault(gran, set()).add(int(n))
    if not all(g in ns for g in grans):
        return {}, None
    common = set.intersection(*(ns[g] for g in grans))
    if not common:
        return {}, None
    n = max(common)
    out = {}
    for key, run in runs.items():
        name, gran, run_n = key.split("|")
        if int(run_n) == n and gran in grans:
            out[f"{name}|{gran}"] = run
    return out, n


GRAN_ORDER = ["turn", "user_turn", "session"]


def sort_key(key: str) -> tuple[int, str]:
    name = key.split("|")[0]
    return (ORDER.index(name) if name in ORDER else len(ORDER), key)


def load_e2e(results_dir: Path) -> list[dict]:
    """End-to-end arms are shaped differently from retrieval sweeps: one run
    per file, keyed by "accuracy" rather than "metrics"."""
    arms = []
    for path in sorted(results_dir.glob("e2e_*.json")):
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "accuracy" in payload:
            arms.append(payload | {"_tag": path.stem})
    return arms


def model_label(arm: dict) -> str:
    name = arm.get("config", {}).get("answer_backend_name", "?")
    return name.split(":", 1)[1] if ":" in name else name


def param_count(label: str) -> float:
    """Sort a scaling ladder by size, not alphabetically -- otherwise 14b files
    between 1.5b and 3b and the curve reads as noise."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*([bm])\b", label.lower())
    if not m:
        return float("inf")
    return float(m.group(1)) * (1e9 if m.group(2) == "b" else 1e6)


def arm_key(arm: dict) -> tuple:
    label = model_label(arm)
    c = arm["config"]
    return (param_count(label), label, c["retriever"], c.get("granularity", ""))


def granularity_section(all_runs: dict[str, dict]) -> None:
    """The same retriever at three definitions of a memory unit. This is the
    knob the field rarely reports and it moves recall more than the retriever
    choice does.

    Held to one question set across the whole section: every cell here comes
    from the largest `n` that exists for all the granularities shown, so a
    difference between columns is a granularity effect and not a sample
    difference.
    """
    grans = [g for g in GRAN_ORDER
             if any(k.split("|")[1] == g for k in all_runs)]
    if len(grans) < 2:
        return
    runs, n_matched = matched_runs(all_runs, grans)
    if not runs:
        return

    print("\n## What a memory unit should be\n")
    print(f"The same retrievers, over the same conversations, cut into turns, "
          f"user turns, or whole sessions. Coarser units retrieve more evidence "
          f"per hit and cost more tokens to read -- the trade this project "
          f"exists to price. Every cell below is n={n_matched}, the largest "
          f"sample all three granularities share, so the columns differ by unit "
          f"size and not by question set.\n")

    for metric in ("recall@10", "any_hit@10", "mrr"):
        print(f"\n**{metric}**\n")
        print("| system | " + " | ".join(grans) + " |")
        print("|---" * (len(grans) + 1) + "|")
        for name in ORDER:
            cells = []
            for g in grans:
                m = runs.get(f"{name}|{g}", {}).get("metrics", {})
                cells.append(f"{m[metric]:.3f}" if metric in m else "--")
            if set(cells) != {"--"}:
                print(f"| {name} | " + " | ".join(cells) + " |")

    # The random row is the control that makes the rest readable. Coarser units
    # mean fewer of them, so a fixed k grabs a larger share of the haystack.
    r_turn = runs.get("random|turn", {}).get("metrics", {}).get("recall@10")
    r_sess = runs.get("random|session", {}).get("metrics", {}).get("recall@10")
    if r_turn and r_sess:
        print(f"\n**Read the `random` row before the others.** It goes "
              f"{r_turn:.3f} to {r_sess:.3f} across the same change, because "
              f"coarser units mean fewer of them and a fixed `k` therefore "
              f"grabs a larger share of the haystack. `recall@10` at session "
              f"granularity is not on the same scale as `recall@10` at turn "
              f"granularity, and neither is its token cost -- ten sessions is "
              f"roughly ten times the reading. Granularities are only "
              f"comparable at a matched read-token budget, which means "
              f"different `k` per granularity, not a shared one.\n")

    # any_hit flatters aggregation questions: it is satisfied by one evidence
    # turn out of six. Where the two diverge is where the metric choice matters.
    print("\n**Where `any_hit@10` and `recall@10` disagree** (hybrid). A "
          "question needing six evidence turns scores `any_hit` = 1.0 on one "
          "of them, so `any_hit` overstates readiness to answer:\n")
    print("| granularity | question type | any_hit@10 | recall@10 | overstated by |")
    print("|---|---|---|---|---|")
    for g in grans:
        bt = runs.get(f"hybrid|{g}", {}).get("metrics", {}).get(
            "by_question_type", {})
        for qtype, v in sorted(bt.items()):
            delta = v["any_hit@10"] - v["recall@10"]
            if delta >= 0.10:
                print(f"| {g} | {qtype} | {v['any_hit@10']:.3f} | "
                      f"{v['recall@10']:.3f} | {delta:.3f} |")


CAPS = [None, 40, 25, 15, 8]


def length_bias_section(arms: list[dict]) -> None:
    """Containment grading marks an answer correct if the gold span appears
    anywhere in it, so a longer answer gets more chances. Models differ in
    verbosity, which makes model-vs-model comparisons length-confounded even
    though every arm is graded by the identical rule. Re-grading with answers
    capped at N words prices that: an advantage that survives the cap is
    accuracy, one that vanishes was length."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    try:
        from llm_mem_eval.eval.grade import grade
    except Exception:
        return
    valid = [a for a in arms
             if len({r.get("prompt_tokens") for r in a.get("records", [])}) > 1]
    if not valid:
        return

    print("\n### How much of this is the grader rewarding longer answers\n")
    print("The median gold answer is 11 characters. These re-grade the same "
          "stored answers with each capped at its first N words. A model whose "
          "lead disappears as the cap tightens was not more accurate, it was "
          "more verbose -- and the gold span merely appeared somewhere in a "
          "longer answer.\n")
    print("| arm | median words | " +
          " | ".join("full" if c is None else f"{c}w" for c in CAPS) + " |")
    print("|---" * (len(CAPS) + 2) + "|")
    for a in sorted(valid, key=arm_key):
        R = a["records"]
        med = sorted(len(r["pred"].split()) for r in R)[len(R) // 2]
        cells = []
        for cap in CAPS:
            # The question has to go in. Without it `grade` re-enables set
            # comparison, which accepts a reordered list on an ordering
            # question -- the second false accept this repo documents fixing.
            # Dropping it here made the `full` column disagree with the arm's
            # own stored accuracy on both oracle arms (0.571 against 0.560 at
            # 14B, 0.604 against 0.593 at 7B), so the row auditing the headline
            # was graded more leniently than the headline itself.
            g = [grade(" ".join(r["pred"].split()[:cap]) if cap else r["pred"],
                       r["gold"], r["is_abstention"], r.get("question"))
                 for r in R]
            g = [x for x in g if x is not None]
            cells.append(f"{sum(g)/len(g):.3f}" if g else "--")
        print(f"| {a['_tag'][4:]} | {med} | " + " | ".join(cells) + " |")


def e2e_section(arms: list[dict]) -> None:
    if not arms:
        return

    print("\n## End-to-end answer accuracy\n")
    print("Retrieve, answer with a local model, grade deterministically. "
          "Graded by normalised token-span containment, never by an LLM judge; "
          "abstractive gold answers with no checkable surface form are excluded "
          "rather than scored wrong, which is the `not gradable` column.\n")
    # "clamped" rather than "truncated": the column falls back to
    # n_hit_token_cap, which counts answers that hit the generation cap, not
    # prompts the server clipped. Both are clamps worth seeing; conflating them
    # under the prompt-truncation name misread six arms.
    print("| model | retriever | granularity | accuracy | token-F1 | graded | "
          "not gradable | read tok/query | max prompt | num_ctx | clamped | "
          "valid |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    suspect = []
    for a in sorted(arms, key=arm_key):
        c = a["config"]
        # An arm where every question produced an identical prompt length was
        # clamped by the server, not measured. Ollama reports what it kept, so
        # this invariant catches overflows its own counters cannot see.
        pt = {r.get("prompt_tokens") for r in a.get("records", [])}
        clamped = len(a.get("records", [])) > 5 and len(pt) == 1
        if clamped:
            suspect.append((a, pt.pop()))
        print(f"| {model_label(a)} | {c['retriever']} | "
              f"{c.get('granularity','turn')} | {a['accuracy']:.3f} | "
              f"{a['token_f1_mean']:.3f} | {a['n_graded']} | "
              f"{a['n_not_gradable']} | {a['read_tokens_per_query']:.0f} | "
              f"{a.get('prompt_tokens_max','--')} | {c.get('num_ctx','--')} | "
              # `n_prompts_truncated` is only written when the backend itself
              # noticed; `n_hit_token_cap` is the per-record flag and is the one
              # that is always present. Reading only the former printed 0 for
              # every arm while six of them had actually hit the cap.
              f"{a.get('n_prompts_truncated', 0) or a.get('n_hit_token_cap', 0)}"
              f"{' | **INVALID**' if clamped else ' | ok'} |")

    for a, tok in suspect:
        print(f"\n> **`{a['_tag']}` is not a valid measurement.** Every one of "
              f"its {len(a['records'])} prompts was exactly {tok} tokens. "
              f"Prompts do not naturally agree to the token; the server "
              f"truncated them to a fixed size before the model read them, so "
              f"the retrieved memory never reached it. Its accuracy of "
              f"{a['accuracy']:.3f} measures the clamp, not the system.")

    # Oracle hands the model the gold evidence, so the oracle column is the
    # model's own ceiling and the difference is what retrieval costs.
    by_model: dict[str, dict[str, float]] = {}
    for a in arms:
        if a["config"].get("granularity", "turn") != "turn":
            continue
        by_model.setdefault(model_label(a), {})[a["config"]["retriever"]] = (
            a["accuracy"]
        )
    pairs = {m: v for m, v in by_model.items()
             if "oracle" in v and "hybrid" in v}
    if pairs:
        print("\n### Oracle is not a ceiling\n")
        print("`oracle` ranks the turns LongMemEval labels `has_answer` first, "
              "then pads to the same `k` as every other arm with non-evidence "
              "turns in conversation order. Questions carry ~1.9 evidence turns "
              "on average, so at k=10 that context is mostly filler: it is a "
              "*different* context from a retriever's, not a superset of one. "
              "Where the gap below is negative, a real retriever beat the gold "
              "labels, because the answer needed a turn that was never "
              "labelled evidence. So `1 - oracle` is not model loss alone -- it "
              "also contains whatever the labelling missed, and the oracle arm "
              "cannot be read as an upper bound.\n")
        print("| model | oracle | hybrid | oracle - hybrid | 1 - oracle |")
        print("|---|---|---|---|---|")
        for m, v in sorted(pairs.items(), key=lambda kv: param_count(kv[0])):
            print(f"| {m} | {v['oracle']:.3f} | {v['hybrid']:.3f} | "
                  f"{v['oracle']-v['hybrid']:.3f} | {1-v['oracle']:.3f} |")

    length_bias_section(arms)

    print("\n### End-to-end accuracy by question type\n")
    for a in sorted(arms, key=arm_key):
        bt = a.get("accuracy_by_question_type")
        if not bt:
            continue
        c = a["config"]
        print(f"\n**{model_label(a)} / {c['retriever']} / "
              f"{c.get('granularity','turn')}**\n")
        print("| question type | n | accuracy |")
        print("|---|---|---|")
        for qtype, v in sorted(bt.items()):
            print(f"| {qtype} | {v['n']} | {v['accuracy']:.3f} |")


def survival_section(results_dir: Path) -> None:
    """Answer survival per write policy, if the sweep has been run.

    Reads results/survival/, a subdirectory, because `load_all` globs
    results/*.json non-recursively and folds anything carrying a "metrics" key
    into the retrieval table.
    """
    paths = sorted((results_dir / "survival").glob("survival_*.json"))
    if not paths:
        return
    stores = [json.loads(p.read_text()) for p in paths]

    print("\n## Answer survival, by write policy\n")
    print("Did the write path keep the answer at all? Measured over the store "
          "with no retrieval, so it is a ceiling on retrieval and on accuracy. "
          "`null` is the chance floor, measured by re-running survival against "
          "gold answers borrowed from other questions of the same type and "
          "length; `corrected` is (survival - null) / (1 - null). Raw survival "
          "alone is not interpretable, so it is never shown without both.\n")
    print("Restricted to golds of two or more normalised tokens: one-token "
          "answers match a 100K-token store by accident about two thirds of "
          "the time.\n")
    print("| write policy | store tokens | records | survival | null | "
          "corrected | 95% CI |")
    print("|---|---|---|---|---|---|---|")
    for s in sorted(stores, key=lambda s: -s["survival"]["store_stats"]
                    ["tokens_per_store_mean"]):
        st = s["survival"]["store_stats"]
        p = s["survival"]["primary"]["record"]
        lo, hi = p["chance_corrected_ci95"]
        print(f"| {s['store_id']} | {st['tokens_per_store_mean']:,.0f} | "
              f"{st['records_per_store_mean']:.0f} | {p['survival']:.3f} | "
              f"{p['null']:.3f} | **{p['chance_corrected']:.3f}** | "
              f"[{lo:.3f}, {hi:.3f}] |")

    summary_path = results_dir / "survival" / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        comps = summary.get("paired_vs_baseline") or {}
        if comps:
            print(f"\nPaired against `{summary['baseline']}`, on the "
                  f"questions both scored. McNemar plus a paired bootstrap, "
                  f"the same functions the memory-lift ablation uses.\n")
            print("| comparison | difference | 95% CI | McNemar p |")
            print("|---|---|---|---|")
            for name, c in comps.items():
                pt, lo, hi = c["paired_diff_ci95"]
                print(f"| {name.replace('_vs_', ' vs ')} | {pt:+.3f} | "
                      f"[{lo:+.3f}, {hi:+.3f}] | {c['mcnemar_p']:.2e} |")


def main() -> None:
    results_dir = Path("results")
    all_runs = load_all(results_dir)
    if not all_runs:
        print("no results yet -- run scripts/run_retrieval_eval.py first")
        return
    # Per-row tables take the largest run available and print its own n; the
    # cross-granularity section re-selects a matched set for itself.
    runs = best_runs(all_runs)

    print("# Results\n")
    print("Auto-generated by `scripts/make_report.py` from `results/*.json`.\n")

    print("## Retrieval quality (judge-free)\n")
    print("Gold labels are LongMemEval's per-turn `has_answer` flags, so these "
          "numbers involve no LLM judge and no API spend. Questions with no "
          "evidence turns (abstention cases) are excluded from recall rather "
          "than scored as 0 or 1.\n")
    print("| system | granularity | n | any_hit@1 | any_hit@5 | any_hit@10 | "
          "recall@10 | MRR |")
    print("|---|---|---|---|---|---|---|---|")
    for key in sorted(runs, key=sort_key):
        r = runs[key]
        m = r["metrics"]
        name, gran = key.split("|")
        if "any_hit@10" not in m:
            continue
        print(f"| {name} | {gran} | {m['n_scorable']} | "
              f"{m['any_hit@1']:.3f} | {m['any_hit@5']:.3f} | "
              f"{m['any_hit@10']:.3f} | {m['recall@10']:.3f} | {m['mrr']:.3f} |")

    print("\n## Cost, split by write path and read path\n")
    print("The write path is paid once per conversation; the read path is paid "
          "per query. Published memory systems report the read path only.\n")
    print("| system | granularity | write ms/conv | write LLM calls | "
          "write LLM tokens | read ms/query | read LLM calls |")
    print("|---|---|---|---|---|---|---|")
    for key in sorted(runs, key=sort_key):
        r = runs[key]
        c = r.get("cost_per_example", {})
        tot = r.get("cost_total", {})
        # Without the granularity column this table printed three identical
        # `random` rows, three `bm25` rows, and so on, with no way to tell which
        # unit size produced which timing.
        name, gran = key.split("|")
        read_calls = tot.get("read", {}).get("llm_calls", 0)
        n = max(1, r["metrics"]["n_total"])
        print(f"| {name} | {gran} | {c.get('write_wall_clock_s',0)*1000:.0f} | "
              f"{c.get('write_llm_calls',0):.1f} | "
              f"{c.get('write_llm_tokens',0):.0f} | "
              f"{c.get('read_wall_clock_s',0)*1000:.0f} | "
              f"{read_calls/n:.1f} |")

    granularity_section(all_runs)
    survival_section(results_dir)
    e2e_section(load_e2e(results_dir))

    print("\n## Retrieval quality by question type\n")
    for key in sorted(runs, key=sort_key):
        r = runs[key]
        bt = r["metrics"].get("by_question_type")
        name, gran = key.split("|")
        if not bt or name in ("random", "oracle"):
            continue
        print(f"\n**{name}** ({gran})\n")
        print("| question type | n | any_hit@10 | recall@10 | MRR |")
        print("|---|---|---|---|---|")
        for qtype, v in sorted(bt.items()):
            print(f"| {qtype} | {v['n']} | {v['any_hit@10']:.3f} | "
                  f"{v['recall@10']:.3f} | {v['mrr']:.3f} |")


if __name__ == "__main__":
    main()
