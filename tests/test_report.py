"""Tests for the generated report agreeing with the arms it reports on.

RESULTS.md is regenerated from `results/*.json` rather than hand-typed, which
only helps if the regeneration grades the same way the runs did. It did not:
`length_bias_section` called `grade()` without the question, which re-enables
set comparison, which accepts a reordered list on an ordering question. The
`full` column -- nominally the arm's own accuracy -- came out at 0.604 against a
stored 0.593 on the 7B oracle arm and 0.571 against 0.560 at 14B. The row
auditing the headline was graded more leniently than the headline.

Runs from the stored predictions. No model, no benchmark download.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from llm_mem_eval.eval.grade import grade  # noqa: E402

ARMS = sorted((REPO / "results").glob("e2e_*.json"))


def _accuracy(records, cap=None, with_question=True):
    scores = [grade(" ".join(r["pred"].split()[:cap]) if cap else r["pred"],
                    r["gold"], r["is_abstention"],
                    r.get("question") if with_question else None)
              for r in records]
    scores = [s for s in scores if s is not None]
    return sum(scores) / len(scores)


@pytest.mark.parametrize("path", ARMS, ids=lambda p: p.stem)
def test_regrading_an_arm_uncapped_reproduces_its_stored_accuracy(path):
    """The `full` column of the length-decay table must be the arm's accuracy.
    If it is not, the two are using different graders and the decay it reports
    is measured against the wrong baseline."""
    payload = json.loads(path.read_text())
    assert _accuracy(payload["records"]) == pytest.approx(payload["accuracy"], abs=1e-9)


def test_dropping_the_question_really_does_change_a_stored_arm():
    """Guards the fix rather than the symptom. If passing the question stops
    mattering anywhere in the stored results, this test is free to be deleted --
    but silently losing the argument again should not be free."""
    differing = []
    for path in ARMS:
        payload = json.loads(path.read_text())
        with_q = _accuracy(payload["records"], with_question=True)
        without_q = _accuracy(payload["records"], with_question=False)
        if with_q != without_q:
            differing.append((path.stem, with_q, without_q))

    assert differing, (
        "no stored arm distinguishes grade(question=...) any more; the "
        "ordering false accept may have been fixed another way, or the "
        "affected arms were dropped"
    )
    for name, with_q, without_q in differing:
        assert without_q > with_q, f"{name}: dropping the question should only be more lenient"


def test_a_missing_split_explains_the_download_instead_of_raising():
    """The likeliest failure on a clean clone, and the least self-explanatory:
    `open()` on an absent 278 MB download says only that a path is missing."""
    from llm_mem_eval.data.loader import load_examples

    with pytest.raises(SystemExit) as e:
        load_examples("data/raw/definitely_not_downloaded")

    msg = str(e.value)
    assert "hf_hub_download" in msg, "must name the way to get the file"
    assert "xiaowu0162/longmemeval" in msg, "must name the dataset"
    assert "definitely_not_downloaded" in msg, "must name the split requested"
    assert "results/" in msg, "must say what still works without it"


def test_new_artifacts_do_not_leak_into_the_retrieval_table() -> None:
    """results/ holds non-run artifacts; the report must ignore them.

    `make_report.load_all` globs `results/*.json` and folds any value dict
    carrying a "metrics" key into the retrieval table. That heuristic is one
    accidentally-named key away from silently listing the benchmark audit or the
    write-cost model as a retriever row. This pins the rule rather than the
    current file list, so a future artifact fails here instead of in RESULTS.md.
    """
    sys.path.insert(0, str(REPO / "scripts"))
    from make_report import is_recency_variant, load_all

    keys = set(load_all(REPO / "results"))
    assert keys, "no runs found; the glob or the artifacts moved"

    known_retrievers = {"bm25", "dense", "hybrid", "oracle", "recency",
                        "random", "none"}
    for key in keys:
        name = key.split("|")[0]
        # A recency-weighted hybrid arm is a real run and belongs in load_all;
        # it is one system at one setting rather than a new system, so what has
        # to be pinned about it is that it stays out of the system-comparison
        # tables. That is the test below, not this one.
        if is_recency_variant(key):
            continue
        assert name in known_retrievers, (
            f"{name!r} was folded into the retrieval table but is not a "
            f"retriever. A non-run artifact in results/ grew a 'metrics' key, "
            f"or a new artifact belongs in a subdirectory."
        )

    for artifact in ("benchmark_audit.json", "write_cost_model.json"):
        path = REPO / "results" / artifact
        if path.exists():
            payload = json.loads(path.read_text())
            assert not any(
                isinstance(v, dict) and "metrics" in v
                for v in payload.values()
            ), f"{artifact} would be read as a run"


def test_the_granularity_table_never_mixes_two_question_sets() -> None:
    """A cross-granularity row is only a granularity comparison if `n` is held
    fixed. `load_all` used to keep the largest run per (retriever, granularity),
    so re-running one granularity at n=500 would have compared it against the
    others at n=100 and reported the sample difference as a unit-size effect.
    """
    sys.path.insert(0, str(REPO / "scripts"))
    from make_report import best_runs, matched_runs

    runs = {
        "bm25|turn|100": {"metrics": {"n_total": 100, "recall@10": 0.1}},
        "bm25|turn|500": {"metrics": {"n_total": 500, "recall@10": 0.9}},
        "bm25|session|100": {"metrics": {"n_total": 100, "recall@10": 0.2}},
    }

    # The comparison drops to the largest n that every granularity has.
    matched, n = matched_runs(runs, ["turn", "session"])
    assert n == 100
    assert matched["bm25|turn"]["metrics"]["n_total"] == 100
    assert matched["bm25|session"]["metrics"]["n_total"] == 100

    # Per-row tables still get the biggest run, because they print their own n.
    assert best_runs(runs)["bm25|turn"]["metrics"]["n_total"] == 500

    # No shared n means no comparison, rather than a wrong one.
    assert matched_runs(
        {"bm25|turn|500": runs["bm25|turn|500"],
         "bm25|session|100": runs["bm25|session|100"]},
        ["turn", "session"],
    ) == ({}, None)

    # A granularity that is absent entirely is not silently dropped from the
    # intersection, which would leave the remaining columns looking matched.
    assert matched_runs(runs, ["turn", "session", "user_turn"]) == ({}, None)


def test_survival_artifacts_live_in_a_subdirectory() -> None:
    """`load_all`'s glob is non-recursive, which is the whole defence."""
    top_level = {p.name for p in (REPO / "results").glob("*.json")}
    assert not any(n.startswith("survival") for n in top_level), (
        "survival artifacts belong in results/survival/ -- at the top level "
        "they are one 'metrics' key away from becoming retriever rows"
    )


def test_recency_sweep_arms_stay_out_of_the_system_comparison_tables() -> None:
    """A parameter sweep must not be read as a set of competing systems.

    `load_all` keys runs `<retriever>|<granularity>|<n>`, so a weighted hybrid
    run at turn/n=500 lands on exactly the key the unweighted baseline occupies.
    Two things could go wrong and neither would raise: the sweep silently
    replacing the baseline it is measured against, or every weight appearing as
    its own row in the retrieval and cost tables. Both are checked against the
    rendered report rather than against the loader, because the rendered report
    is the thing that has to be right.
    """
    import contextlib
    import io

    sys.path.insert(0, str(REPO / "scripts"))
    from make_report import is_recency_variant, load_all, main

    if not any(is_recency_variant(k) for k in load_all(REPO / "results")):
        print("  skip recency hold-out test (no sweep artifacts present)")
        return

    buf = io.StringIO()
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        with contextlib.redirect_stdout(buf):
            main()
    finally:
        os.chdir(cwd)
    report = buf.getvalue()

    sections = report.split("\n## ")
    for section in sections:
        title = section.split("\n", 1)[0]
        weighted = [ln for ln in section.splitlines()
                    if ln.startswith("| hybrid_rw")]
        if title.startswith("A recency prior"):
            continue
        assert not weighted, (
            f"weighted hybrid rows leaked into the {title!r} section: "
            f"{weighted[:2]}"
        )

    assert "A recency prior" in report, "the sweep ran but reported nothing"
    # The baseline must survive alongside the sweep, not be overwritten by it.
    retrieval = report.split("\n## ")[1]
    assert any(ln.startswith("| hybrid | turn |") for ln in retrieval.splitlines()), \
        "the unweighted hybrid baseline vanished from the retrieval table"


def test_readme_survival_rows_match_their_artifacts() -> None:
    """The README's survival table is hand-typed; RESULTS.md is not.

    That asymmetry is where a wrong number survives review: RESULTS.md is
    regenerated and CI diffs it, so drift there is caught mechanically, while a
    README row can be edited into disagreement with the artifact it quotes and
    nothing complains. This parses the rows back out and checks them, so the
    hand-typed half is held to the same standard as the generated half.
    """
    readme = (REPO / "README.md").read_text()
    # `| LexRank @ 50% | 52,166 | 497 | 0.709 | 0.118 | **0.670** |`
    row = re.compile(
        r"^\|\s*(LexRank|lead-k|tail-k)\s*@\s*(\d+)%\s*\|\s*([\d,]+)\s*\|\s*"
        r"(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*\*\*([\d.]+)\*\*\s*\|",
        re.MULTILINE,
    )
    policy = {"LexRank": "lexrank", "lead-k": "leadk", "tail-k": "tailk"}

    checked = 0
    for name, pct, tokens, records, survival, null, corrected in row.findall(readme):
        path = REPO / "results" / "survival" / f"survival_{policy[name]}_{pct}pct.json"
        if not path.exists():
            continue
        s = json.loads(path.read_text())["survival"]
        st, p = s["store_stats"], s["primary"]["record"]
        where = f"README row {name} @ {pct}%"
        assert int(tokens.replace(",", "")) == round(st["tokens_per_store_mean"]), where
        assert int(records) == round(st["records_per_store_mean"]), where
        assert float(survival) == pytest.approx(p["survival"], abs=5e-4), where
        assert float(null) == pytest.approx(p["null"], abs=5e-4), where
        assert float(corrected) == pytest.approx(
            p["chance_corrected"], abs=5e-4
        ), where
        checked += 1

    assert checked >= 4, (
        f"only {checked} README survival rows matched an artifact; the table "
        f"format changed and this test stopped checking anything"
    )
