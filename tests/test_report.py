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
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from memllm.eval.grade import grade  # noqa: E402

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
    from memllm.data.loader import load_examples

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
    from make_report import load_all

    keys = set(load_all(REPO / "results"))
    assert keys, "no runs found; the glob or the artifacts moved"

    known_retrievers = {"bm25", "dense", "hybrid", "oracle", "recency",
                        "random", "none"}
    for key in keys:
        name = key.split("|")[0]
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


def test_survival_artifacts_live_in_a_subdirectory() -> None:
    """`load_all`'s glob is non-recursive, which is the whole defence."""
    top_level = {p.name for p in (REPO / "results").glob("*.json")}
    assert not any(n.startswith("survival") for n in top_level), (
        "survival artifacts belong in results/survival/ -- at the top level "
        "they are one 'metrics' key away from becoming retriever rows"
    )
