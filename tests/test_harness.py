"""Smoke tests for the harness. Run: python tests/test_harness.py

These guard the two things that would silently corrupt every number: the gold
evidence labels surviving unit construction, and the write/read cost split.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memllm.cost import CostLedger  # noqa: E402
from memllm.data.loader import Example, Turn, load_examples  # noqa: E402
from memllm.eval.retrieval_metrics import aggregate, score_example  # noqa: E402
from memllm.retrieval.bm25 import BM25Retriever  # noqa: E402

DATA = Path("data/raw/longmemeval_oracle")


def _toy_example() -> Example:
    turns = [
        Turn("user", "I am allergic to peanuts.", "s1", "2023/01/01 (Sun) 10:00", 0, 0, True),
        Turn("assistant", "Noted, I will avoid peanuts.", "s1", "2023/01/01 (Sun) 10:00", 0, 1, False),
        Turn("user", "The weather is nice today.", "s2", "2023/02/01 (Wed) 10:00", 1, 2, False),
    ]
    return Example("q1", "single-session-user", "What am I allergic to?",
                   "peanuts", "2023/03/01 (Wed) 10:00", turns)


def test_granularity_preserves_evidence() -> None:
    ex = _toy_example()
    for gran in ("turn", "user_turn", "session"):
        units = ex.units(gran)
        assert any(u.is_evidence for u in units), f"evidence lost at {gran}"
        assert len({u.unit_id for u in units}) == len(units), "unit_ids not unique"
    assert len(ex.units("turn")) == 3
    assert len(ex.units("user_turn")) == 2
    assert len(ex.units("session")) == 2
    print("  ok  granularity preserves evidence labels")


def test_cost_split() -> None:
    led = CostLedger()
    led.add_llm("write", 1000, 50)
    led.add_llm("read", 100, 10)
    assert led.write.llm_calls == 1 and led.read.llm_calls == 1
    # Write is paid once, read per query -- the amortisation contract.
    one = led.dollars_at_n_queries(1)
    hundred = led.dollars_at_n_queries(100)
    assert hundred > one
    expected = led.write.dollars() + 100 * led.read.dollars()
    assert abs(hundred - expected) < 1e-12
    print("  ok  cost splits write/read and amortises correctly")


def test_zero_evidence_excluded() -> None:
    ex = _toy_example()
    units = ex.units("turn")
    for u in units:
        u.is_evidence = False
    res = score_example(ex.question_id, ex.question_type, True, units, [(0, 1.0)])
    assert res.recall_at(10) is None, "zero-evidence must not score as 0 or 1"
    agg = aggregate([res])
    assert agg["n_scorable"] == 0 and agg["n_zero_evidence"] == 1
    print("  ok  zero-evidence examples excluded, not scored")


def test_bm25_finds_obvious_evidence() -> None:
    ex = _toy_example()
    units = ex.units("turn")
    led = CostLedger()
    r = BM25Retriever()
    r.index(units, led)
    hits = r.search(ex.question, 3, led)
    res = score_example(ex.question_id, ex.question_type, False, units, hits)
    assert res.any_hit_at(3), "BM25 missed a lexically obvious match"
    assert led.write.wall_clock_s > 0 and led.read.wall_clock_s > 0
    assert led.write.llm_calls == 0, "BM25 must make zero LLM calls"
    print("  ok  bm25 retrieves obvious evidence at zero LLM cost")


def test_embed_cache_replays_cost_not_disk_read() -> None:
    """The cache must report the cost of *computing* embeddings, never the cost
    of reading them back. Getting this wrong would zero out the write-path
    number this project exists to measure."""
    import shutil
    import tempfile

    import numpy as np

    from memllm.retrieval.embed_cache import EmbeddingCache

    tmp = Path(tempfile.mkdtemp())
    try:
        cache = EmbeddingCache(enabled=True, cache_dir=tmp)
        emb = np.random.rand(7, 384).astype("float32")
        true_cost = {"wall_clock_s": 11.843, "tokens": 103_000, "n_items": 7}
        cache.put("m", "k", emb, true_cost)

        got = cache.get("m", "k", 7)
        assert got is not None, "cache miss on a key we just wrote"
        emb2, cost2 = got
        assert np.allclose(emb, emb2)
        assert cost2["wall_clock_s"] == true_cost["wall_clock_s"]
        assert cost2["tokens"] == true_cost["tokens"]

        # A different unit count must miss rather than return stale vectors.
        assert cache.get("m", "k", 8) is None
        assert cache.get("other-model", "k", 7) is None
        assert cache.used_replayed_timings is True
        print("  ok  embedding cache replays true compute cost")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_real_data_loads() -> None:
    if not DATA.exists():
        print(f"  skip real-data test ({DATA} not downloaded)")
        return
    examples = load_examples(DATA, limit=5)
    assert len(examples) == 5
    ex = examples[0]
    assert ex.turns and ex.question and ex.answer
    assert any(t.has_answer for t in ex.turns), "oracle split must carry evidence"
    print(f"  ok  real data loads ({len(ex.turns)} turns in example 0)")


def test_grader_rejects_near_misses() -> None:
    """The failure mode that inflates every judged benchmark: accepting wrong
    answers. Numeric substrings are the sneakiest case."""
    from memllm.eval.grade import grade

    assert grade("You said Target.", "Target") is True
    assert grade("It was about 800 dollars.", "$800") is True
    assert grade("twenty books", "20") is True          # number word
    assert grade("Feb 14", "February 14th") is True     # month abbrev + ordinal

    assert grade("120 pages", "20") is False            # substring trap
    assert grade("You said Walmart.", "Target") is False
    assert grade("I don't know, it wasn't mentioned.", "Target") is False
    assert grade("", "Target") is False

    # Abstention inverts: declining is the correct behaviour.
    assert grade("I cannot find that information.", "n/a", is_abstention=True) is True
    assert grade("It was Target.", "n/a", is_abstention=True) is False

    # Both of these came from inspecting real 1.5B output, where the grader was
    # wrong in the direction that penalises a correct model.
    # 38 gold keys name an acceptable alternative in prose; matching the whole
    # string rejects an answer the key permits.
    alt = "1 day. 2 days (including the last day) is also acceptable."
    assert grade("Two days passed between them.", alt) is True
    assert grade("It was 1 day.", alt) is True
    assert grade("It was 9 days.", alt) is False
    # A gold that asserts unanswerability is an abstention even when the
    # question id is not flagged, and "don't have enough information" is a
    # refusal even though it says neither "don't know" nor "not mentioned".
    unanswerable = ("The information provided is not enough. You did not "
                    "mention buying an iPad case.")
    assert grade("I don't have enough information to determine that.",
                 unanswerable) is True
    assert grade("It took 3 days to arrive.", unanswerable) is False

    # Abstractive gold has no checkable surface form; abstain, do not guess.
    long_gold = ("The user would prefer responses that acknowledge their "
                 "interest in both thrill rides and special seasonal events.")
    assert grade("Something about theme parks.", long_gold) is None
    print("  ok  grader rejects near-misses, abstains on abstractive gold")


def test_grader_matches_regular_plurals() -> None:
    """Both cases are real predictions that were scored wrong. The question
    fixes the referent, so the plural carries no extra meaning."""
    from memllm.eval.grade import grade

    assert grade("You take a cocktail-making class on Fridays.", "Friday") is True
    assert grade("Your new Samsung TV is 55 inches.", "55-inch") is True
    assert grade("It lasted 3 days.", "3 day") is True

    # Guards against collisions, which would be false accepts. Short tokens and
    # the -ss/-us/-is endings are not plurals and must survive intact.
    assert grade("It was a bus.", "bu") is False
    assert grade("You took a class.", "cla") is False
    # No -ies -> -y rule, so a singular ending in -ie still matches its plural.
    assert grade("You watched movies.", "movie") is True
    print("  ok  grader matches regular plurals without colliding")


def test_grader_will_not_accept_a_reordered_answer_to_an_ordering_question() -> None:
    """Set comparison exists so an unordered list can be named in any order.
    Applied to a question whose answer IS the order, it accepts the wrong
    answer -- which it did, on a real 14B prediction, until the question was
    passed in."""
    from memllm.eval.grade import grade

    gold = "JetBlue, Delta, United, American Airlines"
    reordered = "JetBlue, Delta, American Airlines, and then United Airlines"
    ordering_q = "What is the order of airlines I flew with from earliest to latest?"

    assert grade(reordered, gold, question=ordering_q) is False
    assert grade(gold, gold, question=ordering_q) is True

    # Without an ordering question the set behaviour is unchanged: naming every
    # item in a different order is still correct.
    unordered_q = "What processes are used at the Lake Charles Refinery?"
    unordered_gold = ("Atmospheric distillation, fluid catalytic cracking (FCC), "
                      "alkylation, and hydrotreating.")
    rotated = ("fluid catalytic cracking (FCC), alkylation, hydrotreating, "
               "and atmospheric distillation")
    assert grade(rotated, unordered_gold, question=unordered_q) is True

    # Omitting the question keeps the older, more permissive behaviour, so no
    # existing caller silently changes verdict.
    assert grade(reordered, gold) is True
    print("  ok  grader rejects a reordered answer to an ordering question")


def test_grader_audit_has_zero_false_accepts() -> None:
    """Regression guard on the property the whole eval rests on.

    If the grader starts accepting known-wrong answers, every accuracy number in
    the repo becomes an overestimate, and nothing else here would catch it.
    """
    from memllm.eval.grade import grade
    from memllm.eval.grader_audit import audit_grader, build_audit_cases

    # Skip rather than fail when the split is absent, matching
    # test_real_data_loads. A fresh clone that downloaded only longmemeval_s
    # was aborting the whole Kaggle pre-flight here, which reads as a broken
    # grader rather than a missing 15 MB file.
    if not DATA.exists():
        print(f"  skip grader audit ({DATA} not downloaded)")
        return

    ex = load_examples(DATA)[:60]
    cases = build_audit_cases(ex, per_type=10)
    assert cases, "no audit cases constructed"

    report = audit_grader(
        cases,
        lambda c: grade(c.pred, c.gold, c.kind.endswith("abstention"), c.question),
    )
    assert report["false_accept_rate"] == 0.0, (
        f"grader accepts known-wrong answers: {report['false_accept_rate']}")
    hard = report["false_reject_rate_hard"]
    assert hard is None or hard <= 0.05, f"false-reject on rewrites: {hard}"
    print(f"  ok  grader audit: 0 false accepts over {report['n_cases']} constructed cases")


if __name__ == "__main__":
    print("harness smoke tests")
    test_granularity_preserves_evidence()
    test_cost_split()
    test_zero_evidence_excluded()
    test_bm25_finds_obvious_evidence()
    test_embed_cache_replays_cost_not_disk_read()
    test_real_data_loads()
    test_grader_rejects_near_misses()
    test_grader_matches_regular_plurals()
    test_grader_will_not_accept_a_reordered_answer_to_an_ordering_question()
    test_grader_audit_has_zero_false_accepts()
    print("all passed")
