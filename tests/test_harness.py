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


if __name__ == "__main__":
    print("harness smoke tests")
    test_granularity_preserves_evidence()
    test_cost_split()
    test_zero_evidence_excluded()
    test_bm25_finds_obvious_evidence()
    test_embed_cache_replays_cost_not_disk_read()
    test_real_data_loads()
    print("all passed")
