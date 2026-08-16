"""Tests for write policies.

Two of these guard silent-corruption bugs rather than wrong answers. A store
with non-contiguous `unit_id`s still runs and still produces numbers -- they are
just quietly wrong, because retrievers return ids that `score_example`
intersects against a set built from the same list. A store whose `session_date`
does not parse turns the recency control into a no-op without any error. Both
are asserted here and by `check_store` at build time.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memllm.cost import CostLedger, count_tokens  # noqa: E402
from memllm.data.loader import Example, Turn, parse_date  # noqa: E402
from memllm.write import (  # noqa: E402
    ExtractiveSelectionPolicy,
    TruncatedVerbatimPolicy,
    VerbatimPolicy,
    build_policy,
    check_store,
)

SESSIONS = [("s1", "2023/01/01 (Sun) 10:00"),
            ("s2", "2023/02/01 (Wed) 10:00"),
            ("s3", "2023/03/01 (Wed) 10:00")]


def _example() -> Example:
    turns, idx = [], 0
    for si, (sid, date) in enumerate(SESSIONS):
        for j in range(4):
            turns.append(Turn(
                "user" if j % 2 == 0 else "assistant",
                f"Session {si} message {j}. It has a second sentence here. "
                f"And a third one as well, for splitting.",
                sid, date, si, idx, si == 0 and j == 0,
            ))
            idx += 1
    return Example("q1", "single-session-user", "What happened?", "peanuts",
                   "2023/04/01 (Sat) 10:00", turns)


ALL_POLICIES = [
    VerbatimPolicy("turn"),
    TruncatedVerbatimPolicy(0.5, "recency"),
    TruncatedVerbatimPolicy(0.25, "random", seed=0),
    ExtractiveSelectionPolicy(0.5),
    ExtractiveSelectionPolicy(0.1),
]


@pytest.mark.parametrize("policy", ALL_POLICIES, ids=lambda p: p.name)
def test_every_policy_honours_the_store_contracts(policy) -> None:
    units = policy.build(_example(), CostLedger())
    check_store(units)  # raises on either violation
    assert [u.unit_id for u in units] == list(range(len(units)))
    assert all(parse_date(u.session_date) != (0, 0, 0) for u in units)
    assert all(u.text.strip() for u in units), "empty records are not retrievable"


@pytest.mark.parametrize("policy", ALL_POLICIES, ids=lambda p: p.name)
def test_every_policy_bills_only_the_write_phase(policy) -> None:
    led = CostLedger()
    policy.build(_example(), led)
    assert led.read.wall_clock_s == 0.0
    assert led.read.llm_calls == 0
    # These arms call no LLM at all; that is the point of them.
    assert led.write.llm_calls == 0


def test_verbatim_is_exactly_units() -> None:
    ex = _example()
    got = VerbatimPolicy("turn").build(ex, CostLedger())
    assert got == ex.units("turn"), "verbatim must not diverge from units()"


def test_truncated_respects_the_budget() -> None:
    ex = _example()
    full = sum(count_tokens(u.text) for u in ex.units("turn"))
    for frac in (0.1, 0.25, 0.5):
        units = TruncatedVerbatimPolicy(frac, "recency").build(ex, CostLedger())
        kept = sum(count_tokens(u.text) for u in units)
        assert kept <= frac * full + 1e-6, f"over budget at {frac}"
        assert kept > 0, f"nothing kept at {frac}"


def test_truncated_recency_prefers_recent_sessions() -> None:
    units = TruncatedVerbatimPolicy(0.4, "recency").build(_example(), CostLedger())
    kept = {u.session_id for u in units}
    assert "s3" in kept, "the most recent session should survive first"
    assert "s1" not in kept, "the oldest session should be dropped first"


def test_truncated_restores_conversation_order() -> None:
    """Retrievers do not care, but the answering prompt reads units in order."""
    units = TruncatedVerbatimPolicy(0.6, "recency").build(_example(), CostLedger())
    dates = [parse_date(u.session_date) for u in units]
    assert dates == sorted(dates), "store left in reverse-chronological order"


def test_truncated_random_is_seeded_per_example_not_per_run() -> None:
    ex = _example()
    a = TruncatedVerbatimPolicy(0.5, "random", seed=0).build(ex, CostLedger())
    b = TruncatedVerbatimPolicy(0.5, "random", seed=0).build(ex, CostLedger())
    c = TruncatedVerbatimPolicy(0.5, "random", seed=1).build(ex, CostLedger())
    assert [u.text for u in a] == [u.text for u in b], "not reproducible"
    assert [u.text for u in a] != [u.text for u in c], "seed has no effect"


def test_leadk_is_broader_than_truncation_at_the_same_budget() -> None:
    """The property the arm exists to isolate: breadth, not depth."""
    ex = _example()
    lead = ExtractiveSelectionPolicy(0.4).build(ex, CostLedger())
    trunc = TruncatedVerbatimPolicy(0.4, "recency").build(ex, CostLedger())
    assert len(lead) > len(trunc), "lead-k should touch more turns"
    assert len({u.session_id for u in lead}) == len(SESSIONS), \
        "lead-k should reach every session"


def test_leadk_keeps_lead_sentences_not_arbitrary_ones() -> None:
    """Each record must be a sentence-prefix of the turn it came from."""
    ex = _example()
    source = {u.unit_id: u.text for u in ex.units("turn")}
    units = ExtractiveSelectionPolicy(0.2).build(ex, CostLedger())
    assert units, "nothing kept"
    for u in units:
        assert len(u.provenance) == 1, "a lead-k record comes from one turn"
        original = source[u.provenance[0]]
        first_kept = u.text.split(".")[0]
        assert original.startswith(first_kept), (
            f"record does not start at the turn's first sentence: "
            f"{u.text[:40]!r} vs {original[:40]!r}"
        )


def test_check_store_catches_noncontiguous_ids() -> None:
    units = VerbatimPolicy("turn").build(_example(), CostLedger())
    with pytest.raises(ValueError, match="contiguous"):
        check_store([units[0], replace(units[1], unit_id=7)])


def test_check_store_catches_unparseable_dates() -> None:
    units = VerbatimPolicy("turn").build(_example(), CostLedger())
    with pytest.raises(ValueError, match="session_date"):
        check_store([replace(u, session_date="whenever") for u in units])


def test_build_policy_parses_specs() -> None:
    assert build_policy("verbatim_turn").name == "verbatim_turn"
    p = build_policy("truncated_recency_25")
    assert p.fraction == 0.25 and p.rule == "recency" and p.seed == 0
    p = build_policy("truncated_random_25_s2")
    assert p.rule == "random" and p.seed == 2
    assert build_policy("leadk_10").fraction == 0.1
    with pytest.raises(ValueError):
        build_policy("nonsense_policy")
