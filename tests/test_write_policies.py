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

from llm_mem_eval.cost import CostLedger, count_tokens  # noqa: E402
from llm_mem_eval.data.loader import Example, Turn, parse_date  # noqa: E402
from llm_mem_eval.write import (  # noqa: E402
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
    ExtractiveSelectionPolicy(0.5, rule="tail"),
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


def test_tail_k_selects_from_the_other_end_at_the_same_budget() -> None:
    """The control that decides what the lead-k result means.

    If lead-k wins on survival because breadth helps, tail-k should do about as
    well. If it wins because people state facts at the start of a message, tail-k
    collapses. The two arms must therefore differ only in which sentences
    survive -- same budget, same number of records, same reading order.
    """
    ex = _example()
    lead = ExtractiveSelectionPolicy(0.5, rule="lead").build(ex, CostLedger())
    tail = ExtractiveSelectionPolicy(0.5, rule="tail").build(ex, CostLedger())

    lead_tok = sum(count_tokens(u.text) for u in lead)
    tail_tok = sum(count_tokens(u.text) for u in tail)
    assert abs(lead_tok - tail_tok) / max(lead_tok, 1) < 0.15, "budgets diverged"
    assert len(lead) == len(tail), "different number of records"
    assert [u.text for u in lead] != [u.text for u in tail], "rule had no effect"

    source = {u.unit_id: u.text for u in ex.units("turn")}
    for u in tail:
        original = source[u.provenance[0]]
        assert not original.startswith(u.text.split(".")[0]), \
            "tail-k kept the turn's opening sentence"


def test_build_policy_parses_the_tail_rule() -> None:
    assert build_policy("tailk_25").rule == "tail"
    assert build_policy("leadk_25").rule == "lead"
    assert build_policy("tailk_25").name == "tailk_25pct"


def test_a_spec_is_not_a_name() -> None:
    """Artifacts and paired comparisons are keyed by `.name`, not by the spec.

    They differ for every fractional policy, so anything that takes a spec
    from a command line and looks it up among names has to resolve it through
    build_policy first. run_survival_eval.py did not, and its `--baseline`
    silently produced an empty comparison block for every arm but one.
    """
    for spec in ("truncated_recency_25", "truncated_random_25_s1",
                 "leadk_50", "tailk_5"):
        assert build_policy(spec).name != spec, f"{spec} round-tripped"
    # verbatim_turn is the exception, which is why the bug stayed hidden: it
    # was the default baseline and the only spec that is also a name.
    assert build_policy("verbatim_turn").name == "verbatim_turn"


def test_build_policy_keeps_two_word_granularities_whole() -> None:
    """`user_turn` is a granularity; splitting on the first `_` truncated it.

    The old parse produced a policy named `verbatim_user` that constructed
    fine and only failed later, inside build(), with `unknown granularity:
    user` -- an error that points nowhere near the spec that caused it.
    """
    p = build_policy("verbatim_user_turn")
    assert p.granularity == "user_turn"
    assert p.name == "verbatim_user_turn"
    assert p.build(_example(), CostLedger())

    with pytest.raises(ValueError, match="unknown granularity"):
        build_policy("verbatim_paragraph")


def test_malformed_specs_name_the_spec_they_reject() -> None:
    """A typo in --policies used to surface as a bare IndexError.

    Every one of these raises ValueError mentioning the spec, because the
    caller is a command line and the message is the whole diagnostic.
    """
    for spec in ("leadk", "tailk", "truncated_recency", "truncated",
                 "leadk_half", "leadk_0", "leadk_250"):
        with pytest.raises(ValueError) as e:
            build_policy(spec)
        assert spec in str(e.value), f"{spec!r} not named in {e.value!r}"


def test_extractive_does_not_abandon_budget_on_one_bad_depth() -> None:
    """A depth level where nothing fits says nothing about the next one.

    Constructed so the second sentence of every turn is far too long for the
    remaining budget while the third is small. The old `break` stopped at
    depth 1 and left most of the budget unspent, which made realised store
    size depend on sentence-length ordering -- the exact thing the
    skip-don't-stop rule inside the loop exists to remove.
    """
    long_mid = "x" * 4000
    turns = [
        Turn(session_id=f"s{i}", session_date="2023/04/10 (Mon) 17:50",
             session_index=i, turn_index=i, role="user",
             content=f"Alpha fact {i}. {long_mid}. Gamma fact {i}.",
             has_answer=False)
        for i in range(6)
    ]
    ex = replace(_example(), turns=turns)

    units = ExtractiveSelectionPolicy(0.05, rule="lead").build(ex, CostLedger())
    budget = 0.05 * sum(count_tokens(u.text) for u in ex.units("turn"))
    used = sum(count_tokens(u.text) for u in units)

    assert used <= budget, "over budget"
    # Depth 0 alone is ~6 short sentences; reaching depth 2 is the whole point.
    assert any("Gamma" in u.text for u in units), \
        "stopped at the long sentence instead of skipping past it"
