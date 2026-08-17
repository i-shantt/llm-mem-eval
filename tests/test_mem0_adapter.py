"""Tests for the Mem0 adapter that run without mem0 installed.

The adapter is committed but has not been run (see RUNNING.md). What can be
tested offline is the part most likely to be silently wrong when it *is* run:
the cost shim, the batching, and the fact that importing the package does not
require the optional dependency.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_mem_eval.cost import CostLedger  # noqa: E402
from llm_mem_eval.data.loader import Example, Turn  # noqa: E402
from llm_mem_eval.write.mem0_adapter import (  # noqa: E402
    CountingChatClient,
    Mem0OssPolicy,
    PreflightResult,
    _iso_date,
)


class _StubCompletions:
    """Stands in for openai's chat.completions, with and without usage."""

    def __init__(self, report_usage: bool = True):
        self.report_usage = report_usage
        self.calls = 0

    def create(self, **kw):
        self.calls += 1
        msg = SimpleNamespace(content='{"facts": ["a"]}')
        resp = SimpleNamespace(choices=[SimpleNamespace(message=msg)])
        resp.usage = (SimpleNamespace(prompt_tokens=8000, completion_tokens=120)
                      if self.report_usage else None)
        return resp


def test_counting_client_bills_the_write_phase() -> None:
    led = CostLedger()
    client = CountingChatClient(_StubCompletions(), led)
    client.create(messages=[{"role": "user", "content": "hello"}])
    client.create(messages=[{"role": "user", "content": "hello"}])

    assert led.write.llm_calls == 2
    assert led.write.llm_prompt_tokens == 16000
    assert led.write.llm_completion_tokens == 240
    assert led.read.llm_calls == 0, "ingestion must never bill the read path"
    assert client.n_calls == 2
    assert not client.usage_estimated


def test_counting_client_flags_estimation_when_usage_is_absent() -> None:
    """A server that reports no usage makes every write number an estimate."""
    led = CostLedger()
    client = CountingChatClient(_StubCompletions(report_usage=False), led)
    client.create(messages=[{"role": "user", "content": "some prompt text"}])

    assert client.usage_estimated is True
    assert led.write.llm_calls == 1
    assert led.write.llm_prompt_tokens > 0, "fell back to counting the prompt"


def test_batching_granularities_partition_the_conversation() -> None:
    turns = [Turn("user", f"m{i}", f"s{i // 4}", "2023/01/01 (Sun) 10:00",
                  i // 4, i, False) for i in range(12)]
    ex = Example("q1", "single-session-user", "?", "a",
                 "2023/02/01 (Wed) 10:00", turns)

    for batch, expected in [("session", 3), ("pair", 6), ("turn", 12)]:
        p = Mem0OssPolicy(llm_model="stub", batch=batch)
        got = p._batches(ex)
        assert len(got) == expected, f"{batch}: {len(got)} batches"
        # Every turn carried exactly once, whatever the granularity.
        flat = [t.turn_index for b in got for t in b]
        assert sorted(flat) == list(range(12))


def test_pair_batching_never_straddles_a_session() -> None:
    """build() dates a batch from its first turn, so a pair that crosses a
    session boundary is stored under the earlier session's date -- the
    conversation-date contamination the preflight check exists to catch.
    8.1% of LongMemEval-S sessions have an odd turn count, so this is the
    common case, not a corner one.
    """
    # Session s0 has three turns, so a flat pairing would put its last turn
    # in the same add() call as s1's first.
    sizes = [3, 4, 3]
    turns, idx = [], 0
    for si, n in enumerate(sizes):
        for _ in range(n):
            turns.append(Turn("user", f"m{idx}", f"s{si}",
                              f"2023/0{si + 1}/01 (Sun) 10:00", si, idx, False))
            idx += 1
    ex = Example("q1", "single-session-user", "?", "a",
                 "2023/06/01 (Thu) 10:00", turns)

    got = Mem0OssPolicy(llm_model="stub", batch="pair")._batches(ex)
    for b in got:
        assert len({t.session_id for t in b}) == 1, \
            f"batch spans sessions: {[t.session_id for t in b]}"
    assert sorted(t.turn_index for b in got for t in b) == list(range(idx))
    # ceil(3/2) + ceil(4/2) + ceil(3/2) = 2 + 2 + 2, against 5 when flat.
    assert len(got) == 6


def test_estimated_write_tokens_reach_the_manifest() -> None:
    """The flag used to be appended to a `ledger.notes` list that CostLedger
    does not have, so an arm whose write tokens were counted locally rather
    than reported by the server was written to disk with nothing saying so.
    """
    p = Mem0OssPolicy(llm_model="stub")
    assert p.config()["write_tokens_estimated"] is False
    p.usage_estimated = True
    assert p.config()["write_tokens_estimated"] is True


def test_store_is_named_for_the_configuration_not_just_mem0() -> None:
    p = Mem0OssPolicy(llm_model="Qwen/Qwen2.5-7B-Instruct")
    assert p.name == "mem0_oss_v3_qwen2.5-7b-instruct"
    assert p.name != "mem0"
    cfg = p.config()
    assert cfg["mem0_version_pinned"] == "2.0.18"
    assert "managed platform" in cfg["not_the_managed_platform"]


def test_iso_date_conversion() -> None:
    assert _iso_date("2023/04/10 (Mon) 17:50") == "2023-04-10"
    assert _iso_date("2023/12/01 (Fri) 09:00") == "2023-12-01"


def test_preflight_result_reports_every_failure() -> None:
    n_checks = len(PreflightResult.__dataclass_fields__)

    ok = PreflightResult(*([True] * n_checks))
    assert ok.ok and ok.explain() == "preflight OK"

    bad = PreflightResult(*([False] * n_checks))
    assert not bad.ok
    text = bad.explain()
    for expected in ("keyword_search", "fastembed", "sparse slot",
                     "en_core_web_sm", "created_at", "current_date"):
        assert expected in text, f"{expected} not explained"

    # Every check must contribute a line, or a silently-failing one looks fine.
    # Counted rather than listed, so adding a field without an explanation
    # fails here instead of going unreported in a real run.
    assert text.count("\n  - ") == n_checks, (
        f"{n_checks} checks but {text.count(chr(10) + '  - ')} explanations")

    # And each one alone must be enough to fail the preflight.
    for field in PreflightResult.__dataclass_fields__:
        one_bad = PreflightResult(*([True] * n_checks))
        setattr(one_bad, field, False)
        assert not one_bad.ok, f"{field} False still reports ok"


def test_package_imports_without_the_mem0_extra() -> None:
    """CI installs core deps only; `llm_mem_eval.write` must import anyway."""
    import llm_mem_eval.write as w

    assert w.build_policy("verbatim_turn").name == "verbatim_turn"
    if "mem0" not in sys.modules:
        with pytest.raises((ImportError, SystemExit)):
            w.build_policy("mem0_oss_v3_qwen7b")


def test_the_date_patch_pins_both_prompt_dates_not_just_one() -> None:
    """mem0's extraction prompt carries two dates and defaults both to now.

    `generate_additive_extraction_prompt(current_date=, timestamp=)` feeds
    `_resolve_dates`, which fills "Current Date" from `current_date` and
    "Observation Date" from `timestamp`, each defaulting to `datetime.now()`.
    Binding only `timestamp` -- the original bug here -- produced a prompt
    reading "Observation Date 2023-05-20 / Current Date <today>", leaving
    wall-clock today in the prompt as the anchor for "last week". Both must be
    the conversation's date.

    Checked by inspecting the partial the adapter installs, so this runs with
    mem0 absent, which is the only state CI has.
    """
    import functools
    import sys
    from types import ModuleType

    captured = {}

    def fake_prompt(**kw):
        captured.update(kw)
        return "prompt"

    # Stand in for mem0.memory.main just long enough for _add_with_date to
    # patch it. Nothing else about mem0 is needed.
    mod = ModuleType("mem0.memory.main")
    mod.generate_additive_extraction_prompt = fake_prompt
    pkg = ModuleType("mem0")
    mem = ModuleType("mem0.memory")
    saved = {k: sys.modules.get(k) for k in ("mem0", "mem0.memory", "mem0.memory.main")}
    sys.modules.update({"mem0": pkg, "mem0.memory": mem, "mem0.memory.main": mod})

    installed = {}

    class _Memory:
        def add(self, messages, **kw):
            # Whatever the adapter patched in is live at this point.
            fn = sys.modules["mem0.memory.main"].generate_additive_extraction_prompt
            installed["keywords"] = dict(getattr(fn, "keywords", {}) or {})
            installed["metadata"] = kw.get("metadata")
            return {"results": []}

    try:
        Mem0OssPolicy._add_with_date(
            _Memory(), [{"role": "user", "content": "hi"}],
            user_id="u", conversation_date="2023-05-20")
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    kws = installed["keywords"]
    assert kws.get("timestamp") == "2023-05-20", kws
    assert kws.get("current_date") == "2023-05-20", (
        "only the observation date was pinned; the prompt would still carry "
        f"wall-clock today as its Current Date. bound: {kws}")
    # And the stored row's created_at still has to be overridden too.
    assert installed["metadata"]["created_at"].startswith("2023-05-20")
    # The patch must be undone, or every later call inherits this date.
    assert not isinstance(
        sys.modules.get("mem0.memory.main", ModuleType("x")).__dict__.get(
            "generate_additive_extraction_prompt"), functools.partial)


def test_build_reads_the_whole_store_not_mem0s_default_twenty() -> None:
    """`Memory.get_all(top_k=...)` defaults to 20.

    A LongMemEval conversation is ~490 turns, so v3 emits far more than 20
    memories; reading back the default would have measured survival over a
    fraction of the store, and in the direction that makes extraction look
    catastrophically lossy. `build()` must pass an explicit top_k and must fail
    loudly if it saturates.

    Driven through a fake Memory, so it runs with mem0 absent.
    """
    from llm_mem_eval.write.mem0_adapter import GET_ALL_TOP_K

    assert GET_ALL_TOP_K > 20

    seen = {}

    class _FakeMemory:
        def __init__(self, n_rows):
            self.n_rows = n_rows
            self.llm = SimpleNamespace(
                client=SimpleNamespace(chat=SimpleNamespace(
                    completions=_StubCompletions())))

        def add(self, messages, **kw):
            return {"results": [{"id": "m1"}]}

        def get_all(self, **kw):
            seen.update(kw)
            return {"results": [{"id": f"m{i}", "memory": f"fact {i}",
                                 "created_at": "2023-04-10T00:00:00+00:00"}
                                for i in range(self.n_rows)]}

    ex = Example(question_id="q1", question_type="t", question="?", answer="a",
                 question_date="2023/04/11 (Tue) 10:00",
                 turns=[Turn(role="user", content="hi", session_id="s1",
                             session_date="2023/04/10 (Mon) 17:50",
                             session_index=0, turn_index=0, has_answer=False)])

    policy = Mem0OssPolicy(llm_model="stub")
    policy._new_memory = lambda collection, path: _FakeMemory(25)
    policy._add_with_date = lambda m, msgs, **kw: m.add(msgs)

    units = policy.build(ex, CostLedger())
    assert seen.get("top_k") == GET_ALL_TOP_K, (
        f"build() must ask for the whole store; it passed {seen.get('top_k')!r}")
    # 25 rows come back, so all 25 become units -- not mem0's default 20.
    assert len(units) == 25

    # Saturation is an error, not a silent truncation.
    policy._new_memory = lambda collection, path: _FakeMemory(GET_ALL_TOP_K)
    with pytest.raises(RuntimeError, match="truncated on read"):
        policy.build(ex, CostLedger())
