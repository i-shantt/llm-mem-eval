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

from memllm.cost import CostLedger  # noqa: E402
from memllm.data.loader import Example, Turn  # noqa: E402
from memllm.write.mem0_adapter import (  # noqa: E402
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
    ok = PreflightResult(True, True, True, True, True)
    assert ok.ok and ok.explain() == "preflight OK"

    bad = PreflightResult(False, False, False, False, False)
    assert not bad.ok
    text = bad.explain()
    for expected in ("keyword_search", "fastembed", "sparse slot",
                     "en_core_web_sm", "created_at"):
        assert expected in text, f"{expected} not explained"


def test_package_imports_without_the_mem0_extra() -> None:
    """CI installs core deps only; `from memllm.write import ...` must work."""
    import memllm.write as w

    assert w.build_policy("verbatim_turn").name == "verbatim_turn"
    if "mem0" not in sys.modules:
        with pytest.raises((ImportError, SystemExit)):
            w.build_policy("mem0_oss_v3_qwen7b")
