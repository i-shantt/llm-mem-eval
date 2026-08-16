"""Model Mem0 v3's write-path cost as a function of caller batching.

    python scripts/model_write_cost.py            # writes results/write_cost_model.json

Mem0's v3 ingestion is ADD-only and makes **one LLM call per `add()`**, whatever
number of messages that call carries (`mem0/memory/main.py` at tag v2.0.18,
"=== V3 PHASED BATCH PIPELINE ===", one `generate_response` at line 956). The
paper's second, per-fact update call is gone.

That removes the write path's dependence on the number of extracted facts and
replaces it with a dependence on something the *caller* chooses: how many
messages to hand each `add()`. An agent that calls `add()` after every turn and a
batch job that calls it once per session run the same code and do not pay the
same amount.

Three terms, and only one of them is granularity-invariant:

  system prompt   ADDITIVE_EXTRACTION_PROMPT, sent in full on every call
                  (main.py:942, 958). Measured, not assumed: see SYSTEM_PROMPT_TOKENS.
  resent context  each call re-sends the last 10 messages of the session
                  (`get_last_messages(..., limit=10)`, main.py:919) plus the top
                  10 existing memories (`top_k=10`, main.py:925). More calls
                  means the same text is paid for more times.
  new content     the messages themselves. Every turn is carried by exactly one
                  call at any granularity, so this term is constant.

So the write cost is dominated by the two terms that scale with the *number of
calls*, which is the caller's decision. This script measures that span on
LongMemEval's actual message counts.

Estimates are labelled. Exact: call counts, the system prompt size, the content
and resent-context token counts. Assumed: the length of a stored memory, the
completion length, and the user-prompt template overhead -- all three are
reported with a sensitivity range because none can be measured without running
the extractor.

Prompt caching is priced alongside the uncached figure. The system prompt is an
identical 7.6K-token prefix on every call, which is exactly what prefix caching
is for, and omitting it would overstate the cost of the high-call-count regimes.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memllm.cost import count_tokens  # noqa: E402
from memllm.data.loader import Example, load_examples  # noqa: E402

# Measured from mem0/configs/prompts.py at tag v2.0.18 with tiktoken cl100k_base.
# Pinned as a constant rather than fetched so this script runs offline and the
# number is reviewable; scripts/verify_write_cost_inputs.py re-derives it.
SYSTEM_PROMPT_TOKENS = 7671
MEM0_VERSION = "2.0.18"

# main.py:919 and :925 -- both are hardcoded in v3, not configurable.
LAST_K_MESSAGES = 10
EXISTING_MEMORIES_TOP_K = 10

# --- assumptions, each with a sensitivity range ---------------------------
# A mem0 v3 memory is one self-contained factual sentence.
MEM_TOKENS_EACH = 30
MEM_TOKENS_RANGE = (15, 60)
# Extracted-facts JSON emitted per call. Capped by max_tokens (default 2000).
COMPLETION_TOKENS_PER_CALL = 300
COMPLETION_TOKENS_RANGE = (100, 800)
# Headings and delimiters that generate_additive_extraction_prompt adds.
TEMPLATE_OVERHEAD_TOKENS = 200

# gpt-5-mini is mem0's OSS default (mem0/llms/openai.py). Prices are USD per 1M
# tokens and are the one input here that this repo cannot verify from a primary
# source, so both a cached and an uncached figure are shown and the ratio -- not
# the absolute dollar amount -- is what the analysis rests on.
PRICE_IN_PER_M = 0.25
PRICE_CACHED_IN_PER_M = 0.025
PRICE_OUT_PER_M = 2.00

# Read-path tokens per query, for the break-even. Ours is measured
# (results/e2e_7b_hybrid_turn_k10_n100.json); Mem0's is its own reported LoCoMo
# Table 2 "memory tokens" figure. Different benchmarks -- the break-even shows
# the shape of the amortisation, not a head-to-head result.
READ_TOKENS_OURS = 2097
READ_TOKENS_MEM0 = 1764


def batches(ex: Example, granularity: str) -> list[list[int]]:
    """Indices of the turns carried by each add() call."""
    n = len(ex.turns)
    if granularity == "per_turn":
        return [[i] for i in range(n)]
    if granularity == "per_pair":
        return [list(range(i, min(i + 2, n))) for i in range(0, n, 2)]
    if granularity == "per_session":
        out: dict[str, list[int]] = {}
        for i, t in enumerate(ex.turns):
            out.setdefault(t.session_id, []).append(i)
        return list(out.values())
    if granularity == "per_conversation":
        return [list(range(n))]
    raise ValueError(f"unknown granularity: {granularity}")


def model_example(ex: Example, granularity: str, mem_tokens: int,
                  completion_tokens: int) -> dict:
    tok = [count_tokens(f"{t.role}: {t.content}") for t in ex.turns]
    calls = batches(ex, granularity)

    resent = 0
    for b in calls:
        # The 10 messages immediately preceding this batch, which v3 pulls from
        # its own history table and re-sends on every call.
        start = b[0]
        resent += sum(tok[max(0, start - LAST_K_MESSAGES):start])

    n_calls = len(calls)
    fixed = n_calls * (SYSTEM_PROMPT_TOKENS + TEMPLATE_OVERHEAD_TOKENS
                       + EXISTING_MEMORIES_TOP_K * mem_tokens)
    content = sum(tok)  # granularity-invariant: every turn is carried once
    prompt_tokens = fixed + resent + content
    return {
        "n_calls": n_calls,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": n_calls * completion_tokens,
        "system_prompt_tokens": n_calls * SYSTEM_PROMPT_TOKENS,
        "resent_context_tokens": resent,
        "new_content_tokens": content,
    }


def price(prompt_tokens: int, completion_tokens: int, n_calls: int,
          cached: bool) -> float:
    """USD. Under caching the system prefix is billed once at full rate."""
    if not cached:
        return (prompt_tokens * PRICE_IN_PER_M
                + completion_tokens * PRICE_OUT_PER_M) / 1e6
    cacheable = max(0, n_calls - 1) * SYSTEM_PROMPT_TOKENS
    full = prompt_tokens - cacheable
    return (full * PRICE_IN_PER_M + cacheable * PRICE_CACHED_IN_PER_M
            + completion_tokens * PRICE_OUT_PER_M) / 1e6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/longmemeval_s")
    ap.add_argument("--limit", type=int, default=0, help="0 = all questions")
    ap.add_argument("--out", default="results/write_cost_model.json")
    args = ap.parse_args()

    examples = load_examples(args.data)
    if args.limit:
        examples = examples[: args.limit]

    grans = ["per_turn", "per_pair", "per_session", "per_conversation"]
    summary = {}
    for g in grans:
        rows = [model_example(ex, g, MEM_TOKENS_EACH, COMPLETION_TOKENS_PER_CALL)
                for ex in examples]
        mean = {k: statistics.mean(r[k] for r in rows) for k in rows[0]}
        total_tok = mean["prompt_tokens"] + mean["completion_tokens"]
        summary[g] = {
            **mean,
            "total_tokens": total_tok,
            "usd_uncached": price(mean["prompt_tokens"],
                                  mean["completion_tokens"],
                                  mean["n_calls"], cached=False),
            "usd_cached": price(mean["prompt_tokens"],
                                mean["completion_tokens"],
                                mean["n_calls"], cached=True),
            "share_system_prompt": mean["system_prompt_tokens"] / total_tok,
            "share_resent_context": mean["resent_context_tokens"] / total_tok,
            "share_new_content": mean["new_content_tokens"] / total_tok,
        }

    # Sensitivity on the two assumed quantities, at per-session granularity.
    sens = {}
    for label, mt, ct in [
        ("low", MEM_TOKENS_RANGE[0], COMPLETION_TOKENS_RANGE[0]),
        ("mid", MEM_TOKENS_EACH, COMPLETION_TOKENS_PER_CALL),
        ("high", MEM_TOKENS_RANGE[1], COMPLETION_TOKENS_RANGE[1]),
    ]:
        rows = [model_example(ex, "per_session", mt, ct) for ex in examples]
        sens[label] = {
            "mem_tokens_each": mt,
            "completion_tokens_per_call": ct,
            "total_tokens": statistics.mean(
                r["prompt_tokens"] + r["completion_tokens"] for r in rows),
        }

    # Break-even against this repo's measured read path. Expressed in tokens as
    # well as dollars because the token figure is price-independent, and the
    # prices here are the one input with no primary source.
    read_saving = READ_TOKENS_OURS - READ_TOKENS_MEM0
    for g in grans:
        s = summary[g]
        s["break_even_queries_tokens"] = s["total_tokens"] / read_saving

    span = summary["per_turn"]["total_tokens"] / summary["per_session"]["total_tokens"]
    payload = {
        "mem0_version": MEM0_VERSION,
        "source": "mem0/memory/main.py and mem0/configs/prompts.py at tag v2.0.18",
        "benchmark": "LongMemEval-S",
        "n_conversations": len(examples),
        "_session_count_note": (
            "Session granularity averages ~48 calls, not the ~50 you get from "
            "len(haystack_sessions). The split contains 1,230 empty sessions "
            "and 15 conversations reuse a session id; an empty session triggers "
            "no add() call, so 48 is the right figure for a call count."
        ),
        "_break_even_note": (
            "break_even_queries_tokens = write tokens / (this repo's read "
            "tokens per query - Mem0's reported read tokens per query) = "
            f"total / ({READ_TOKENS_OURS} - {READ_TOKENS_MEM0}). It ignores the "
            "8x price premium on output tokens, so it is a floor. The point is "
            "not the number: it is that under v3 there is no single break-even, "
            "because the write cost is set by the caller's batching."
        ),
        "exact_inputs": {
            "system_prompt_tokens": SYSTEM_PROMPT_TOKENS,
            "last_k_messages": LAST_K_MESSAGES,
            "existing_memories_top_k": EXISTING_MEMORIES_TOP_K,
            "llm_calls_per_add": 1,
            "note": "v3 is ADD-only; the paper's per-fact update call is gone, "
                    "so calls no longer scale with the number of extracted facts.",
        },
        "assumed_inputs": {
            "mem_tokens_each": MEM_TOKENS_EACH,
            "mem_tokens_range": list(MEM_TOKENS_RANGE),
            "completion_tokens_per_call": COMPLETION_TOKENS_PER_CALL,
            "completion_tokens_range": list(COMPLETION_TOKENS_RANGE),
            "template_overhead_tokens": TEMPLATE_OVERHEAD_TOKENS,
            "note": "None of these can be measured without running the "
                    "extractor. See `sensitivity`: they move the total by far "
                    "less than the batching choice does.",
        },
        "prices_usd_per_1m": {
            "input": PRICE_IN_PER_M,
            "cached_input": PRICE_CACHED_IN_PER_M,
            "output": PRICE_OUT_PER_M,
            "note": "gpt-5-mini list rates, mem0's OSS default model. Not "
                    "verifiable from a primary source in this repo, so the "
                    "analysis rests on the ratio between granularities, which "
                    "is price-independent.",
        },
        "per_conversation": summary,
        "sensitivity_at_per_session": sens,
        "headline": {
            "call_count_span": (f"{summary['per_session']['n_calls']:.0f} calls "
                                f"per conversation at session granularity vs "
                                f"{summary['per_turn']['n_calls']:.0f} at turn "
                                f"granularity"),
            "token_span_ratio": span,
            "reading": (
                "v3's write cost is set by how the caller batches, not by the "
                "conversation. The same library, same model and same "
                "conversation span roughly an order of magnitude depending on "
                "whether add() is called per turn or per session, because the "
                "7.7K-token system prompt and the 10-message resent window are "
                "paid once per call."
            ),
        },
    }

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))

    print(f"Mem0 v3 (mem0ai {MEM0_VERSION}) write cost per LongMemEval "
          f"conversation, n={len(examples)}\n")
    hdr = (f"{'batching':<18}{'calls':>7}{'tokens':>12}{'sys%':>7}{'resent%':>9}"
           f"{'new%':>7}{'$ uncached':>12}{'$ cached':>10}{'break-even':>12}")
    print(hdr)
    print("-" * len(hdr))
    for g in grans:
        s = summary[g]
        print(f"{g:<18}{s['n_calls']:>7.0f}{s['total_tokens']:>12,.0f}"
              f"{s['share_system_prompt']:>7.0%}{s['share_resent_context']:>9.0%}"
              f"{s['share_new_content']:>7.0%}"
              f"{s['usd_uncached']:>12.3f}{s['usd_cached']:>10.3f}"
              f"{s['break_even_queries_tokens']:>12,.0f}")
    print(f"\nspan, per_turn / per_session: {span:.1f}x tokens")
    print("sensitivity at per_session (assumed inputs low/mid/high): "
          + ", ".join(f"{k} {v['total_tokens']:,.0f}" for k, v in sens.items()))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
