"""Write policies: how a conversation becomes a store.

The read half of a memory system lives in `memllm.retrieval`; this is the write
half. See `memllm.write.base` for the Protocol and the two store contracts.

`build_policy` mirrors `scripts/run_retrieval_eval.build_retriever`: one place
that turns a name into an object, so adding a policy is one branch.
"""

from __future__ import annotations

from memllm.write.base import WritePolicy, check_store, renumber
from memllm.write.extractive import ExtractiveSelectionPolicy
from memllm.write.truncated import TruncatedVerbatimPolicy
from memllm.write.verbatim import VerbatimPolicy

__all__ = [
    "WritePolicy", "VerbatimPolicy", "TruncatedVerbatimPolicy",
    "ExtractiveSelectionPolicy", "build_policy", "check_store", "renumber",
]


def build_policy(spec: str) -> WritePolicy:
    """Parse a policy spec string.

        verbatim_turn
        truncated_recency_25            truncated_random_25_s1
        leadk_25
        mem0_oss_v3_<model>             (requires the optional mem0 extra)

    Percentages are integers, so `truncated_recency_5` is a 5% token budget.
    """
    parts = spec.split("_")

    if parts[0] == "verbatim":
        return VerbatimPolicy(granularity=parts[1] if len(parts) > 1 else "turn")

    if parts[0] == "truncated":
        rule = parts[1]
        fraction = int(parts[2]) / 100
        seed = int(parts[3][1:]) if len(parts) > 3 and parts[3].startswith("s") else 0
        return TruncatedVerbatimPolicy(fraction=fraction, rule=rule, seed=seed)

    if parts[0] == "leadk":
        return ExtractiveSelectionPolicy(fraction=int(parts[1]) / 100)

    if spec.startswith("mem0"):
        # Imported here, not at module scope, so the package stays importable
        # without the mem0 extra installed -- CI installs only the core deps.
        from memllm.write.mem0_adapter import Mem0OssPolicy

        return Mem0OssPolicy.from_spec(spec)

    raise ValueError(f"unknown write policy: {spec}")
