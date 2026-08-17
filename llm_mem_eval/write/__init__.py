"""Write policies: how a conversation becomes a store.

The read half of a memory system lives in `llm_mem_eval.retrieval`; this is the
write half. See `llm_mem_eval.write.base` for the Protocol and the two store
contracts.

`build_policy` mirrors `scripts/run_retrieval_eval.build_retriever`: one place
that turns a name into an object, so adding a policy is one branch.
"""

from __future__ import annotations

from llm_mem_eval.write.base import WritePolicy, check_store, renumber
from llm_mem_eval.write.centrality import CentralitySelectionPolicy
from llm_mem_eval.write.extractive import ExtractiveSelectionPolicy
from llm_mem_eval.write.truncated import TruncatedVerbatimPolicy
from llm_mem_eval.write.verbatim import VerbatimPolicy

__all__ = [
    "WritePolicy", "VerbatimPolicy", "TruncatedVerbatimPolicy",
    "ExtractiveSelectionPolicy", "CentralitySelectionPolicy", "build_policy",
    "check_store", "renumber",
]


GRANULARITIES = ("turn", "user_turn", "session")


def _percent(spec: str, parts: list[str], i: int) -> float:
    """The `25` in `leadk_25`, as a fraction.

    Split out so a malformed spec raises a ValueError naming the spec, rather
    than the bare IndexError or ValueError that indexing and `int()` would
    raise on their own. A policy spec usually arrives from a `--policies`
    command line, where the failure mode being guarded against is a typo.
    """
    if len(parts) <= i:
        raise ValueError(f"write policy {spec!r} is missing its percentage")
    try:
        pct = int(parts[i])
    except ValueError:
        raise ValueError(
            f"write policy {spec!r}: expected an integer percentage, "
            f"got {parts[i]!r}"
        ) from None
    if not 0 < pct <= 100:
        raise ValueError(
            f"write policy {spec!r}: percentage must be in (0, 100], got {pct}"
        )
    return pct / 100


def build_policy(spec: str) -> WritePolicy:
    """Parse a policy spec string.

        verbatim_turn                   verbatim_user_turn
        truncated_recency_25            truncated_random_25_s1
        leadk_25                        tailk_25
        lexrank_25
        mem0_oss_v3_<model>             (requires the optional mem0 extra)

    Percentages are integers, so `truncated_recency_5` is a 5% token budget.
    """
    parts = spec.split("_")

    if parts[0] == "verbatim":
        # Joined, not `parts[1]`: `user_turn` is a granularity, and taking only
        # the first token silently built a `verbatim_user` policy that failed
        # later, inside build(), with an unrelated-looking error.
        granularity = "_".join(parts[1:]) or "turn"
        if granularity not in GRANULARITIES:
            raise ValueError(
                f"write policy {spec!r}: unknown granularity "
                f"{granularity!r}, expected one of {GRANULARITIES}"
            )
        return VerbatimPolicy(granularity=granularity)

    if parts[0] == "truncated":
        if len(parts) < 2:
            raise ValueError(f"write policy {spec!r} is missing its rule")
        rule = parts[1]
        fraction = _percent(spec, parts, 2)
        seed = int(parts[3][1:]) if len(parts) > 3 and parts[3].startswith("s") else 0
        return TruncatedVerbatimPolicy(fraction=fraction, rule=rule, seed=seed)

    if parts[0] in ("leadk", "tailk"):
        return ExtractiveSelectionPolicy(
            fraction=_percent(spec, parts, 1),
            rule="lead" if parts[0] == "leadk" else "tail",
        )

    if parts[0] == "lexrank":
        return CentralitySelectionPolicy(fraction=_percent(spec, parts, 1))

    if spec.startswith("mem0"):
        # Imported here, not at module scope, so the package stays importable
        # without the mem0 extra installed -- CI installs only the core deps.
        from llm_mem_eval.write.mem0_adapter import Mem0OssPolicy

        return Mem0OssPolicy.from_spec(spec)

    raise ValueError(f"unknown write policy: {spec}")
