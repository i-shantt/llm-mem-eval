"""Write policies: the other half of a memory system.

A retriever answers "given a store, what comes back?". A write policy answers
the question before it: "given a conversation, what goes into the store?".
Published memory systems differ far more in the second than the first, and it is
the half that costs money, so the repo needs a name for it.

The Protocol deliberately mirrors `memllm.retrieval.base.Retriever`: a `name`, a
`config()` that goes verbatim into the artifact manifest, and one method that
does the work and bills a `CostLedger`. A policy returns `list[MemoryUnit]`, the
same type `Example.units()` returns, so every existing retriever and every
existing metric works over a policy's output with no changes at all.

Policies bill the **write** phase. Nothing here should touch `ledger.read`.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Protocol, runtime_checkable

from memllm.cost import CostLedger
from memllm.data.loader import Example, MemoryUnit, parse_date


@runtime_checkable
class WritePolicy(Protocol):
    name: str

    def config(self) -> dict:
        """Everything needed to reproduce this store. Goes into the manifest."""
        ...

    def build(self, ex: Example, ledger: CostLedger) -> list[MemoryUnit]:
        ...


def renumber(units: list[MemoryUnit]) -> list[MemoryUnit]:
    """Reassign `unit_id` to 0..n-1, preserving order.

    Any policy that filters or reorders must end with this. A store whose ids
    have gaps still *works* -- it just quietly mis-scores, because retrievers
    return ids and `score_example` intersects them against a set built from the
    same list.
    """
    return [replace(u, unit_id=i) for i, u in enumerate(units)]


def check_store(units: list[MemoryUnit]) -> None:
    """Raise if a store breaks a contract that would otherwise fail silently.

    Called by build_store.py on every policy's output. Cheap, and it converts
    two silent-corruption bugs into a stack trace.
    """
    ids = [u.unit_id for u in units]
    if ids != list(range(len(units))):
        raise ValueError(
            f"unit_id must be contiguous from 0; got {ids[:5]}... "
            f"(n={len(units)}). Call renumber() at the end of build()."
        )
    undated = [u.unit_id for u in units if parse_date(u.session_date) == (0, 0, 0)]
    if undated:
        raise ValueError(
            f"{len(undated)} units have an unparseable session_date "
            f"(e.g. unit {undated[0]}). RecencyRetriever would sort the whole "
            f"store to (0,0,0) and the recency control would silently become a "
            f"no-op. session_date must match {parse_date.__doc__.splitlines()[0]}"
        )
