"""LongMemEval loading and memory-unit construction.

The dataset gives each question its own haystack of ~48 sessions (~490 turns,
~104K tokens; measured in results/token_stats_turn.json). Every turn carries a
`has_answer` flag, which we treat as a gold retrieval label -- this is what lets
us measure retrieval quality without an LLM judge.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Literal

Granularity = Literal["turn", "user_turn", "session"]

_DATE_RE = re.compile(r"(\d{4})/(\d{2})/(\d{2})")


def parse_date(date_str: str) -> tuple[int, int, int]:
    """'2023/04/10 (Mon) 17:50' -> (2023, 4, 10). Used for recency priors."""
    m = _DATE_RE.search(date_str)
    if not m:
        return (0, 0, 0)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)))


@dataclass
class Turn:
    role: str
    content: str
    session_id: str
    session_date: str
    session_index: int
    turn_index: int
    has_answer: bool


@dataclass
class MemoryUnit:
    """One retrievable item. `is_evidence` is the gold label.

    Also the store format for `llm_mem_eval.write` policies, so a store made by
    any write path -- verbatim turns, truncated turns, LLM-extracted facts --
    flows into every existing retriever and metric unchanged. Two contracts a
    policy must honour, both of which fail silently rather than loudly:

    - `unit_id` must be contiguous unique ints from 0. `Hit` is `(unit_id,
      score)`, `RandomRetriever` samples the list, and `score_example` builds an
      id set; duplicate ids merge evidence without an error.
    - `session_date` must parse under `parse_date`, or `RecencyRetriever` sorts
      every unit to `(0, 0, 0)` and the recency control silently becomes a no-op.

    `provenance` and `meta` are defaulted so no existing construction site
    changes: they carry source turn indices and policy-specific fields (a mem0
    memory id, its created_at) for stores that can supply them.
    """

    unit_id: int
    text: str
    session_id: str
    session_date: str
    session_index: int
    is_evidence: bool
    roles: tuple[str, ...] = ()
    provenance: tuple[int, ...] = ()
    meta: dict = field(default_factory=dict)

    @property
    def date_key(self) -> tuple[int, int, int]:
        return parse_date(self.session_date)


@dataclass
class Example:
    question_id: str
    question_type: str
    question: str
    answer: str
    question_date: str
    turns: list[Turn] = field(default_factory=list)

    @property
    def is_abstention(self) -> bool:
        """LongMemEval marks abstention questions with an `_abs` suffix."""
        return self.question_id.endswith("_abs")

    @property
    def n_evidence_turns(self) -> int:
        return sum(1 for t in self.turns if t.has_answer)

    def units(self, granularity: Granularity = "turn") -> list[MemoryUnit]:
        """Build retrievable units at the requested granularity.

        turn       -- every turn, user and assistant
        user_turn  -- user turns only (assistant turns are largely restatement)
        session    -- whole session concatenated, evidence if any turn matches
        """
        if granularity == "turn":
            return [
                MemoryUnit(
                    unit_id=i,
                    text=f"{t.role}: {t.content}",
                    session_id=t.session_id,
                    session_date=t.session_date,
                    session_index=t.session_index,
                    is_evidence=t.has_answer,
                    roles=(t.role,),
                )
                for i, t in enumerate(self.turns)
            ]

        if granularity == "user_turn":
            return [
                MemoryUnit(
                    unit_id=i,
                    text=t.content,
                    session_id=t.session_id,
                    session_date=t.session_date,
                    session_index=t.session_index,
                    is_evidence=t.has_answer,
                    roles=(t.role,),
                )
                for i, t in enumerate(x for x in self.turns if x.role == "user")
            ]

        if granularity == "session":
            by_session: dict[str, list[Turn]] = {}
            for t in self.turns:
                by_session.setdefault(t.session_id, []).append(t)
            units = []
            for i, (sid, turns) in enumerate(by_session.items()):
                units.append(
                    MemoryUnit(
                        unit_id=i,
                        text="\n".join(f"{t.role}: {t.content}" for t in turns),
                        session_id=sid,
                        session_date=turns[0].session_date,
                        session_index=turns[0].session_index,
                        is_evidence=any(t.has_answer for t in turns),
                        roles=tuple(t.role for t in turns),
                    )
                )
            return units

        raise ValueError(f"unknown granularity: {granularity}")


def load_examples(path: str | Path, limit: int | None = None) -> list[Example]:
    """Load a LongMemEval split (longmemeval_s / _m / _oracle).

    The splits are a deliberate non-vendored download, so a missing file is the
    likeliest failure on a clean clone -- and the first command anyone runs is
    one that needs it. A bare FileNotFoundError names a path and says nothing
    about where the file comes from, so raise the download instead.
    """
    p = Path(path)
    if not p.is_file():
        raise SystemExit(
            f"LongMemEval is not downloaded -- no file at {p}\n\n"
            f"Fetch it with:\n\n"
            f"    pip install huggingface_hub\n"
            f"    python -c \"from huggingface_hub import hf_hub_download; \\\n"
            f"      hf_hub_download('xiaowu0162/longmemeval', '{p.name}', \\\n"
            f"      repo_type='dataset', local_dir='{p.parent}')\"\n\n"
            f"longmemeval_s is ~278 MB; longmemeval_oracle is ~15 MB and is\n"
            f"enough for the grader audit. tests/ and scripts/make_report.py\n"
            f"need neither -- they run from the stored arms in results/."
        )
    with open(p) as f:
        raw = json.load(f)
    if limit is not None:
        raw = raw[:limit]

    examples = []
    for r in raw:
        turns: list[Turn] = []
        sessions = r["haystack_sessions"]
        dates = r["haystack_dates"]
        sids = r["haystack_session_ids"]
        for si, (session, date, sid) in enumerate(zip(sessions, dates, sids)):
            for turn in session:
                turns.append(
                    Turn(
                        role=turn["role"],
                        content=turn["content"],
                        session_id=sid,
                        session_date=date,
                        session_index=si,
                        turn_index=len(turns),
                        has_answer=bool(turn.get("has_answer", False)),
                    )
                )
        examples.append(
            Example(
                question_id=r["question_id"],
                question_type=r["question_type"],
                question=r["question"],
                answer=r["answer"],
                question_date=r["question_date"],
                turns=turns,
            )
        )
    return examples


def stratified_subset(
    examples: list[Example], n: int, seed: int = 0
) -> list[Example]:
    """Sample n examples preserving the question_type distribution.

    Weeks-scale compute means we subsample; doing it stratified keeps the
    knowledge-update and abstention slices intact, which are the interesting ones.
    """
    import random

    rng = random.Random(seed)
    by_type: dict[str, list[Example]] = {}
    for e in examples:
        by_type.setdefault(e.question_type, []).append(e)

    total = len(examples)
    picked: list[Example] = []
    for qtype, group in sorted(by_type.items()):
        rng.shuffle(group)
        k = max(1, round(n * len(group) / total))
        picked.extend(group[:k])

    rng.shuffle(picked)
    return picked[:n]


def iter_units(
    examples: list[Example], granularity: Granularity
) -> Iterator[tuple[Example, list[MemoryUnit]]]:
    for ex in examples:
        yield ex, ex.units(granularity)
