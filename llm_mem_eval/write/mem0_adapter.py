"""Run Mem0's own open-source ingestion as a write policy.

STATUS: written, tested against a stub, and **not run**. No store produced by
this module exists in results/. See RUNNING.md for why, and for what it would
take. Nothing in the README depends on it.

The point of running Mem0's code rather than reimplementing its prompt is that
"I ran your pipeline" is a claim that survives review and "I ran something like
your prompt" is not. What this module does is thin by design: build `Memory`
from a config, wrap its LLM client so calls land on a `CostLedger`, feed it the
conversation in batches, and read the resulting memories back out as
`MemoryUnit`s so every existing retriever and metric works over them unchanged.

Everything below was read from `mem0ai` at git tag **v2.0.18**, which is the v3
ADD-only pipeline (`main.py` imports `ADDITIVE_EXTRACTION_PROMPT`, does not
import `get_update_memory_messages`, and hardcodes `"event": "ADD"`). Pin the
version. The algorithm changed materially between the 2025 paper and this tag,
and a store built from an unpinned install cannot be attributed to either.

Naming, deliberately: a store from this policy is `mem0_oss_v3_<model>` and
never `mem0`. Mem0's published LongMemEval score is for their **managed
platform**, which their own benchmark README says carries optimisations absent
from the open-source SDK, and this runs an open extraction model rather than
their `gpt-5-mini` default. Two steps removed, named as such in the success case
as much as the failure case.
"""

from __future__ import annotations

import functools
import os
import tempfile
from dataclasses import dataclass

from llm_mem_eval.cost import CostLedger, count_tokens
from llm_mem_eval.data.loader import Example, MemoryUnit, Turn
from llm_mem_eval.write.base import renumber

MEM0_PINNED_VERSION = "2.0.18"

# Batching is the caller's choice and it dominates the write cost -- see
# scripts/model_write_cost.py, which measures an 8.8x span between per-turn and
# per-session on this benchmark. Whatever is used must be recorded in the
# manifest as an explicit deviation, because Mem0 does not prescribe one.
DEFAULT_BATCH = "session"


class CountingChatClient:
    """Wraps an OpenAI-compatible completions client and bills every call.

    Deliberately holds no reference to mem0: it is a plain wrapper around the
    object at `memory.llm.client.chat.completions`, so it can be unit-tested
    against a stub with mem0 uninstalled. Wrapping the instance also survives
    mem0 refactoring its LLM classes, which subclassing would not.
    """

    def __init__(self, inner, ledger: CostLedger, phase: str = "write"):
        self.inner = inner
        self.ledger = ledger
        self.phase = phase
        self.n_calls = 0
        self.usage_estimated = False

    def create(self, **kw):
        resp = self.inner.create(**kw)
        self.n_calls += 1
        usage = getattr(resp, "usage", None)
        if usage is not None:
            self.ledger.add_llm(self.phase, usage.prompt_tokens,
                                usage.completion_tokens)
        else:
            # A server that does not report usage means every write-cost number
            # in this arm is an estimate. Flagged rather than silently mixed in,
            # the way the embedding cache already flags replayed timings.
            self.usage_estimated = True
            prompt = sum(count_tokens(str(m.get("content", "")))
                         for m in kw.get("messages", []))
            text = ""
            try:
                text = resp.choices[0].message.content or ""
            except Exception:
                pass
            self.ledger.add_llm(self.phase, prompt, count_tokens(text))
        return resp


@dataclass
class PreflightResult:
    bm25_implemented: bool
    bm25_slot_present: bool
    bm25_encoder_available: bool
    spacy_available: bool
    conversation_date_honoured: bool

    @property
    def ok(self) -> bool:
        return all(vars(self).values())

    def explain(self) -> str:
        problems = []
        if not self.bm25_implemented:
            problems.append(
                "vector store does not implement keyword_search -- BM25 is "
                "dropped for the whole session. chroma and faiss do this; "
                "qdrant is the only no-server store that does not.")
        if not self.bm25_encoder_available:
            problems.append(
                "fastembed missing -- _get_bm25_encoder() returns None and BM25 "
                "is dropped even on qdrant. pip install fastembed.")
        if not self.bm25_slot_present:
            problems.append(
                "collection has no bm25 sparse slot -- it predates v3 hybrid "
                "search. Delete the collection and rebuild.")
        if not self.spacy_available:
            problems.append(
                "spaCy en_core_web_sm missing -- extract_entities() returns [] "
                "and the entity boost is silently off.")
        if not self.conversation_date_honoured:
            problems.append(
                "stored created_at does not match the conversation date -- the "
                "extractor is resolving 'last week' against today. This "
                "contaminates every temporal memory in the run.")
        return ("preflight OK" if not problems
                else "preflight FAILED:\n  - " + "\n  - ".join(problems))


class Mem0OssPolicy:
    """Mem0 OSS v3 ingestion, driven by a locally served open model.

    All four degradation modes checked by `preflight()` are silent in mem0: the
    run completes and produces a store either way, just a worse one. Running
    without preflight is how you publish a number for a configuration you did
    not have.
    """

    def __init__(self, llm_model: str, llm_base_url: str = "http://localhost:8000/v1",
                 embed_model: str = "BAAI/bge-small-en-v1.5",
                 embed_dims: int = 384, batch: str = DEFAULT_BATCH,
                 workdir: str | None = None):
        self.llm_model = llm_model
        self.llm_base_url = llm_base_url
        self.embed_model = embed_model
        self.embed_dims = embed_dims
        self.batch = batch
        self.workdir = workdir
        # Set by build() when a server returns no usage block and the write
        # tokens have to be counted locally instead. Surfaced in config().
        self.usage_estimated = False
        short = llm_model.rstrip("/").split("/")[-1].replace(":", "-").lower()
        self.name = f"mem0_oss_v3_{short}"

    @classmethod
    def from_spec(cls, spec: str) -> "Mem0OssPolicy":
        """`mem0_oss_v3_<model-shorthand>`, resolved via MEM0_LLM_MODEL."""
        model = os.environ.get("MEM0_LLM_MODEL")
        if not model:
            raise SystemExit(
                f"{spec}: set MEM0_LLM_MODEL to the model id your server "
                f"advertises, e.g. Qwen/Qwen2.5-7B-Instruct. The spec's "
                f"shorthand names the artifact; it cannot name the server."
            )
        return cls(llm_model=model,
                   llm_base_url=os.environ.get("MEM0_BASE_URL",
                                               "http://localhost:8000/v1"),
                   batch=os.environ.get("MEM0_BATCH", DEFAULT_BATCH))

    def config(self) -> dict:
        cfg = {
            "policy": "mem0_oss_v3",
            "mem0_version_pinned": MEM0_PINNED_VERSION,
            # Read after every build() has run, so this reflects the whole arm.
            "write_tokens_estimated": self.usage_estimated,
            "llm_model": self.llm_model,
            "embed_model": self.embed_model,
            "batch": self.batch,
            "batch_note": (
                "Mem0 does not prescribe a batching granularity and v3 makes "
                "one LLM call per add() regardless of message count, so this "
                "choice sets the write cost. Recorded as an explicit deviation."
            ),
            "not_the_managed_platform": (
                "This is the open-source SDK with an open extraction model. "
                "Mem0's published LongMemEval score is for the managed "
                "platform. Do not compare the two."
            ),
        }
        try:
            import mem0

            cfg["mem0_version_installed"] = getattr(mem0, "__version__", "unknown")
        except ImportError:
            cfg["mem0_version_installed"] = "not installed"
        return cfg

    # -- mem0 plumbing -----------------------------------------------------

    def _mem0_config(self, collection: str, path: str) -> dict:
        return {
            # `openai` rather than `vllm`: the provider is an OpenAI-compatible
            # client either way, and this path gets mem0's parameter handling.
            "llm": {"provider": "openai",
                    "config": {"model": self.llm_model, "temperature": 0.0,
                               "openai_base_url": self.llm_base_url,
                               "api_key": "EMPTY", "max_tokens": 4000}},
            "embedder": {"provider": "huggingface",
                         "config": {"model": self.embed_model,
                                    "embedding_dims": self.embed_dims}},
            # qdrant with a local path: embedded, no server, and the only
            # no-server store that implements keyword_search.
            "vector_store": {"provider": "qdrant",
                             "config": {"collection_name": collection,
                                        "path": f"{path}/qdrant",
                                        "embedding_model_dims": self.embed_dims}},
            "history_db_path": f"{path}/history.db",
        }

    def _new_memory(self, collection: str, path: str):
        # posthog is a core mem0 dependency; on a network-restricted kernel an
        # un-disabled telemetry call is a hang, not an error.
        os.environ["MEM0_TELEMETRY"] = "False"
        os.environ.pop("OPENAI_API_KEY", None)
        # OPENROUTER_API_KEY silently redirects the openai provider away from
        # our own server.
        os.environ.pop("OPENROUTER_API_KEY", None)

        from mem0 import Memory

        return Memory.from_config(self._mem0_config(collection, path))

    def preflight(self) -> PreflightResult:
        """Check the four things mem0 degrades on silently."""
        from mem0.vector_stores.base import VectorStoreBase

        with tempfile.TemporaryDirectory() as tmp:
            m = self._new_memory("preflight", tmp)
            store = type(m.vector_store)
            bm25_impl = store.keyword_search is not VectorStoreBase.keyword_search
            slot = bool(getattr(m.vector_store, "_has_bm25_slot", False))
            try:
                encoder = m.vector_store._get_bm25_encoder() is not None
            except Exception:
                encoder = False
            try:
                from mem0.utils.spacy_models import get_nlp_full

                spacy_ok = get_nlp_full() is not None
            except Exception:
                spacy_ok = False

            date = "2023-05-20"
            self._add_with_date(
                m, [{"role": "user", "content": "I met Priya in Lisbon."}],
                user_id="preflight", conversation_date=date)
            rows = m.get_all(filters={"user_id": "preflight"})["results"]
            date_ok = bool(rows) and str(rows[0].get("created_at", "")).startswith(date)

        return PreflightResult(bm25_impl, slot, encoder, spacy_ok, date_ok)

    @staticmethod
    def _add_with_date(memory, messages, *, user_id: str,
                       conversation_date: str, **kw):
        """`add()` with the conversation's date rather than wall-clock time.

        Two separate leaks, both needing a workaround because `add(timestamp=)`
        is hard-rejected in OSS with a "Platform-only" ValueError:

        1. Stored `created_at` defaults to `datetime.now()`. A `created_at` in
           `metadata` survives `_strip_identity_keys` and wins.
        2. The extraction prompt's "Observation Date" also defaults to now, and
           `main.py` never forwards the `timestamp` argument that
           `generate_additive_extraction_prompt` accepts. Patched for the
           duration of the call.

        Without both, the extractor resolves "last week" against today's date
        and every relative time reference in the store is wrong. This is the
        failure behind mem0 issue #3944.

        Not thread-safe: the patch is process-global. Fine for a sequential
        ingest loop, wrong for a thread pool.
        """
        import mem0.memory.main as mm

        original = mm.generate_additive_extraction_prompt
        mm.generate_additive_extraction_prompt = functools.partial(
            original, timestamp=conversation_date)
        try:
            meta = dict(kw.pop("metadata", None) or {})
            meta.setdefault("created_at", f"{conversation_date}T00:00:00+00:00")
            return memory.add(messages, user_id=user_id, metadata=meta, **kw)
        finally:
            mm.generate_additive_extraction_prompt = original

    def _batches(self, ex: Example) -> list[list[Turn]]:
        if self.batch == "session":
            out: dict[str, list[Turn]] = {}
            for t in ex.turns:
                out.setdefault(t.session_id, []).append(t)
            return list(out.values())
        if self.batch == "pair":
            # Paired *within* a session, not across the flat turn list. A pair
            # straddling a session boundary would be stamped with the earlier
            # session's date by build(), which is the conversation-date
            # contamination the preflight check exists to catch.
            out: dict[str, list[Turn]] = {}
            for t in ex.turns:
                out.setdefault(t.session_id, []).append(t)
            return [ts[i:i + 2]
                    for ts in out.values()
                    for i in range(0, len(ts), 2)]
        if self.batch == "turn":
            return [[t] for t in ex.turns]
        raise ValueError(f"unknown batch granularity: {self.batch}")

    def build(self, ex: Example, ledger: CostLedger) -> list[MemoryUnit]:
        with tempfile.TemporaryDirectory(dir=self.workdir) as tmp:
            m = self._new_memory(ex.question_id.replace("-", "_"), tmp)
            counter = CountingChatClient(m.llm.client.chat.completions, ledger)
            m.llm.client.chat.completions = counter

            # A memory's date is the session that last touched it, so the
            # recency control sees the store the way a user would.
            last_date: dict[str, str] = {}
            provenance: dict[str, set[int]] = {}

            with ledger.timer("write"):
                for batch in self._batches(ex):
                    iso = _iso_date(batch[0].session_date)
                    res = self._add_with_date(
                        m,
                        [{"role": t.role, "content": t.content} for t in batch],
                        user_id=ex.question_id,
                        conversation_date=iso,
                    )
                    for r in (res or {}).get("results", []):
                        mid = r.get("id")
                        provenance.setdefault(mid, set()).update(
                            t.turn_index for t in batch)
                        last_date[mid] = batch[0].session_date

                rows = m.get_all(filters={"user_id": ex.question_id})["results"]

            fallback = ex.turns[0].session_date
            units = [
                MemoryUnit(
                    unit_id=i,
                    text=r.get("memory", ""),
                    session_id=str(r.get("id", i)),
                    session_date=last_date.get(r.get("id"), fallback),
                    session_index=i,
                    # Labelled later by containment, exactly as every other
                    # store is, so all arms are read off one ruler.
                    is_evidence=False,
                    roles=(),
                    provenance=tuple(sorted(provenance.get(r.get("id"), ()))),
                    meta={"mem0_id": r.get("id"),
                          "created_at": r.get("created_at")},
                )
                for i, r in enumerate(rows) if r.get("memory")
            ]
            # Sticky across examples: one server that reported no usage makes
            # the arm's write column part-estimated, and the manifest has to
            # say so. This used to append to a `ledger.notes` list that
            # CostLedger does not have, so the flag was raised and dropped --
            # the exact silent mixing the comment on _CountingChatClient says
            # is being prevented.
            self.usage_estimated |= counter.usage_estimated
            return renumber(units)


def _iso_date(session_date: str) -> str:
    """'2023/04/10 (Mon) 17:50' -> '2023-04-10'."""
    from llm_mem_eval.data.loader import parse_date

    y, mo, d = parse_date(session_date)
    return f"{y:04d}-{mo:02d}-{d:02d}"
