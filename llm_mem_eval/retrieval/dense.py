"""Dense retrieval with a small local embedding model.

bge-small-en-v1.5 is ~33M params and runs on CPU or MPS/CUDA. No API calls, so
the write path costs embedding forward passes and wall-clock, not dollars.
"""

from __future__ import annotations

import time

import numpy as np

from ..cost import CostLedger, count_tokens
from ..data.loader import MemoryUnit
from .base import Hit
from .embed_cache import EmbeddingCache

# bge models want an instruction prefix on the query side only.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_MODEL_CACHE: dict[tuple[str, str], object] = {}


def resolve_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_model(model_name: str, device: str = "auto"):
    """Cache models across examples -- we re-index per example, and reloading
    the encoder 500 times would dominate the measured write cost."""
    dev = resolve_device(device)
    key = (model_name, dev)
    if key not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer

        _MODEL_CACHE[key] = SentenceTransformer(model_name, device=dev)
    return _MODEL_CACHE[key]


class DenseRetriever:
    name = "dense"

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: str = "auto",
        batch_size: int = 128,
        cache: "EmbeddingCache | None" = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.cache = cache
        self._emb: np.ndarray | None = None
        self._unit_ids: list[int] = []
        self._use_prefix = "bge" in model_name.lower()

    def warmup(self) -> None:
        """Load the encoder and run one forward pass before measurement starts.

        Without this, a cache hit skips model loading during index(), and the
        first search() call silently absorbs ~6s of model load into the
        read-path timing.
        """
        model = get_model(self.model_name, self.device)
        model.encode(["warmup"], convert_to_numpy=True, show_progress_bar=False)

    def _encode(self, texts: list[str]) -> np.ndarray:
        model = get_model(self.model_name, self.device)
        return model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def index(
        self,
        units: list[MemoryUnit],
        ledger: CostLedger,
        cache_key: str | None = None,
    ) -> None:
        texts = [u.text for u in units]
        self._unit_ids = [u.unit_id for u in units]

        if self.cache is not None and cache_key is not None:
            cached = self.cache.get(self.model_name, cache_key, len(texts))
            if cached is not None:
                self._emb, cost = cached
                # Replay the cost actually incurred when these were computed,
                # rather than the cost of reading them off disk.
                ledger.write.wall_clock_s += cost["wall_clock_s"]
                ledger.add_embed(
                    "write", n_items=cost["n_items"], tokens=cost["tokens"]
                )
                return

        # Warm the model outside the timer so model load isn't billed to
        # the first example's write path.
        get_model(self.model_name, self.device)
        tokens = sum(count_tokens(t) for t in texts)
        t0 = time.perf_counter()
        self._emb = self._encode(texts)
        elapsed = time.perf_counter() - t0
        ledger.write.wall_clock_s += elapsed
        ledger.add_embed("write", n_items=len(texts), tokens=tokens)

        if self.cache is not None and cache_key is not None:
            self.cache.put(
                self.model_name,
                cache_key,
                self._emb,
                {
                    "wall_clock_s": elapsed,
                    "tokens": tokens,
                    "n_items": len(texts),
                },
            )

    def search(
        self,
        query: str,
        k: int,
        ledger: CostLedger,
        question_date: str | None = None,
    ) -> list[Hit]:
        if self._emb is None:
            raise RuntimeError("index() must be called before search()")
        q = BGE_QUERY_PREFIX + query if self._use_prefix else query
        with ledger.timer("read"):
            qv = self._encode([q])[0]
            sims = self._emb @ qv
            top = np.argsort(-sims)[:k]
        ledger.add_embed("read", n_items=1, tokens=count_tokens(q))
        return [(self._unit_ids[i], float(sims[i])) for i in top]

    def scores_for(self, query: str) -> np.ndarray:
        """Full score vector, for use by the hybrid retriever."""
        if self._emb is None:
            raise RuntimeError("index() must be called before scores_for()")
        q = BGE_QUERY_PREFIX + query if self._use_prefix else query
        return self._emb @ self._encode([q])[0]
