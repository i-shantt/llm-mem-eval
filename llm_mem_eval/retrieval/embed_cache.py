"""Disk cache for unit embeddings, with honest cost replay.

Caching is necessary -- embedding one LongMemEval haystack takes ~11s, and we
re-run sweeps constantly. But naive caching would report a near-zero write cost
on the second run, which would silently destroy the only number this project
exists to measure.

So the cache stores the *cost* alongside the vectors: the wall-clock and token
count actually incurred the first time. On a hit we replay those into the
ledger instead of timing the disk read. Runs that replayed cached timings are
flagged in the results payload, and `--no-cache` forces authoritative timing.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

CACHE_DIR = Path("data/emb_cache")


class EmbeddingCache:
    def __init__(self, enabled: bool = True, cache_dir: Path = CACHE_DIR) -> None:
        self.enabled = enabled
        self.cache_dir = cache_dir
        self.hits = 0
        self.misses = 0
        if enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def content_digest(texts: list[str]) -> str:
        """Fingerprint of exactly the strings that were embedded.

        The key used to be (model, cache_key, n_units), which is not enough to
        identify what was embedded. Callers key on the question and granularity
        -- `f"{ex.question_id}|{args.granularity}"` -- so two *different* stores
        for the same question collide whenever they hold the same number of
        records. That is not hypothetical: the verbatim and lead-k write policies
        both produce 497 records for a typical conversation, so routing a write
        policy's store through DenseRetriever with the cache on would have
        returned the other policy's vectors and reported it as a hit. Hashing the
        content makes the key say what it means.
        """
        h = hashlib.sha1()
        for t in texts:
            h.update(t.encode("utf-8", "replace"))
            h.update(b"\x00")  # so ["ab","c"] and ["a","bc"] differ
        return h.hexdigest()[:16]

    def _path(self, model_name: str, cache_key: str, texts: list[str]) -> Path:
        digest = hashlib.sha1(
            f"{model_name}|{cache_key}|{len(texts)}|"
            f"{self.content_digest(texts)}".encode()
        ).hexdigest()[:20]
        return self.cache_dir / f"{digest}.npz"

    def get(
        self, model_name: str, cache_key: str, texts: list[str]
    ) -> tuple[np.ndarray, dict] | None:
        if not self.enabled:
            return None
        p = self._path(model_name, cache_key, texts)
        if not p.exists():
            self.misses += 1
            return None
        try:
            with np.load(p, allow_pickle=False) as z:
                emb = z["emb"]
                cost = json.loads(str(z["cost"]))
        except Exception:
            self.misses += 1
            return None
        if emb.shape[0] != len(texts):
            self.misses += 1
            return None
        self.hits += 1
        return emb, cost

    def put(
        self,
        model_name: str,
        cache_key: str,
        texts: list[str],
        emb: np.ndarray,
        cost: dict,
    ) -> None:
        if not self.enabled:
            return
        p = self._path(model_name, cache_key, texts)
        try:
            np.savez(p, emb=emb, cost=np.array(json.dumps(cost)))
        except Exception:
            pass  # cache is an optimisation; never fail a run over it

    @property
    def used_replayed_timings(self) -> bool:
        return self.hits > 0

    def stats(self) -> dict:
        return {
            "enabled": self.enabled,
            "hits": self.hits,
            "misses": self.misses,
            "used_replayed_timings": self.used_replayed_timings,
        }
