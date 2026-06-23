"""EmbeddingsCache: exact-match cache for (text, model) -> embedding vector.

Avoids re-computing embeddings for inputs you've already embedded. The key is
deterministic from (text, model_name), so cache hits are O(1) and don't need a
vector index.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
import redis

from .base import EmbeddingProvider, from_bytes, stable_id, to_bytes


class EmbeddingsCache:
    def __init__(
        self,
        client: redis.Redis,
        name: str = "embedcache",
        provider: Optional[EmbeddingProvider] = None,
        ttl: Optional[int] = None,
    ):
        self.client = client
        self.name = name
        self.provider = provider
        self.ttl = ttl

    def _key(self, text: str, model_name: str) -> str:
        return f"{self.name}:{stable_id(model_name, text)}"

    def set(
        self,
        text: str,
        model_name: str,
        embedding: np.ndarray,
        metadata: Optional[str] = None,
        ttl: Optional[int] = None,
    ) -> str:
        key = self._key(text, model_name)
        mapping = {
            "text": text,
            "model": model_name,
            "dim": int(np.asarray(embedding).shape[-1]),
            "vector": to_bytes(embedding),
        }
        if metadata is not None:
            mapping["metadata"] = metadata
        self.client.hset(key, mapping=mapping)
        eff_ttl = ttl if ttl is not None else self.ttl
        if eff_ttl:
            self.client.expire(key, eff_ttl)
        return key

    def get(self, text: str, model_name: str) -> Optional[dict]:
        key = self._key(text, model_name)
        raw = self.client.hgetall(key)
        if not raw:
            return None
        raw = {
            (k.decode() if isinstance(k, bytes) else k): v
            for k, v in raw.items()
        }
        dim = int(raw["dim"])
        vector = from_bytes(raw["vector"], dim)
        return {
            "key": key,
            "text": raw["text"].decode() if isinstance(raw["text"], bytes) else raw["text"],
            "model": raw["model"].decode() if isinstance(raw["model"], bytes) else raw["model"],
            "vector": vector,
            "metadata": (
                raw.get("metadata").decode()
                if isinstance(raw.get("metadata"), bytes)
                else raw.get("metadata")
            ),
        }

    def exists(self, text: str, model_name: str) -> bool:
        return bool(self.client.exists(self._key(text, model_name)))

    def drop(self, text: str, model_name: str) -> int:
        return self.client.delete(self._key(text, model_name))

    def mget(self, texts: Iterable[str], model_name: str) -> List[Optional[dict]]:
        return [self.get(t, model_name) for t in texts]

    def mdrop(self, texts: Iterable[str], model_name: str) -> int:
        keys = [self._key(t, model_name) for t in texts]
        return self.client.delete(*keys) if keys else 0

    def clear(self) -> int:
        deleted = 0
        for k in self.client.scan_iter(match=f"{self.name}:*", count=500):
            deleted += self.client.delete(k)
        return deleted

    def embed(self, text: str, model_name: Optional[str] = None) -> np.ndarray:
        """Convenience: embed-through-cache. Requires a provider."""
        if self.provider is None:
            raise RuntimeError("EmbeddingsCache.embed requires a provider")
        mname = model_name or type(self.provider).__name__
        hit = self.get(text, mname)
        if hit is not None:
            return hit["vector"]
        v = self.provider.embed(text)
        self.set(text, mname, v)
        return v
