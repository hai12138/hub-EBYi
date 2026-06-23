"""Shared utilities: embedding providers, Redis vector index helpers."""

from __future__ import annotations

import hashlib
from typing import List, Optional, Protocol

import numpy as np
import redis
from redis.commands.search.field import VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.exceptions import ResponseError


class EmbeddingProvider(Protocol):
    """Anything that can turn text into a float32 vector of fixed dimension."""

    dim: int

    def embed(self, text: str) -> np.ndarray: ...

    def embed_many(self, texts: List[str]) -> List[np.ndarray]: ...


class HashEmbeddings:
    """Deterministic hash-based embedding, useful for tests / no-network demos.

    Not semantically meaningful, but stable: same input → same vector.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim

    def _seed(self, text: str) -> int:
        return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")

    def embed(self, text: str) -> np.ndarray:
        rng = np.random.default_rng(self._seed(text))
        v = rng.standard_normal(self.dim).astype(np.float32)
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_many(self, texts: List[str]) -> List[np.ndarray]:
        return [self.embed(t) for t in texts]


class SentenceTransformerEmbeddings:
    """Real sentence-transformers backend. Lazy-imported."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> np.ndarray:
        v = self._model.encode(text, normalize_embeddings=True)
        return np.asarray(v, dtype=np.float32)

    def embed_many(self, texts: List[str]) -> List[np.ndarray]:
        vs = self._model.encode(texts, normalize_embeddings=True)
        return [np.asarray(v, dtype=np.float32) for v in vs]


def to_bytes(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def from_bytes(buf: bytes, dim: int) -> np.ndarray:
    return np.frombuffer(buf, dtype=np.float32, count=dim)


def stable_id(*parts: str) -> str:
    h = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return h[:32]


class VectorIndex:
    """Thin wrapper around RediSearch HNSW vector index over a HASH key prefix."""

    def __init__(
        self,
        client: redis.Redis,
        index_name: str,
        prefix: str,
        dim: int,
        extra_fields: Optional[List] = None,
        distance_metric: str = "COSINE",
    ):
        self.client = client
        self.index_name = index_name
        self.prefix = prefix.rstrip(":") + ":"
        self.dim = dim
        self.distance_metric = distance_metric
        self.extra_fields = extra_fields or []

    def exists(self) -> bool:
        try:
            self.client.ft(self.index_name).info()
            return True
        except ResponseError:
            return False

    def create(self, recreate: bool = False) -> None:
        if self.exists():
            if not recreate:
                return
            self.client.ft(self.index_name).dropindex(delete_documents=False)

        schema = [
            VectorField(
                "vector",
                "HNSW",
                {
                    "TYPE": "FLOAT32",
                    "DIM": self.dim,
                    "DISTANCE_METRIC": self.distance_metric,
                },
            ),
            *self.extra_fields,
        ]
        definition = IndexDefinition(prefix=[self.prefix], index_type=IndexType.HASH)
        self.client.ft(self.index_name).create_index(schema, definition=definition)

    def drop(self, delete_documents: bool = True) -> None:
        if self.exists():
            self.client.ft(self.index_name).dropindex(delete_documents=delete_documents)

    def key(self, doc_id: str) -> str:
        return self.prefix + doc_id

    def upsert(self, doc_id: str, mapping: dict, ttl: Optional[int] = None) -> str:
        key = self.key(doc_id)
        self.client.hset(key, mapping=mapping)
        if ttl:
            self.client.expire(key, ttl)
        return key

    def delete(self, doc_id: str) -> int:
        return self.client.delete(self.key(doc_id))

    def search(
        self,
        vector: np.ndarray,
        k: int = 1,
        return_fields: Optional[List[str]] = None,
        filter_expr: str = "*",
    ):
        from redis.commands.search.query import Query

        q = (
            Query(f"({filter_expr})=>[KNN {k} @vector $vec AS score]")
            .sort_by("score")
            .return_fields(*(return_fields or []), "score")
            .dialect(2)
            .paging(0, k)
        )
        return self.client.ft(self.index_name).search(q, query_params={"vec": to_bytes(vector)})


def cosine_distance_to_similarity(distance: float) -> float:
    """RediSearch COSINE returns 1 - cosine_similarity. Invert to similarity in [0, 1]."""
    return max(0.0, 1.0 - float(distance))
