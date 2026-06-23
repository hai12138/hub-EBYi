"""SemanticCache: cache LLM prompt→response pairs by prompt similarity.

On lookup we embed the prompt, kNN-search the index, and if the top hit's
similarity exceeds the threshold we return its stored response.
"""

from __future__ import annotations

import json
import time
from typing import List, Optional

import redis
from redis.commands.search.field import TagField, TextField

from .base import (
    EmbeddingProvider,
    VectorIndex,
    cosine_distance_to_similarity,
    stable_id,
    to_bytes,
)


class SemanticCache:
    def __init__(
        self,
        client: redis.Redis,
        provider: EmbeddingProvider,
        name: str = "semcache",
        distance_threshold: float = 0.2,
        ttl: Optional[int] = None,
    ):
        """distance_threshold is in cosine-distance space (0=identical, 2=opposite).
        A hit means distance <= threshold, i.e. similarity >= 1 - threshold."""
        self.client = client
        self.provider = provider
        self.name = name
        self.distance_threshold = distance_threshold
        self.ttl = ttl

        self.index = VectorIndex(
            client=client,
            index_name=f"{name}-idx",
            prefix=f"{name}",
            dim=provider.dim,
            extra_fields=[
                TextField("prompt"),
                TextField("response"),
                TagField("tag"),
            ],
        )
        self.index.create()

    def _doc_id(self, prompt: str) -> str:
        return stable_id("semcache", prompt)

    def store(
        self,
        prompt: str,
        response: str,
        metadata: Optional[dict] = None,
        tag: Optional[str] = None,
        ttl: Optional[int] = None,
    ) -> str:
        vec = self.provider.embed(prompt)
        doc_id = self._doc_id(prompt)
        mapping = {
            "prompt": prompt,
            "response": response,
            "vector": to_bytes(vec),
            "created_at": int(time.time()),
        }
        if metadata is not None:
            mapping["metadata"] = json.dumps(metadata, ensure_ascii=False)
        if tag is not None:
            mapping["tag"] = tag
        eff_ttl = ttl if ttl is not None else self.ttl
        return self.index.upsert(doc_id, mapping, ttl=eff_ttl)

    def check(
        self,
        prompt: str,
        num_results: int = 1,
        distance_threshold: Optional[float] = None,
        tag: Optional[str] = None,
    ) -> List[dict]:
        thresh = distance_threshold if distance_threshold is not None else self.distance_threshold
        vec = self.provider.embed(prompt)
        filter_expr = f"@tag:{{{tag}}}" if tag else "*"
        res = self.index.search(
            vec,
            k=num_results,
            return_fields=["prompt", "response", "metadata", "tag", "created_at"],
            filter_expr=filter_expr,
        )
        hits = []
        for d in res.docs:
            distance = float(d.score)
            if distance > thresh:
                continue
            md = getattr(d, "metadata", None)
            hits.append(
                {
                    "key": d.id,
                    "prompt": getattr(d, "prompt", None),
                    "response": getattr(d, "response", None),
                    "tag": getattr(d, "tag", None),
                    "metadata": json.loads(md) if md else None,
                    "distance": distance,
                    "similarity": cosine_distance_to_similarity(distance),
                }
            )
        return hits

    def drop(self, prompt: str) -> int:
        return self.index.delete(self._doc_id(prompt))

    def clear(self) -> None:
        self.index.drop(delete_documents=True)
        self.index.create()
