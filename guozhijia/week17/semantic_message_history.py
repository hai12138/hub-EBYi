"""SemanticMessageHistory: chat history with semantic retrieval.

Stores messages keyed by (session_tag, message_id) and can return either the
most recent messages or those semantically relevant to a query.
"""

from __future__ import annotations

import time
import uuid
from typing import List, Optional

import redis
from redis.commands.search.field import NumericField, TagField, TextField
from redis.commands.search.query import Query

from .base import (
    EmbeddingProvider,
    VectorIndex,
    cosine_distance_to_similarity,
    to_bytes,
)


class SemanticMessageHistory:
    def __init__(
        self,
        client: redis.Redis,
        provider: EmbeddingProvider,
        name: str = "msghist",
        session_tag: str = "default",
        distance_threshold: float = 0.4,
    ):
        self.client = client
        self.provider = provider
        self.name = name
        self.session_tag = session_tag
        self.distance_threshold = distance_threshold

        self.index = VectorIndex(
            client=client,
            index_name=f"{name}-idx",
            prefix=name,
            dim=provider.dim,
            extra_fields=[
                TagField("session"),
                TagField("role"),
                TextField("content"),
                NumericField("timestamp", sortable=True),
            ],
        )
        self.index.create()

    def _doc_id(self) -> str:
        return f"{self.session_tag}-{uuid.uuid4().hex[:16]}"

    def add_message(self, role: str, content: str, session_tag: Optional[str] = None) -> str:
        session = session_tag or self.session_tag
        vec = self.provider.embed(content)
        mapping = {
            "session": session,
            "role": role,
            "content": content,
            "timestamp": int(time.time() * 1000),
            "vector": to_bytes(vec),
        }
        return self.index.upsert(self._doc_id(), mapping)

    def add_messages(self, messages: List[dict], session_tag: Optional[str] = None) -> List[str]:
        return [self.add_message(m["role"], m["content"], session_tag) for m in messages]

    def get_recent(
        self,
        top_k: int = 5,
        session_tag: Optional[str] = None,
        as_text: bool = False,
    ):
        session = session_tag or self.session_tag
        q = (
            Query(f"@session:{{{session}}}")
            .sort_by("timestamp", asc=False)
            .return_fields("role", "content", "timestamp", "session")
            .paging(0, top_k)
            .dialect(2)
        )
        res = self.client.ft(self.index.index_name).search(q)
        msgs = [
            {
                "role": getattr(d, "role", None),
                "content": getattr(d, "content", None),
                "timestamp": int(getattr(d, "timestamp", 0)),
            }
            for d in res.docs
        ]
        msgs.reverse()
        if as_text:
            return "\n".join(f"{m['role']}: {m['content']}" for m in msgs)
        return msgs

    def get_relevant(
        self,
        prompt: str,
        top_k: int = 5,
        distance_threshold: Optional[float] = None,
        session_tag: Optional[str] = None,
    ) -> List[dict]:
        session = session_tag or self.session_tag
        thresh = distance_threshold if distance_threshold is not None else self.distance_threshold
        vec = self.provider.embed(prompt)
        res = self.index.search(
            vec,
            k=top_k,
            return_fields=["role", "content", "timestamp", "session"],
            filter_expr=f"@session:{{{session}}}",
        )
        out = []
        for d in res.docs:
            distance = float(d.score)
            if distance > thresh:
                continue
            out.append(
                {
                    "role": getattr(d, "role", None),
                    "content": getattr(d, "content", None),
                    "timestamp": int(getattr(d, "timestamp", 0)),
                    "distance": distance,
                    "similarity": cosine_distance_to_similarity(distance),
                }
            )
        return out

    def clear(self, session_tag: Optional[str] = None) -> int:
        session = session_tag or self.session_tag
        q = (
            Query(f"@session:{{{session}}}")
            .return_fields("session")
            .paging(0, 10000)
            .dialect(2)
        )
        res = self.client.ft(self.index.index_name).search(q)
        if not res.docs:
            return 0
        return self.client.delete(*[d.id for d in res.docs])
