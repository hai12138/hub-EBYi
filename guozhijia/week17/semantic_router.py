"""SemanticRouter: route an input to the best-matching named "route".

Each route owns one or more example utterances ("references"). A query is
embedded and matched against all references; the route whose top match clears
the distance threshold wins. Multiple aggregation strategies are supported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import redis
from redis.commands.search.field import TagField, TextField

from .base import (
    EmbeddingProvider,
    VectorIndex,
    cosine_distance_to_similarity,
    stable_id,
    to_bytes,
)


@dataclass
class Route:
    name: str
    references: List[str]
    distance_threshold: float = 0.3
    metadata: Dict = field(default_factory=dict)


@dataclass
class RouteMatch:
    name: Optional[str]
    distance: float
    similarity: float
    reference: Optional[str] = None


class SemanticRouter:
    AGGREGATIONS = ("min", "avg", "sum")

    def __init__(
        self,
        client: redis.Redis,
        provider: EmbeddingProvider,
        routes: List[Route],
        name: str = "semrouter",
        aggregation: str = "min",
    ):
        if aggregation not in self.AGGREGATIONS:
            raise ValueError(f"aggregation must be one of {self.AGGREGATIONS}")

        self.client = client
        self.provider = provider
        self.name = name
        self.aggregation = aggregation
        self.routes: Dict[str, Route] = {r.name: r for r in routes}

        self.index = VectorIndex(
            client=client,
            index_name=f"{name}-idx",
            prefix=name,
            dim=provider.dim,
            extra_fields=[
                TagField("route"),
                TextField("reference"),
            ],
        )
        self.index.create()
        self._ingest_routes()

    def _ref_id(self, route_name: str, reference: str) -> str:
        return stable_id(route_name, reference)

    def _ingest_routes(self) -> None:
        for route in self.routes.values():
            for ref in route.references:
                vec = self.provider.embed(ref)
                self.index.upsert(
                    self._ref_id(route.name, ref),
                    {"route": route.name, "reference": ref, "vector": to_bytes(vec)},
                )

    def add_route(self, route: Route) -> None:
        self.routes[route.name] = route
        for ref in route.references:
            vec = self.provider.embed(ref)
            self.index.upsert(
                self._ref_id(route.name, ref),
                {"route": route.name, "reference": ref, "vector": to_bytes(vec)},
            )

    def remove_route(self, name: str) -> int:
        route = self.routes.pop(name, None)
        if route is None:
            return 0
        removed = 0
        for ref in route.references:
            removed += self.index.delete(self._ref_id(name, ref))
        return removed

    def __call__(self, statement: str, top_k: int = 1) -> RouteMatch:
        return self.route(statement, top_k=top_k)

    def route(self, statement: str, top_k: int = 5) -> RouteMatch:
        """Return the best-matching route, or RouteMatch(None, ...) if none clears its threshold."""
        if not self.routes:
            return RouteMatch(None, float("inf"), 0.0)

        vec = self.provider.embed(statement)
        # Pull enough candidates to aggregate per-route. Each route gets at most
        # len(refs) hits; over-fetch a bit to be safe.
        k = max(top_k, sum(len(r.references) for r in self.routes.values()))
        res = self.index.search(
            vec,
            k=k,
            return_fields=["route", "reference"],
        )

        per_route: Dict[str, List[tuple]] = {}
        for d in res.docs:
            route_name = getattr(d, "route", None)
            if route_name not in self.routes:
                continue
            per_route.setdefault(route_name, []).append((float(d.score), getattr(d, "reference", None)))

        best: Optional[RouteMatch] = None
        for route_name, hits in per_route.items():
            distances = [h[0] for h in hits]
            if self.aggregation == "min":
                agg = min(distances)
            elif self.aggregation == "avg":
                agg = sum(distances) / len(distances)
            else:
                agg = sum(distances)

            route = self.routes[route_name]
            if agg > route.distance_threshold:
                continue

            top_ref = min(hits, key=lambda h: h[0])[1]
            cand = RouteMatch(
                name=route_name,
                distance=agg,
                similarity=cosine_distance_to_similarity(agg),
                reference=top_ref,
            )
            if best is None or cand.distance < best.distance:
                best = cand

        return best or RouteMatch(None, float("inf"), 0.0)

    def route_many(self, statement: str, top_k: int = 3) -> List[RouteMatch]:
        """All routes whose aggregated distance clears their threshold, best first."""
        if not self.routes:
            return []
        vec = self.provider.embed(statement)
        k = sum(len(r.references) for r in self.routes.values())
        res = self.index.search(vec, k=k, return_fields=["route", "reference"])

        per_route: Dict[str, List[tuple]] = {}
        for d in res.docs:
            per_route.setdefault(getattr(d, "route", ""), []).append((float(d.score), getattr(d, "reference", None)))

        matches: List[RouteMatch] = []
        for route_name, hits in per_route.items():
            route = self.routes.get(route_name)
            if route is None:
                continue
            distances = [h[0] for h in hits]
            if self.aggregation == "min":
                agg = min(distances)
            elif self.aggregation == "avg":
                agg = sum(distances) / len(distances)
            else:
                agg = sum(distances)
            if agg > route.distance_threshold:
                continue
            matches.append(
                RouteMatch(
                    name=route_name,
                    distance=agg,
                    similarity=cosine_distance_to_similarity(agg),
                    reference=min(hits, key=lambda h: h[0])[1],
                )
            )
        matches.sort(key=lambda m: m.distance)
        return matches[:top_k]

    def clear(self) -> None:
        self.index.drop(delete_documents=True)
        self.index.create()
        self._ingest_routes()
