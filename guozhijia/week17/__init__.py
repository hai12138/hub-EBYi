"""redisvl_mini: a compact reimplementation of four redisvl modules.

Modules:
  - EmbeddingsCache         (exact-match cache for embedding vectors)
  - SemanticCache           (similarity-based LLM response cache)
  - SemanticMessageHistory  (chat history with semantic recall)
  - SemanticRouter          (intent routing by example similarity)
"""

from .base import (
    EmbeddingProvider,
    HashEmbeddings,
    SentenceTransformerEmbeddings,
    VectorIndex,
)
from .embeddings_cache import EmbeddingsCache
from .semantic_cache import SemanticCache
from .semantic_message_history import SemanticMessageHistory
from .semantic_router import Route, RouteMatch, SemanticRouter

__all__ = [
    "EmbeddingProvider",
    "HashEmbeddings",
    "SentenceTransformerEmbeddings",
    "VectorIndex",
    "EmbeddingsCache",
    "SemanticCache",
    "SemanticMessageHistory",
    "SemanticRouter",
    "Route",
    "RouteMatch",
]
