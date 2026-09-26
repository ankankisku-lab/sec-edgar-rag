"""V2 (BM25 only) and V3 (hybrid BM25 + dense) retrievers over the same collection.

Hybrid: dense top-k and BM25 top-k are retrieved separately, then fused:
  * relative -- LlamaIndex's relative score fusion: min-max normalise each list's
                scores, then alpha * dense + (1 - alpha) * sparse
  * rrf      -- Reciprocal Rank Fusion: sum of 1 / (rrf_k + rank); ignores raw
                score scales entirely
"""
from __future__ import annotations

from functools import lru_cache

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores.types import VectorStoreQueryResult

from src.config import load_config
from src.embeddings.embedder import get_embed_model
from src.retrieval.qdrant_index import get_vector_store


def rrf_fusion(dense: VectorStoreQueryResult, sparse: VectorStoreQueryResult,
               alpha: float = 0.5, top_k: int = 10) -> VectorStoreQueryResult:
    """Drop-in for QdrantVectorStore.hybrid_fusion_fn (alpha is unused by RRF)."""
    k = load_config("retrieval")["hybrid"]["rrf_k"]
    scores: dict[str, float] = {}
    nodes = {}
    for result in (dense, sparse):
        for rank, node in enumerate(result.nodes or [], start=1):
            scores[node.node_id] = scores.get(node.node_id, 0.0) + 1.0 / (k + rank)
            nodes[node.node_id] = node
    best = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return VectorStoreQueryResult(nodes=[nodes[i] for i in best], similarities=[scores[i] for i in best], ids=best)


@lru_cache(maxsize=4)
def _index(fusion: str) -> VectorStoreIndex:
    store = get_vector_store(fusion_fn=rrf_fusion if fusion == "rrf" else None)
    return VectorStoreIndex.from_vector_store(store, embed_model=get_embed_model())


def retrieve_bm25(query: str, top_k: int = 10, filters=None) -> list[NodeWithScore]:
    retriever = _index("relative").as_retriever(vector_store_query_mode="sparse", similarity_top_k=top_k,
                                                sparse_top_k=top_k, filters=filters)
    return retriever.retrieve(query)


def retrieve_hybrid(query: str, top_k: int | None = None, filters=None, fusion: str | None = None,
                    alpha: float | None = None) -> list[NodeWithScore]:
    cfg = load_config("retrieval")["hybrid"]
    fusion = fusion or cfg["fusion"]
    retriever = _index(fusion).as_retriever(
        vector_store_query_mode="hybrid",
        similarity_top_k=cfg["dense_top_k"], sparse_top_k=cfg["sparse_top_k"],
        hybrid_top_k=top_k or cfg["top_k"], alpha=cfg["alpha"] if alpha is None else alpha,
        filters=filters,
    )
    return retriever.retrieve(query)
