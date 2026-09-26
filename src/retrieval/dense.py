"""V1 baseline retriever: dense similarity search over child chunks in Qdrant."""
from __future__ import annotations

from functools import lru_cache

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore

from src.embeddings.embedder import get_embed_model
from src.retrieval.qdrant_index import get_vector_store


@lru_cache(maxsize=1)
def get_index() -> VectorStoreIndex:
    return VectorStoreIndex.from_vector_store(get_vector_store(), embed_model=get_embed_model())


def retrieve(query: str, top_k: int = 10, filters=None) -> list[NodeWithScore]:
    return get_index().as_retriever(similarity_top_k=top_k, filters=filters).retrieve(query)


def retrieve_hyde(query: str, top_k: int = 10, filters=None) -> list[NodeWithScore]:
    """Dense search with the HyDE embedding (hypothetical passage + question)."""
    from llama_index.core.schema import QueryBundle
    from src.query.hyde import hyde_embedding
    bundle = QueryBundle(query_str=query, embedding=hyde_embedding(query))
    return get_index().as_retriever(similarity_top_k=top_k, filters=filters).retrieve(bundle)
