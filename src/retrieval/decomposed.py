"""V5 retrieval: decompose the question, retrieve per sub-question, interleave.

Each sub-question runs through V4 (BM25 top-60 -> linearized BGE rerank). The
ranked lists are interleaved round-robin (1st of sub-q 1, 1st of sub-q 2, 2nd of
sub-q 1, ...) and deduplicated, so every required fact is represented near the
top instead of the first sub-question crowding out the second.
"""
from __future__ import annotations

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle

from src.query.decomposition import decompose
from src.retrieval.reranker import retrieve_reranked

LAST_SUB_QUESTIONS: list[str] = []  # for tracing / evaluation


def interleave(lists: list[list[NodeWithScore]], top_k: int) -> list[NodeWithScore]:
    out, seen = [], set()
    for rank in range(max((len(l) for l in lists), default=0)):
        for hits in lists:
            if rank < len(hits) and hits[rank].node.node_id not in seen:
                seen.add(hits[rank].node.node_id)
                out.append(hits[rank])
                if len(out) == top_k:
                    return out
    return out


def retrieve_decomposed(query: str, top_k: int = 10, per_sub_k: int | None = None) -> list[NodeWithScore]:
    subs = decompose(query)
    LAST_SUB_QUESTIONS[:] = subs
    lists = [retrieve_reranked(s, top_k=per_sub_k or top_k) for s in subs]
    return interleave(lists, top_k)


class DecomposedRetriever(BaseRetriever):
    def __init__(self, top_k: int):
        super().__init__()
        self.top_k = top_k

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        return retrieve_decomposed(query_bundle.query_str, top_k=self.top_k)
