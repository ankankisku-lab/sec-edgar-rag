"""V4: candidate pool (dense and/or BM25) re-scored by the BGE cross-encoder.

A cross-encoder reads query and chunk together (full attention across both), so
it can check that *this row* is "Operating income" for *this period*, which a
single dense vector or bag-of-words score cannot. It is too slow to run over the
whole index, so it only re-scores a ~50-60 candidate pool.

The reranker sees the same metadata header as the embedder (company, form,
fiscal period, heading path, caption, units), so it can tell Q3 2025 from Q3 2024.
"""
from __future__ import annotations

import time
from functools import lru_cache

from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.schema import NodeWithScore, QueryBundle

from src.config import load_config
from src.retrieval.dense import retrieve as dense_retrieve
from src.retrieval.hybrid import retrieve_bm25

LAST_TIMINGS: dict[str, float] = {}  # per-stage latency of the most recent call (for evaluation)


class TableAwareRerank(SentenceTransformerRerank):
    """SentenceTransformerRerank that shows the cross-encoder table rows as sentences
    ("Operating income: Three Months Ended June 28, 2025 = 28,202; ...") instead of
    markdown, which a model trained on prose passages scores poorly."""

    linearize: bool = True
    batch_size: int = 16   # measured on 60 candidates: 439 ms / 906 MiB vs 939 ms / 1218 MiB at 32

    def _postprocess_nodes(self, nodes, query_bundle=None):
        from src.retrieval.linearize import rerank_text
        if not self.linearize or not nodes:
            return super()._postprocess_nodes(nodes, query_bundle)
        scores = self._model.predict([(query_bundle.query_str, rerank_text(n.node)) for n in nodes],
                                     batch_size=self.batch_size)
        for n, s in zip(nodes, scores):
            n.score = float(s)
        return sorted(nodes, key=lambda n: -n.score)[: self.top_n]


@lru_cache(maxsize=1)
def get_reranker() -> TableAwareRerank:
    import torch
    cfg = load_config("retrieval")["rerank"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reranker = TableAwareRerank(model=cfg["model"], top_n=cfg["top_n"], device=device)
    reranker.batch_size = cfg.get("batch_size", 16)
    if device == "cuda" and cfg.get("fp16", True):
        reranker._model.model.half()  # LlamaIndex exposes no dtype option; fp16 halves VRAM
    # Rank on raw logits: the default sigmoid saturates at ~0.999 for every strong match,
    # turning the top of the list into ties broken arbitrarily.
    reranker._model.activation_fn = torch.nn.Identity()
    return reranker


def candidate_pool(query: str, pool: str) -> list[NodeWithScore]:
    """Named pool from configs/retrieval.yaml, e.g. {dense: 30, bm25: 30}; order-preserving union."""
    sizes = load_config("retrieval")["rerank"]["pools"][pool]
    lists = []
    if sizes.get("bm25"):
        lists.append(retrieve_bm25(query, top_k=sizes["bm25"]))
    if sizes.get("dense"):
        lists.append(dense_retrieve(query, top_k=sizes["dense"]))
    if sizes.get("hyde_dense"):
        from src.retrieval.dense import retrieve_hyde
        lists.append(retrieve_hyde(query, top_k=sizes["hyde_dense"]))
    seen, out = set(), []
    for hits in lists:
        for h in hits:
            if h.node.node_id not in seen:
                seen.add(h.node.node_id)
                out.append(h)
    return out


def retrieve_reranked(query: str, top_k: int = 10, pool: str | None = None,
                      linearize: bool | None = None) -> list[NodeWithScore]:
    cfg = load_config("retrieval")["rerank"]
    t0 = time.perf_counter()
    candidates = candidate_pool(query, pool or cfg["default_pool"])
    t1 = time.perf_counter()
    reranker = get_reranker()
    reranker.linearize = cfg["linearize_tables"] if linearize is None else linearize
    reranked = reranker.postprocess_nodes(candidates, query_bundle=QueryBundle(query))
    t2 = time.perf_counter()
    LAST_TIMINGS.update(pool_ms=(t1 - t0) * 1000, rerank_ms=(t2 - t1) * 1000, n_candidates=len(candidates))
    return reranked[:top_k]
