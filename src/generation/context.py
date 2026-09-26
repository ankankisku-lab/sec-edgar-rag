"""Parent-context expansion: retrieved children -> the parents the LLM reads.

Children are small for precise retrieval; the LLM needs the complete unit
(the whole income statement, the whole MD&A subsection). Parents are added in
reranker order, deduplicated, within a token budget. A parent too large for the
budget (a 60-row table) falls back to the retrieved child, which is a row group
that still carries the table's period headers, caption and units.
"""
from __future__ import annotations

from typing import List, Optional

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle

from src.chunking.hierarchical import n_tokens
from src.config import load_config
from src.retrieval.parent_store import ParentStore


class ParentExpander(BaseNodePostprocessor):
    budget_tokens: int = 5000
    max_parent_tokens: int = 1500

    @classmethod
    def from_config(cls) -> "ParentExpander":
        cfg = load_config("generation")["context"]
        return cls(budget_tokens=cfg["budget_tokens"], max_parent_tokens=cfg["max_parent_tokens"])

    def _postprocess_nodes(self, nodes: List[NodeWithScore],
                           query_bundle: Optional[QueryBundle] = None) -> List[NodeWithScore]:
        parents = ParentStore().get([n.node.metadata["parent_id"] for n in nodes if "parent_id" in n.node.metadata])
        out, used, seen = [], 0, set()
        for child in nodes:
            parent = parents.get(child.node.metadata.get("parent_id"))
            unit = child.node
            if parent is not None and n_tokens(parent.get_content(MetadataMode.LLM)) <= self.max_parent_tokens:
                unit = parent
            if unit.node_id in seen or (parent is not None and parent.node_id in seen):
                continue
            cost = n_tokens(unit.get_content(MetadataMode.LLM))
            if used + cost > self.budget_tokens:
                continue  # try smaller units further down the ranking
            seen.add(unit.node_id)
            if parent is not None:
                seen.add(parent.node_id)
            used += cost
            # CitationQueryEngine rebuilds sources as new nodes without their ids; keep the
            # original id in metadata (hidden from the LLM and embedder) for tracing and evaluation.
            unit = unit.model_copy(deep=True)
            unit.metadata["origin_id"] = unit.node_id
            unit.excluded_llm_metadata_keys = [*unit.excluded_llm_metadata_keys, "origin_id"]
            unit.excluded_embed_metadata_keys = [*unit.excluded_embed_metadata_keys, "origin_id"]
            out.append(NodeWithScore(node=unit, score=child.score))
        return out
