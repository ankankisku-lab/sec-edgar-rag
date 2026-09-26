"""Grounded answer generation (LlamaIndex CitationQueryEngine + local Qwen via Ollama).

    question -> V4 retrieval (BM25 top-60 -> linearized BGE rerank)
             -> parent expansion (token budget)
             -> numbered sources -> Qwen3-4B (Ollama) with a grounding prompt
             -> <calc> arithmetic done in Python -> number verification

Usage:
    python -m src.generation.generator "What was Apple's operating income in Q3 fiscal 2025?"
"""
from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from functools import lru_cache

from llama_index.core import PromptTemplate
from llama_index.core.query_engine import CitationQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle

from src.config import load_config
from src.generation import ollama_server
from src.generation.context import ParentExpander
from src.generation.numbers import apply_calcs, verify_numbers
from src.retrieval.reranker import retrieve_reranked

QA_TEMPLATE = PromptTemplate(
    "You are a financial analyst answering questions about SEC 10-K and 10-Q filings.\n"
    "Use ONLY the numbered sources below. Rules:\n"
    "1. Cite the source number after every fact, like [1] or [2][3].\n"
    "2. Quote figures exactly as they appear in the source, always with units and the period they "
    "belong to: dollar amounts from a table 'in millions' as $28,202 million; share counts as "
    "7,433 million shares (no $ sign); per-share amounts as $1.57 per share.\n"
    "3. Filings show several periods side by side: check that the company, the fiscal period "
    "and the column match the question before quoting a number.\n"
    "4. Never do arithmetic yourself, not even rounding. Every difference, ratio or percentage "
    "change must be written as <calc>expression</calc> with plain numbers (no commas, negatives "
    "as -171), for example <calc>(28202 - 25352) / 25352 * 100</calc>%. It is computed for you.\n"
    "5. When the question compares periods or companies, state each value with its period and "
    "then the difference as a <calc>, e.g. 'up $<calc>28202 - 25352</calc> million'.\n"
    "6. Only if none of the requested figures is in the sources, reply exactly: \"The provided "
    "filings do not contain this information.\" Never add that sentence after an answer. "
    "Do not use outside knowledge.\n"
    "Be concise: answer first, then a short explanation if the question asks why or how.\n\n"
    "Sources:\n{context_str}\n\n"
    "Question: {query_str}\n"
    "Answer:"
)


class V4Retriever(BaseRetriever):
    """BM25 top-60 candidates re-scored by the table-aware BGE reranker."""

    def __init__(self, top_k: int):
        super().__init__()
        self.top_k = top_k

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        return retrieve_reranked(query_bundle.query_str, top_k=self.top_k)


@lru_cache(maxsize=1)
def get_llm():
    from llama_index.llms.ollama import Ollama
    cfg = load_config("generation")["llm"]
    return Ollama(model=cfg["model"], base_url=ollama_server.base_url(), temperature=cfg["temperature"],
                  context_window=cfg["context_window"], request_timeout=cfg["request_timeout_s"],
                  keep_alive=cfg["keep_alive"], thinking=False,
                  additional_kwargs={"seed": cfg["seed"], "num_predict": cfg["max_output_tokens"]})


@lru_cache(maxsize=2)
def get_query_engine(decompose: bool = False) -> CitationQueryEngine:
    """decompose=True: multi-fact questions are split into sub-questions, each retrieved
    separately and interleaved (V5); the LLM still answers the original question."""
    from llama_index.core import get_response_synthesizer
    from src.retrieval.decomposed import DecomposedRetriever
    cfg = load_config("generation")
    top_k = cfg["retrieval"]["top_k_children"]
    return CitationQueryEngine(
        retriever=DecomposedRetriever(top_k) if decompose else V4Retriever(top_k),
        llm=get_llm(),
        node_postprocessors=[ParentExpander.from_config()],
        citation_chunk_size=4096,           # never re-split our parents; the budget is enforced upstream
        metadata_mode=MetadataMode.LLM,     # company / form / period / section / caption / units
        response_synthesizer=get_response_synthesizer(llm=get_llm(), text_qa_template=QA_TEMPLATE,
                                                       response_mode="compact"),
    )


@contextmanager
def ollama_running():
    """Start the portable Ollama server for the duration of a block if it isn't up."""
    proc = None if ollama_server.is_up() else ollama_server.start("q8_0")
    try:
        yield
    finally:
        if proc is not None:
            ollama_server.stop(proc)


def answer(question: str, decompose: bool = False) -> dict:
    t0 = time.perf_counter()
    response = get_query_engine(decompose).query(question)
    total_ms = (time.perf_counter() - t0) * 1000
    sources_text = "\n".join(n.node.get_content(metadata_mode=MetadataMode.NONE) for n in response.source_nodes)
    text, calcs = apply_calcs(str(response))
    sources = [{"n": i, "node_id": s.node.metadata.get("origin_id"),
                "filing_id": s.node.metadata.get("filing_id"), "form": s.node.metadata.get("form_type"),
                "period": s.node.metadata.get("period"), "section": s.node.metadata.get("heading_path"),
                "url": s.node.metadata.get("source_url"), "score": round(float(s.score or 0), 3)}
               for i, s in enumerate(response.source_nodes, 1)]
    if decompose:
        from src.retrieval.decomposed import LAST_SUB_QUESTIONS
        sub_questions = list(LAST_SUB_QUESTIONS)
    else:
        sub_questions = [question]
    return {"question": question, "sub_questions": sub_questions,
            "answer": text.strip(), "sources": sources, "calculations": calcs,
            "unverified_numbers": verify_numbers(text, sources_text, question, calcs),
            "latency_ms": round(total_ms)}


def main() -> None:
    question = " ".join(sys.argv[1:]) or "What was Apple's operating income in the third quarter of fiscal 2025?"
    with ollama_running():
        result = answer(question)
    print(f"Q: {result['question']}\n\nA: {result['answer']}\n")
    for s in result["sources"]:
        print(f"  [{s['n']}] {s['filing_id']} | {s['period']} | {(s['section'] or '')[-70:]}")
    if result["calculations"]:
        print("\ncalculations:", [(c["expression"], c.get("text", c.get("error"))) for c in result["calculations"]])
    print(f"\nunverified numbers: {result['unverified_numbers'] or 'none'} | latency {result['latency_ms']} ms")


if __name__ == "__main__":
    main()
