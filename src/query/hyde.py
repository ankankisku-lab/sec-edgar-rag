"""HyDE: search with a hypothetical answer passage instead of the bare question.

The local LLM writes a short passage in 10-K/10-Q style that would answer the
question (LlamaIndex HyDEQueryTransform). Filing text (MD&A explanations, risk
factors) is closer to such a passage than to a question, so its embedding can land
nearer the relevant chunks. The passage is used ONLY for retrieval; it is never
shown to the answering LLM or the user, and any figures it invents are irrelevant.

BGE embeds queries and passages asymmetrically: the hypothetical passage is embedded
as a passage (no query instruction) and averaged with the question's query
embedding (LlamaIndex's default would embed the passage with the query instruction).
Passages are cached on disk so experiments are reproducible.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache

import numpy as np

from src.config import DATA_DIR, load_config

CACHE_PATH = DATA_DIR / "eval" / "cache" / "hyde.jsonl"

HYDE_PROMPT = (
    "Write a short passage (at most 80 words) as it would appear in a company's SEC 10-K or 10-Q "
    "filing that answers the question below. Use the typical wording of the relevant section "
    "(for example Management's Discussion and Analysis, risk factors, or financial statement notes). "
    "Do not add any introduction or explanation.\n"
    "Question: {context_str}\n"
    "Passage:"
)


@lru_cache(maxsize=1)
def get_hyde_transform():
    from llama_index.core import PromptTemplate
    from llama_index.core.indices.query.query_transform import HyDEQueryTransform
    from llama_index.llms.ollama import Ollama
    from src.generation import ollama_server
    cfg = load_config("generation")["llm"]
    llm = Ollama(model=cfg["model"], base_url=ollama_server.base_url(), temperature=0.0,
                 context_window=cfg["context_window"], request_timeout=cfg["request_timeout_s"],
                 keep_alive=cfg["keep_alive"], thinking=False, additional_kwargs={"seed": 0, "num_predict": 160})
    return HyDEQueryTransform(llm=llm, hyde_prompt=PromptTemplate(HYDE_PROMPT), include_original=False)


def _key(question: str) -> str:
    model = load_config("generation")["llm"]["model"]
    return hashlib.sha1(f"{model}\n{HYDE_PROMPT}\n{question}".encode()).hexdigest()


@lru_cache(maxsize=1)
def _cache() -> dict[str, str]:
    if not CACHE_PATH.exists():
        return {}
    with open(CACHE_PATH, encoding="utf-8") as fh:
        return {r["key"]: r["passage"] for r in map(json.loads, fh)}


def hypothetical_passage(question: str) -> str:
    key = _key(question)
    cache = _cache()
    if key not in cache:
        bundle = get_hyde_transform().run(question)
        cache[key] = bundle.custom_embedding_strs[0].strip()
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"key": key, "question": question, "passage": cache[key]}, ensure_ascii=False) + "\n")
    return cache[key]


def hyde_embedding(question: str) -> list[float]:
    """Mean of the passage embedding (as a passage) and the question embedding (as a query)."""
    from src.embeddings.embedder import get_embed_model
    model = get_embed_model()
    passage = np.asarray(model.get_text_embedding(hypothetical_passage(question)))
    query = np.asarray(model.get_query_embedding(question))
    v = (passage + query) / 2
    return (v / np.linalg.norm(v)).tolist()
