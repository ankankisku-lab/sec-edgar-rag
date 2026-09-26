"""Sub-question decomposition for multi-fact questions.

    "Compare X's net income in fiscal 2025 Q1 with Q3"
        -> ["What was X's net income in fiscal 2025 Q1?", "What was X's net income in fiscal 2025 Q3?"]

One call to the local LLM (Ollama JSON mode). Each sub-question is a single,
self-contained lookup: one company, one metric, one explicit period, which is
exactly the shape our retriever handles well (a compound question pulls in
MD&A comparison text instead of the statement rows). Single-fact questions come
back unchanged.

The few-shot examples deliberately use companies and metrics that are NOT in the
evaluation set (Costco, Intel/AMD, Tesla), so the eval questions don't leak into
the prompt.

Decompositions are cached on disk (key: question + model + prompt hash), so
retrieval and generation experiments use identical sub-questions.
"""
from __future__ import annotations

import hashlib
import re
import json
import logging
from functools import lru_cache

from src.config import DATA_DIR, load_config

log = logging.getLogger(__name__)

CACHE_PATH = DATA_DIR / "eval" / "cache" / "decompositions.jsonl"
MAX_SUB_QUESTIONS = 4

PROMPT = """You split questions about SEC 10-K and 10-Q filings into the minimal list of \
self-contained lookup questions needed to answer them.
Rules:
- Each sub-question asks for ONE metric of ONE company for ONE explicit period, and repeats the \
company name and the period exactly as the question states them.
- Comparisons between periods or companies become one sub-question per value. Do not add a \
sub-question for the comparison or calculation itself.
- If the question needs only one lookup, or asks for an explanation from one filing, return it \
unchanged as the only item.
- Do not answer. Return JSON only: {{"sub_questions": ["...", "..."]}}

Question: Compare Costco's net sales in fiscal 2025 Q1 with fiscal 2025 Q2. Which quarter was higher?
{{"sub_questions": ["What were Costco's net sales in fiscal 2025 Q1?", "What were Costco's net sales in fiscal 2025 Q2?"]}}

Question: Compare Intel's and AMD's net income in their most recent annual reports (fiscal 2024 and fiscal 2024).
{{"sub_questions": ["What was Intel's net income in fiscal 2024?", "What was AMD's net income in fiscal 2024?"]}}

Question: How did Tesla's research and development expense for the three months ended March 31, 2025 compare with the three months ended March 31, 2024?
{{"sub_questions": ["What was Tesla's research and development expense for the three months ended March 31, 2025?", "What was Tesla's research and development expense for the three months ended March 31, 2024?"]}}

Question: What risks did Tesla describe about its supply chain in its fiscal 2024 annual report?
{{"sub_questions": ["What risks did Tesla describe about its supply chain in its fiscal 2024 annual report?"]}}

Question: {question}
"""


@lru_cache(maxsize=1)
def get_json_llm():
    from llama_index.llms.ollama import Ollama
    from src.generation import ollama_server
    cfg = load_config("generation")["llm"]
    return Ollama(model=cfg["model"], base_url=ollama_server.base_url(), temperature=0.0, json_mode=True,
                  context_window=cfg["context_window"], request_timeout=cfg["request_timeout_s"],
                  keep_alive=cfg["keep_alive"], thinking=False, additional_kwargs={"seed": 0, "num_predict": 256})


def _key(question: str) -> str:
    model = load_config("generation")["llm"]["model"]
    return hashlib.sha1(f"{model}\n{PROMPT}\n{question}".encode()).hexdigest()


@lru_cache(maxsize=1)
def _cache() -> dict[str, list[str]]:
    if not CACHE_PATH.exists():
        return {}
    with open(CACHE_PATH, encoding="utf-8") as fh:
        return {r["key"]: r["sub_questions"] for r in map(json.loads, fh)}


def parse(raw: str, question: str) -> list[str]:
    """Robust to small-model JSON slips; falls back to the original question."""
    try:
        data = json.loads(raw)
        subs = data.get("sub_questions") if isinstance(data, dict) else data
        subs = [s.strip() for s in subs if isinstance(s, str) and s.strip()]
    except (json.JSONDecodeError, AttributeError, TypeError):
        log.warning("unparseable decomposition %r for %r", raw[:200], question)
        subs = []
    subs = list(dict.fromkeys(subs))[:MAX_SUB_QUESTIONS]
    # One lookup never needs rewording, and the rewording can lose intent
    # ("What drove X's growth?" came back as "What was X?"): keep the original.
    return subs if len(subs) > 1 else [question]


# Router: only questions with a comparison / multi-period cue go to the LLM. On eval v1
# this agreed with the LLM's own decisions on all 235 questions (72 split, all cued; none
# of 164 single lookups cued) and removes the ~2 s LLM call from single-fact questions.
# Trend wording is included although the eval set has none.
MULTI_FACT_CUES = re.compile(
    r"\b(compare[ds]?|comparison|versus|vs\.?|than|change[ds]? from|between|both|relative to|"
    r"a year earlier|year[- ]over[- ]year|trend(ed|s)?|over the (last|past)|each of|and \w+'s)\b", re.I)


def needs_decomposition(question: str) -> bool:
    return bool(MULTI_FACT_CUES.search(question))


def decompose(question: str) -> list[str]:
    if not needs_decomposition(question):
        return [question]
    key = _key(question)
    cache = _cache()
    if key in cache:
        return cache[key]
    raw = get_json_llm().complete(PROMPT.format(question=question)).text
    subs = parse(raw, question)
    cache[key] = subs
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"key": key, "question": question, "sub_questions": subs}, ensure_ascii=False) + "\n")
    return subs
