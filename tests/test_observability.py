from datetime import datetime, timedelta

import pandas as pd

from src.observability.latency_report import breakdown
from src.observability.tracing import span


def _row(trace, sid, parent, name, kind, start_ms, dur_ms, inp=None):
    t0 = datetime(2026, 1, 1)
    return {"trace": trace, "span": sid, "parent": parent, "name": name, "kind": kind,
            "start": t0 + timedelta(milliseconds=start_ms), "end": t0 + timedelta(milliseconds=start_ms + dur_ms),
            "input": inp, "ms": float(dur_ms)}


def test_breakdown_sums_stages_without_double_counting_nested_llm_spans():
    df = pd.DataFrame([
        _row("t", "root", None, "rag.answer", "CHAIN", 0, 10_000, "Q?"),
        _row("t", "d", "root", "query.decompose", "CHAIN", 0, 1_000),
        _row("t", "d_llm", "d", "Ollama.chat", "LLM", 10, 900),           # LLM inside decompose: counted as decompose
        _row("t", "p", "root", "retrieve.candidate_pool", "RETRIEVER", 1_000, 100),
        _row("t", "r", "root", "rerank.cross_encoder", "RERANKER", 1_100, 400),
        _row("t", "llm", "root", "Ollama.predict", "LLM", 1_500, 8_000),
        _row("t", "llm2", "llm", "Ollama.chat", "LLM", 1_510, 7_900),      # nested LLM span: not counted again
        _row("t", "cv", "root", "answer.calc_and_verify", "TOOL", 9_500, 5),
    ])
    row = breakdown(df, last=1).iloc[0]
    assert (row.decompose, row.pool, row.rerank, row.calc_verify) == (1000, 100, 400, 5)
    assert row.llm == 8000              # the decomposition's LLM call belongs to "decompose"
    assert row.total_ms == 10_000
    assert row.other == 10_000 - (1000 + 100 + 400 + 8000 + 5)   # stages never exceed the total


def test_span_is_a_noop_without_setup():
    with span("x", "CHAIN", input_value={"a": 1}, extra=None) as s:
        s.set_attribute("k", "v")   # no exporter configured: must not raise
