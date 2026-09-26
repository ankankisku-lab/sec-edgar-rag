"""Phase 19: OpenTelemetry tracing into Arize Phoenix (local, open source).

    setup_tracing()  -> registers Phoenix as the OTLP exporter and instruments LlamaIndex
                        (retrievers, node postprocessors / reranker, synthesizer, LLM calls
                        are traced automatically by OpenInference).
    span(...)        -> context manager for the custom steps LlamaIndex can't see: the
                        decomposition router, BM25 pool vs rerank timing, <calc> and
                        number verification. Uses OpenInference span kinds so Phoenix
                        renders them as CHAIN / RETRIEVER / RERANKER spans.

When setup_tracing() was never called, OpenTelemetry's no-op tracer is used, so
evaluation runs pay nothing for the instrumentation.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace

from src.config import load_config

_TRACER = trace.get_tracer("sec-edgar-rag")
_SETUP_DONE = False


def setup_tracing() -> bool:
    """Idempotent; returns True if tracing is active."""
    global _SETUP_DONE
    cfg = load_config("observability")["tracing"]
    if _SETUP_DONE or not cfg.get("enabled", False):
        return _SETUP_DONE
    from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
    # Plain OpenTelemetry SDK instead of phoenix.otel.register(): register() 0.17.1 reads a private
    # exporter attribute (_headers) that opentelemetry-exporter 1.45 no longer has. Phoenix groups
    # spans by the `openinference.project.name` resource attribute.
    from openinference.instrumentation import TraceConfig
    from opentelemetry.sdk.trace import SpanLimits
    # OpenInference records every retrieved document as span attributes (60 BM25 candidates x
    # text/score/metadata): OTel's default cap of 128 attributes per span silently dropped most of
    # them. Long values (full chunk texts) are truncated to keep traces a sensible size.
    limits = SpanLimits(max_span_attributes=cfg.get("max_span_attributes", 2048),
                        max_span_attribute_length=cfg.get("max_attribute_length", 4000))
    provider = TracerProvider(resource=Resource.create({"openinference.project.name": cfg["project_name"]}),
                              span_limits=limits)
    exporter = OTLPSpanExporter(endpoint=cfg["endpoint"])
    provider.add_span_processor(BatchSpanProcessor(exporter) if cfg.get("batch", True) else SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    import logging
    logging.getLogger("opentelemetry.attributes").setLevel(logging.ERROR)  # expected truncation notices
    # Embedding vectors (384 floats per text) are useless in a trace viewer and bloat storage.
    LlamaIndexInstrumentor().instrument(tracer_provider=provider, config=TraceConfig(hide_embedding_vectors=True))
    _SETUP_DONE = True
    return True


def _value(v: Any) -> Any:
    if isinstance(v, (str, bool, int, float)):
        return v
    return json.dumps(v, ensure_ascii=False, default=str)


@contextmanager
def span(name: str, kind: str = "CHAIN", input_value: Any = None, **attributes: Any):
    """A traced step. `kind` is an OpenInference span kind (CHAIN, RETRIEVER, RERANKER, LLM, TOOL)."""
    with _TRACER.start_as_current_span(name) as s:
        s.set_attribute("openinference.span.kind", kind)
        if input_value is not None:
            s.set_attribute("input.value", _value(input_value))
        for k, v in attributes.items():
            if v is not None:
                s.set_attribute(k, _value(v))
        yield s


def set_output(s, value: Any, **attributes: Any) -> None:
    s.set_attribute("output.value", _value(value))
    for k, v in attributes.items():
        if v is not None:
            s.set_attribute(k, _value(v))


def flush(timeout_ms: int = 10_000) -> None:
    provider = trace.get_tracer_provider()
    if hasattr(provider, "force_flush"):
        provider.force_flush(timeout_ms)
