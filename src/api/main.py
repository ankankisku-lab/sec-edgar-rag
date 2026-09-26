"""Phase 20: FastAPI service for the SEC-EDGAR RAG pipeline.

    POST /query    question -> grounded answer with citations, calculations, unverified
                   numbers, sub-questions, latency breakdown and the Phoenix trace id
    GET  /health   Qdrant, Ollama and model readiness (503 until everything is up)
    GET  /metrics  request counts, errors, queue, latency percentiles, refusal and
                   unverified-number rates over the recent window

One GPU runs the embedder, reranker and LLM, so pipeline calls are serialised with an
asyncio lock and executed in a worker thread: /health and /metrics stay responsive while
an answer is being generated, and at most `max_queue` requests may wait (else 503).

Startup order matters on 4 GB VRAM: the embedder, BM25 and reranker are loaded and
exercised first, then Ollama starts and loads the LLM, so Ollama sizes its GPU layers
around their steady-state memory (see README, Phase 11).

Run (single worker -- one GPU):
    .venv\\Scripts\\python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from typing import Any

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from src.config import load_config

log = logging.getLogger("sec_rag.api")
CFG = load_config("api")
REFUSAL = "do not contain this information"


# ------------------------------------------------------------------ schemas
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=CFG["pipeline"]["question_max_chars"],
                          examples=["What was Apple's operating income in fiscal 2025 Q3?"])
    decompose: bool | None = Field(None, description="split comparisons into sub-questions (default from config)")


class Source(BaseModel):
    n: int
    filing_id: str | None
    form: str | None
    period: str | None
    section: str | None
    url: str | None
    score: float


class QueryResponse(BaseModel):
    request_id: str
    question: str
    answer: str
    sub_questions: list[str]
    sources: list[Source]
    calculations: list[dict[str, Any]]
    unverified_numbers: list[str]
    latency_ms: dict[str, int]
    trace_id: str | None


# ------------------------------------------------------------------ state
class State:
    def __init__(self):
        self.ready = False
        self.startup_error: str | None = None
        self.ollama_proc = None
        self.lock = asyncio.Lock()
        self.waiting = 0
        self.requests = self.errors = self.rejected = 0
        self.recent: deque = deque(maxlen=CFG["metrics"]["window"])  # (total_ms, refused, unverified)
        self.started_at = time.time()


STATE = State()


def _warm_up() -> None:
    """Load models in a VRAM-friendly order, then the LLM (see module docstring)."""
    from src.observability.tracing import setup_tracing
    setup_tracing()
    from src.retrieval.reranker import retrieve_reranked
    retrieve_reranked("warm-up: what was Apple's net income?")     # BM25 + reranker on the GPU
    from src.embeddings.embedder import embed_query
    embed_query("warm-up")                                          # embedder (dense/HyDE paths)
    from src.generation import ollama_server
    if not ollama_server.is_up():
        STATE.ollama_proc = ollama_server.start("q8_0")
    from src.generation.generator import get_llm
    get_llm().complete("Reply with OK.")                           # load the LLM into memory


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await run_in_threadpool(_warm_up)
        STATE.ready = True
        log.info("pipeline ready")
    except Exception as e:  # keep serving /health so the failure is visible
        STATE.startup_error = f"{type(e).__name__}: {e}"
        log.exception("startup failed")
    yield
    if STATE.ollama_proc is not None:
        from src.generation import ollama_server
        ollama_server.stop(STATE.ollama_proc)


app = FastAPI(title="SEC-EDGAR Financial RAG", version="1.0", lifespan=lifespan,
              description="Grounded answers over Nasdaq-100 10-K/10-Q filings with verified numbers.")


# ------------------------------------------------------------------ endpoints
@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    if not STATE.ready:
        raise HTTPException(503, detail=STATE.startup_error or "pipeline is still starting")
    if STATE.waiting >= CFG["pipeline"]["max_queue"]:
        STATE.rejected += 1
        raise HTTPException(503, detail="busy: too many queued requests, retry later")
    decompose = CFG["pipeline"]["decompose_default"] if req.decompose is None else req.decompose
    request_id = uuid.uuid4().hex[:12]
    STATE.waiting += 1
    try:
        await STATE.lock.acquire()  # one pipeline run at a time on the single GPU
    finally:
        STATE.waiting -= 1          # also on cancellation while waiting, so the queue never leaks
    try:
        STATE.requests += 1
        t0 = time.perf_counter()
        from src.generation.generator import answer
        try:
            result = await asyncio.wait_for(run_in_threadpool(answer, req.question, decompose),
                                            timeout=CFG["pipeline"]["request_timeout_s"])
        except asyncio.TimeoutError:
            STATE.errors += 1
            raise HTTPException(504, detail="pipeline timed out")
        except Exception as e:
            STATE.errors += 1
            log.exception("request %s failed", request_id)
            raise HTTPException(500, detail=f"internal error (request {request_id}): {type(e).__name__}")
        total_ms = round((time.perf_counter() - t0) * 1000)
    finally:
        STATE.lock.release()
    refused = REFUSAL in result["answer"].lower()
    STATE.recent.append((total_ms, refused, bool(result["unverified_numbers"])))
    return QueryResponse(
        request_id=request_id, question=req.question, answer=result["answer"],
        sub_questions=result["sub_questions"],
        sources=[Source(**{k: s.get(k) for k in Source.model_fields}) for s in result["sources"]],
        calculations=result["calculations"], unverified_numbers=result["unverified_numbers"],
        latency_ms={"total": total_ms, **result.get("latency_breakdown_ms", {})}, trace_id=result.get("trace_id"),
    )


@app.get("/health")
async def health() -> JSONResponse:
    checks: dict[str, Any] = {"pipeline_ready": STATE.ready}
    if STATE.startup_error:
        checks["startup_error"] = STATE.startup_error
    qcfg = load_config("qdrant")
    try:
        r = await run_in_threadpool(requests.get, f"{qcfg['url']}/collections/{qcfg['collection']}", timeout=5)
        info = r.json()["result"]
        checks["qdrant"] = {"ok": r.ok, "points": info.get("points_count"), "status": info.get("status")}
    except Exception as e:
        checks["qdrant"] = {"ok": False, "error": type(e).__name__}
    from src.generation import ollama_server
    checks["ollama"] = {"ok": await run_in_threadpool(ollama_server.is_up)}
    try:
        tcfg = load_config("observability")["tracing"]
        base = tcfg["endpoint"].rsplit("/v1/traces", 1)[0]
        pr = await run_in_threadpool(requests.get, f"{base}/healthz", timeout=3)
        checks["phoenix"] = {"ok": pr.ok, "required": False}
    except Exception:
        checks["phoenix"] = {"ok": False, "required": False}
    healthy = STATE.ready and checks["qdrant"].get("ok") and checks["ollama"]["ok"]
    return JSONResponse({"status": "ok" if healthy else "unavailable", **checks}, status_code=200 if healthy else 503)


@app.get("/metrics")
async def metrics() -> dict:
    recent = list(STATE.recent)
    lat = sorted(r[0] for r in recent)

    def q(p):
        return lat[min(len(lat) - 1, int(p * (len(lat) - 1) + 0.5))] if lat else None

    return {
        "uptime_s": round(time.time() - STATE.started_at),
        "requests_total": STATE.requests, "errors_total": STATE.errors, "rejected_busy_total": STATE.rejected,
        "in_flight": int(STATE.lock.locked()), "queued": STATE.waiting,
        "window": len(recent), "latency_ms_p50": q(0.5), "latency_ms_p95": q(0.95),
        "refusal_rate": round(sum(r[1] for r in recent) / len(recent), 3) if recent else None,
        "unverified_number_rate": round(sum(r[2] for r in recent) / len(recent), 3) if recent else None,
    }


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception) -> JSONResponse:
    log.exception("unhandled error on %s", request.url.path)
    return JSONResponse({"detail": "internal error"}, status_code=500)
