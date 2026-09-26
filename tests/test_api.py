"""API contract tests with the pipeline stubbed out (no GPU, Qdrant or Ollama needed)."""
import asyncio

import pytest
from fastapi.testclient import TestClient

import src.api.main as api
import src.generation.generator as generator


def fake_answer(question, decompose):
    return {"question": question, "sub_questions": [question], "contexts": ["Source 1:\n..."],
            "answer": "Operating income was $28,202 million [1].",
            "sources": [{"n": 1, "node_id": "x", "filing_id": "AAPL_10Q_2025-06-28", "form": "10-Q",
                         "period": "fiscal 2025 Q3", "section": "Item 1", "url": "https://sec.gov/x", "score": 7.1}],
            "calculations": [], "unverified_numbers": [], "latency_ms": 10,
            "latency_breakdown_ms": {"retrieval": 4, "generation": 6}, "trace_id": "ab" * 16}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, "_warm_up", lambda: None)            # skip model loading
    monkeypatch.setattr(generator, "answer", fake_answer)
    api.STATE.__init__()
    with TestClient(api.app) as c:
        yield c


def test_query_returns_answer_sources_and_latency(client):
    r = client.post("/query", json={"question": "What was Apple's operating income in fiscal 2025 Q3?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"].startswith("Operating income")
    assert body["sources"][0]["filing_id"] == "AAPL_10Q_2025-06-28"
    assert set(body["latency_ms"]) == {"total", "retrieval", "generation"}
    assert body["trace_id"] == "ab" * 16 and len(body["request_id"]) == 12


def test_validation_rejects_empty_and_oversized_questions(client):
    assert client.post("/query", json={"question": ""}).status_code == 422
    assert client.post("/query", json={"question": "x" * 5000}).status_code == 422


def test_metrics_count_requests(client):
    client.post("/query", json={"question": "What was Apple's revenue?"})
    m = client.get("/metrics").json()
    assert m["requests_total"] == 1 and m["errors_total"] == 0 and m["unverified_number_rate"] == 0.0


def test_not_ready_and_busy_return_503(client):
    api.STATE.ready = False
    assert client.post("/query", json={"question": "What was Apple's revenue?"}).status_code == 503
    api.STATE.ready = True
    api.STATE.waiting = api.CFG["pipeline"]["max_queue"]
    r = client.post("/query", json={"question": "What was Apple's revenue?"})
    assert r.status_code == 503 and "busy" in r.json()["detail"]


def test_pipeline_error_is_500_and_counted(client, monkeypatch):
    def boom(q, d):
        raise RuntimeError("gpu fell over")
    monkeypatch.setattr(generator, "answer", boom)
    r = client.post("/query", json={"question": "What was Apple's revenue?"})
    assert r.status_code == 500 and "RuntimeError" in r.json()["detail"]
    assert client.get("/metrics").json()["errors_total"] == 1
    assert api.STATE.lock.locked() is False   # the GPU lock is always released


def test_chat_page_is_served_and_calls_the_api(client):
    r = client.get("/")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert 'fetch("/query"' in r.text and 'fetch("/health"' in r.text
    assert "/" not in client.get("/openapi.json").json()["paths"]
