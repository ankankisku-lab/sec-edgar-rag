"""API contract tests with the pipeline stubbed out (no GPU, Qdrant or Ollama needed)."""
import asyncio

import pytest
from fastapi.testclient import TestClient

import src.api.main as api
import src.generation.generator as generator


CONTEXT = ("Source 1:\ncompany: Apple\nperiod: fiscal 2025 Q3 (quarter ended June 28, 2025)\n\n"
           "|  | Three Months Ended |  |\n|---|---|---|\n|  | June 28, 2025 | June 29, 2024 |\n"
           "| Operating income | $28,202 | $25,352 |\n")


def fake_answer(question, decompose):
    return {"question": question, "sub_questions": [question], "contexts": [CONTEXT],
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


def test_citations_point_to_the_quoted_cell(client):
    body = client.post("/query", json={"question": "What was Apple's operating income in fiscal 2025 Q3?"}).json()
    passage = body["sources"][0]["passage"]
    assert passage["header"]["company"] == "Apple" and passage["blocks"][0]["type"] == "table"
    [num] = body["numbers"]
    assert num["token"] == "$28,202" and num["status"] == "cell" and num["in_cited_source"]
    assert body["answer"][num["start"]:num["end"]] == "$28,202"
    cell = num["cells"][0]
    assert (cell["source"], cell["label"], cell["column"]) == (1, "Operating income", "Three Months Ended · June 28, 2025")
    assert passage["blocks"][cell["block"]]["rows"][cell["row"]][cell["col"]] == "$28,202"


def test_answer_survives_a_broken_filing_locator(client, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "src.generation.filing_locator", None)   # import now fails
    r = client.post("/query", json={"question": "What was Apple's operating income in fiscal 2025 Q3?"})
    assert r.status_code == 200 and r.json()["numbers"][0]["status"] == "cell"


def test_original_filing_is_served_without_scripts(client, monkeypatch):
    import src.generation.filing_locator as F
    monkeypatch.setattr(F, "render_filing", lambda fid, node, mark: "<html><body>filing</body></html>" if fid == "AAPL_10Q_2025-06-28" else None)
    r = client.get("/filing/AAPL_10Q_2025-06-28?node=12&mark=$28,202")
    assert r.status_code == 200 and "script-src" not in r.headers["content-security-policy"]
    assert r.headers["content-security-policy"].startswith("default-src 'none'")
    assert client.get("/filing/UNKNOWN_10Q_2025-06-28").status_code == 404
    assert client.get("/filing/AAPL_10Q_2025-06-28?node=-1").status_code == 422


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
