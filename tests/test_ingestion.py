import pytest

from src.ingestion.checkpoint import Checkpoint
from src.ingestion.downloader import ValidationError, validate
from src.ingestion.metadata import select_filings


def _submissions(rows):
    keys = ["form", "filingDate", "reportDate", "accessionNumber", "primaryDocument"]
    return {"filings": {"recent": {k: [r[i] for r in rows] for i, k in enumerate(keys)}, "files": []}}


def test_select_filings_takes_latest_n_per_form_and_excludes_others():
    subs = _submissions([
        ("10-Q", "2025-05-01", "2025-03-31", "a1", "q1.htm"),
        ("10-K", "2025-02-01", "2024-12-31", "a2", "k1.htm"),
        ("10-Q", "2025-08-01", "2025-06-30", "a3", "q2.htm"),
        ("10-K/A", "2025-03-01", "2024-12-31", "a4", "ka.htm"),
        ("4", "2025-08-02", "", "a5", "f4.xml"),
        ("10-K", "2024-02-01", "2023-12-31", "a6", "k0.htm"),
        ("10-Q", "2024-11-01", "2024-09-30", "a7", "q0.htm"),
    ])
    picked = select_filings(None, subs, {"10-K": 1, "10-Q": 2}, refresh=False)
    assert [p["accessionNumber"] for p in picked] == ["a3", "a1", "a2"]


def test_checkpoint_roundtrip_and_attempt_count(tmp_path):
    path = tmp_path / "ck.json"
    ck = Checkpoint(path)
    ck.mark("X", "failed", error="boom")
    ck.mark("X", "done", error=None, bytes=10)
    reloaded = Checkpoint(path)
    assert reloaded.is_done("X")
    assert reloaded.get("X")["attempts"] == 2
    assert reloaded.summary() == {"done": 1}


def test_validate_rejects_error_pages_and_wrong_documents():
    good = b"<html><body>FORM 10-K annual report" + b"x" * 30_000 + b"</body></html>"
    validate(good, "10-K", 20_000)
    with pytest.raises(ValidationError, match="too small"):
        validate(b"<html>10-K</html>", "10-K", 20_000)
    with pytest.raises(ValidationError, match="never mentions"):
        validate(good, "10-Q", 20_000)
    with pytest.raises(ValidationError, match="not an HTML"):
        validate(b"%PDF" + b"10-K" * 10_000, "10-K", 20_000)


def test_read_dei_handles_nested_ix_tags():
    from src.ingestion.verify import read_dei
    doc = ('<ix:nonNumeric name="dei:DocumentPeriodEndDate" format="x"><ix:nonNumeric '
           'name="dei:CurrentFiscalYearEndDate">January&#160;25</ix:nonNumeric>, 2026</ix:nonNumeric>'
           '<ix:nonNumeric name="dei:DocumentType"><span>10-K</span></ix:nonNumeric>')
    dei = read_dei(doc)
    assert dei["DocumentPeriodEndDate"] == "January 25, 2026"
    assert dei["DocumentType"] == "10-K"
