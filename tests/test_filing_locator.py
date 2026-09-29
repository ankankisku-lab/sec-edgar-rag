import json

import pytest

import src.generation.filing_locator as F

FILING = """<html xmlns:ix="http://www.xbrl.org/2013/inlineXBRL"><head><title>10-Q</title>
<script type="text/javascript" src="/QQpw/injected-by-bot-protection"></script></head><body onload="x()">
<a href="javascript:alert(1)">bad link</a><iframe src="https://example.com"></iframe>
<div style="display:none"><ix:header><ix:resources>
<xbrli:context id="c-q3"><xbrli:period><xbrli:startDate>2025-03-30</xbrli:startDate><xbrli:endDate>2025-06-28</xbrli:endDate></xbrli:period></xbrli:context>
</ix:resources></ix:header></div>
<div><span>Item 1. Financial Statements</span></div>
<div>CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS</div>
<table>
<tr><td></td><td>Three Months Ended June 28, 2025</td></tr>
<tr><td>Net sales</td><td><ix:nonFraction contextRef="c-q3" name="us-gaap:Revenues" id="f-1">94,036</ix:nonFraction></td></tr>
<tr><td>Operating income</td><td><ix:nonFraction contextRef="c-q3" name="us-gaap:OperatingIncomeLoss" id="f-2">28,202</ix:nonFraction></td></tr>
</table>
<div><span>Item 2. Management's Discussion and Analysis</span></div>
<table>
<tr><td>Americas</td><td>9,800</td></tr>
<tr><td>Total operating income</td><td><ix:nonFraction contextRef="c-q3" name="us-gaap:OperatingIncomeLoss" id="f-9">28,202</ix:nonFraction></td></tr>
</table>
<p>Gross margin increased $1.9 billion or 15% driven by growth in Services.</p>
</body></html>"""


@pytest.fixture
def filing(tmp_path, monkeypatch):
    monkeypatch.setattr(F, "RAW_DIR", tmp_path)
    F._index.cache_clear()
    (tmp_path / "AAPL").mkdir()
    (tmp_path / "AAPL" / "AAPL_10Q_2025-06-28.htm").write_text(FILING, encoding="utf-8")
    (tmp_path / "AAPL" / "AAPL_10Q_2025-06-28.json").write_text(json.dumps(
        {"company": "Apple Inc.", "form_type": "10-Q", "period_of_report": "2025-06-28",
         "document_url": "https://www.sec.gov/Archives/edgar/data/320193/x/aapl-20250628.htm"}), encoding="utf-8")
    yield "AAPL_10Q_2025-06-28"
    F._index.cache_clear()


def test_filing_ids_are_validated(filing):
    assert F.raw_path(filing) is not None
    for bad in ("../secrets", "AAPL_10Q_2025-06-28/../../x", "aapl_10q_2025-06-28", "AAPL_8K_2025-06-28", ""):
        assert F.raw_path(bad) is None
    assert F.render_filing("../../etc/passwd") is None


def test_same_fact_in_two_tables_is_told_apart_by_row_label(filing):
    idx = F._index(filing)
    by_label = F.locate_cell(filing, 28202.0, "Operating income", "Three Months Ended · June 28, 2025")
    assert by_label["fact_id"] == "f-2" and by_label["method"] == "ixbrl" and by_label["unique"]
    segment = F.locate_cell(filing, 28202.0, "Total operating income", "", table_values={9800.0, 28202.0})
    assert segment["fact_id"] == "f-9" and segment["unique"]
    assert idx["elements"][segment["node"]].get("id") == "f-9"
    assert F._item_of(idx, idx["elements"][segment["node"]]) == "2"


def test_period_score_uses_the_column_header():
    q3 = {"startdate": "2025-03-30", "enddate": "2025-06-28"}
    assert F._period_score(q3, "Three Months Ended · June 28, 2025") == 2
    assert F._period_score(q3, "Nine Months Ended · June 28, 2025") == 1
    assert F._period_score(q3, "2025") == 1


def test_prose_numbers_and_rendering(filing):
    hit = F.locate_text(filing, "$1.9", "Gross margin increased", "billion or 15% driven")
    assert hit and hit["method"] == "text" and hit["unique"]
    html = F.render_filing(filing, hit["node"], "$1.9")
    assert 'id="sec-rag-target"' in html and '<mark class="sec-rag-mark">$1.9</mark>' in html
    assert '<base href="https://www.sec.gov/Archives/edgar/data/320193/x/">' in html
    low = html.lower()
    assert "<script" not in low and "<iframe" not in low and "onload" not in low and "javascript:" not in low
    cell = F.render_filing(filing, F.locate_cell(filing, 28202.0, "Operating income")["node"], "$28,202")
    assert "sec-rag-cell" in cell and "sec-rag-row" in cell and "highlighted: <b>$28,202</b>" in cell
