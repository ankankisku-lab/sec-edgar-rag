"""Find a cited number in the original SEC filing and serve the filing with it highlighted.

The citation view (source_view.py) traces each number of an answer to a cell or sentence of the
parsed passage. This module finds the same spot in the original filing HTML on disk
(data/raw/<TICKER>/<filing_id>.htm), without re-parsing the corpus:

- numbers in tables are matched to their iXBRL fact (<ix:nonFraction id="f-91">28,202</...>):
  same value, ranked by row label, the fact's reporting period against the column header, and
  the table caption. The fact id is also a working anchor on sec.gov (document_url#f-91).
- untagged table cells are matched by value and row label; numbers quoted from prose by the
  words around them.

render_filing() returns the original HTML with one element marked as the URL target (#sec-rag-
target) and a style sheet that highlights it via CSS :target -- no script is added.

Usage (how often a citation resolves to exactly one place, on saved generation runs):
    python -m src.generation.filing_locator --runs G5 G6 G7 G8
"""
from __future__ import annotations

import argparse
import json
import re
from bisect import bisect_right
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path

import lxml.html
from lxml import etree

from src.config import RAW_DIR
from src.generation.numbers import NUM_RE, _value
from src.generation.source_view import cell_value, norm

FILING_ID_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}_10[KQ]_\d{4}-\d{2}-\d{2}$")
DATE_RE = re.compile(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+(\d{1,2}),?\s*(\d{4})")
ITEM_RE = re.compile(r"^\s*item\s+(\d{1,2}[a-z]?)\b", re.I)
DURATION_DAYS = {"three": 91, "six": 182, "nine": 273, "twelve": 365, "year": 365, "fiscal year": 365}
TARGET = "sec-rag-target"


# ---------------------------------------------------------------------------- the filing on disk
def raw_path(filing_id: str) -> Path | None:
    """The original filing for an id like AAPL_10Q_2025-06-28, or None (ids are validated)."""
    if not FILING_ID_RE.match(filing_id or ""):
        return None
    path = (RAW_DIR / filing_id.split("_")[0] / f"{filing_id}.htm").resolve()
    return path if path.is_file() and RAW_DIR.resolve() in path.parents else None


def filing_meta(filing_id: str) -> dict:
    path = raw_path(filing_id)
    side = path.with_suffix(".json") if path else None
    return json.loads(side.read_text(encoding="utf-8")) if side and side.is_file() else {}


def _parse(path: Path):
    return lxml.html.parse(str(path))


def _text(el) -> str:
    return " ".join(el.text_content().split())


@lru_cache(maxsize=8)  # an answer's sources usually span 1-4 filings
def _index(filing_id: str) -> dict | None:
    """Parsed tree plus the iXBRL facts and reporting periods of one filing (cached)."""
    path = raw_path(filing_id)
    if path is None:
        return None
    tree = _parse(path)
    root = tree.getroot()
    periods = {}
    for ctx in root.iter("xbrli:context"):
        p = {el.tag.split(":")[-1]: (el.text or "").strip() for el in ctx.iter() if isinstance(el.tag, str)
             and el.tag in ("xbrli:startdate", "xbrli:enddate", "xbrli:instant")}
        # dimensions, e.g. (us-gaap:StatementBusinessSegmentsAxis, msft:ProductivityAndBusinessProcessesMember)
        p["dims"] = [(m.get("dimension", ""), (m.text or "").strip()) for m in ctx.iter("xbrldi:explicitmember")]
        periods[ctx.get("id")] = p
    facts = []
    for el in root.iter("ix:nonfraction"):
        v = _value(_text(el))
        if v is not None:
            facts.append((el, abs(v)))
    # Elements are referenced by document-order position: deterministic for the same file and
    # parser, and (unlike an XPath) safe to accept back from a URL. iXBRL tags such as
    # "ix:nonfraction" would also read as undeclared namespace prefixes in an XPath.
    # Untagged numeric cells: MD&A and other summary tables repeat statement figures without iXBRL.
    cells = []
    for td in root.iter("td"):
        if next(td.iter("ix:nonfraction"), None) is None:
            t = _text(td)
            if len(t) < 40 and (v := cell_value(t)) is not None:
                cells.append((td, v))
    elements = list(root.iter())
    order = {el: i for i, el in enumerate(elements)}
    # Positions of "Item 7." style headings, to tell which Item a candidate sits in (10-Ks repeat
    # some paragraphs, e.g. buybacks in both Item 5 and Item 7).
    items = [(i, m.group(1).upper()) for i, el in enumerate(elements) if isinstance(el.tag, str)
             and len(t := _text(el)) < 150 and (m := ITEM_RE.match(t))]
    full, starts, owners, begin = _full_text(root)
    return {"tree": tree, "periods": periods, "facts": facts, "cells": cells, "elements": elements,
            "order": order, "items": items, "full": full, "starts": starts, "owners": owners, "begin": begin}


def _full_text(root) -> tuple[str, list[int], list, dict]:
    """The document's text with whitespace collapsed (as in _text(root)), built once, with the
    element that owns each piece and where each element begins -- so finding a phrase is a string
    search instead of a text_content() call per element."""
    parts, starts, owners, begin, length = [], [], [], {}, 0
    space = True  # the text so far ends with a space (or is empty)

    def add(s: str, owner) -> None:
        nonlocal length, space
        words = s.split()
        if not words:
            if s and not space:
                parts.append(" "); length += 1; space = True
            return
        if s[0].isspace() and not space:
            parts.append(" "); length += 1
        starts.append(length); owners.append(owner)
        chunk = " ".join(words)
        parts.append(chunk); length += len(chunk)
        space = False
        if s[-1].isspace():
            parts.append(" "); length += 1; space = True

    for event, el in etree.iterwalk(root, events=("start", "end")):
        if event == "start":
            begin[el] = length
            if isinstance(el.tag, str) and el.tag not in ("script", "style") and el.text:
                add(el.text, el)
        elif el.tail and el.getparent() is not None:
            add(el.tail, el.getparent())
    return "".join(parts), starts, owners, begin


def _item_of(idx: dict, el) -> str | None:
    pos = idx["order"].get(el, -1)
    before = [code for i, code in idx["items"] if i <= pos]
    return before[-1] if before else None


# ---------------------------------------------------------------------------- scoring helpers
def _row_label(el) -> str:
    tr = next((a for a in el.iterancestors("tr")), None)
    if tr is None:
        return ""
    for cell in tr.iter("td", "th"):
        t = _text(cell)
        if re.search(r"[A-Za-z]", t):
            return norm(t)
    return ""


def _label_score(found: str, wanted: str) -> int:
    if not found or not wanted:
        return 0
    if found == wanted:
        return 3
    return 2 if found.startswith(wanted) or wanted.startswith(found) else 0


def _to_date(s: str) -> date | None:
    try:
        return date.fromisoformat(s[:10])
    except ValueError:
        return None


def _period_score(period: dict, column: str) -> int:
    """+1 if the fact's end date is a date in the column header, +1 if its length matches
    'Three Months Ended' / 'Year Ended' etc."""
    if not period or not column:
        return 0
    end = _to_date(period.get("enddate") or period.get("instant") or "")
    score = 0
    dates = list(DATE_RE.finditer(column))
    for m in dates:
        try:
            if end and datetime.strptime(f"{m.group(1)} {m.group(2)} {m.group(3)}", "%b %d %Y").date() == end:
                score += 1
                break
        except ValueError:
            continue
    if not dates and end and re.search(rf"\b{end.year}\b", column):   # a year-only header ("2025")
        score += 1
    start = _to_date(period.get("startdate", ""))
    if start and end:
        days = (end - start).days
        col = column.lower()
        expected = next((d for k, d in DURATION_DAYS.items() if re.search(rf"\b{k}\b", col)), None)
        if expected and abs(days - expected) <= 20:
            score += 1
    return score


def _caption_score(idx: dict, el, caption: str) -> int:
    """+1 if the table's caption words appear in the 800 characters of text before the table."""
    words = norm(caption.split("|")[0]) if caption else ""
    table = next((a for a in el.iterancestors("table")), None)
    if len(words) < 8 or table is None:
        return 0
    at = idx["begin"].get(table, 0)
    return int(words[:60] in norm(idx["full"][max(0, at - 800):at]))


def _table_overlap(el, table_values: set[float], cache: dict) -> float:
    """Share of the quoted passage table's numbers found in the HTML table around el: the same
    fact (e.g. net income) appears in several statements, and this says which one was quoted."""
    table = next((a for a in el.iterancestors("table")), None)
    if table is None or not table_values:
        return 0.0
    if table not in cache:
        cache[table] = {v for td in table.iter("td") if (v := cell_value(_text(td))) is not None}
    return len(table_values & cache[table]) / len(table_values)


def _best(scored: list[tuple[int, object]]) -> tuple[object | None, bool]:
    if not scored:
        return None, False
    scored.sort(key=lambda s: -s[0])
    unique = len(scored) == 1 or scored[0][0] > scored[1][0]
    return scored[0][1], unique


# ---------------------------------------------------------------------------- locating
def locate_cell(filing_id: str, value: float, label: str, column: str = "", caption: str = "",
                table_values: set[float] | None = None, item: str | None = None) -> dict | None:
    """The spot in the filing for a quoted table value: an iXBRL fact or an untagged cell (MD&A
    summary tables repeat statement figures untagged), ranked together by row label (0-30),
    overlap with the quoted table's numbers (0-20), the fact's reporting period vs the column
    header (0-6), the Item it sits in (0-2) and the table caption (0-1). fact_id is the best iXBRL
    fact for the value -- a working sec.gov anchor even when the quoted table is an untagged one."""
    idx = _index(filing_id)
    if idx is None:
        return None
    wanted, tables = norm(label), {}
    tv = table_values or set()

    def score(el, period_pts: int) -> int:
        return (10 * _label_score(_row_label(el), wanted) + round(20 * _table_overlap(el, tv, tables))
                + 3 * period_pts + 2 * (item is not None and _item_of(idx, el) == item) + _caption_score(idx, el, caption))

    facts = [(score(el, _period_score(idx["periods"].get(el.get("contextref"), {}), column)), el)
             for el, v in idx["facts"] if v == value]
    cells = [(score(td, 0), td) for td, v in idx["cells"] if v == value and _label_score(_row_label(td), wanted)]
    el, unique = _best(facts + cells)
    if el is None:
        return None
    fact, _ = _best(facts)
    tagged = el.tag == "ix:nonfraction"
    return {"node": idx["order"][el], "method": "ixbrl" if tagged else "table", "unique": unique,
            "fact_id": fact.get("id") if fact is not None else None,
            "concept": fact.get("name") if fact is not None else None,
            "segment": segment_note(idx, fact) if fact is not None else None}


SEGMENT_AXES = {"us-gaap:StatementBusinessSegmentsAxis", "srt:StatementGeographicalAxis"}


def _member_label(member: str) -> str:
    """msft:ProductivityAndBusinessProcessesMember -> Productivity and Business Processes"""
    name = member.split(":")[-1].removesuffix("Member")
    words = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name).split()
    return " ".join(w.lower() if w in ("And", "Of", "The", "For", "In") and i else w for i, w in enumerate(words))


def segment_note(idx: dict, fact) -> dict | None:
    """When a quoted value is one business segment's (or region's) figure and the filing reports a
    different company-wide figure for the same line item and period, name the segment and give the
    total -- e.g. Microsoft's Productivity and Business Processes operating income quoted as
    "operating income". Product/service breakdowns (Costco's net sales) are not segments."""
    ctx = idx["periods"].get(fact.get("contextref"), {})
    segments = [(axis, member) for axis, member in ctx.get("dims", []) if axis in SEGMENT_AXES]
    if not segments:
        return None
    period = (ctx.get("startdate"), ctx.get("enddate"), ctx.get("instant"))
    value, totals = _value(_text(fact)), []
    for el, _ in idx["facts"]:
        c = idx["periods"].get(el.get("contextref"), {})
        if el.get("name") == fact.get("name") and not c.get("dims") \
                and (c.get("startdate"), c.get("enddate"), c.get("instant")) == period:
            totals.append(el)
    if not totals or any(_value(_text(t)) == value for t in totals):
        return None  # no company-wide figure, or the segment is the whole company
    total = totals[0]
    sign = "-" if total.get("sign") == "-" else ""
    return {"member": _member_label(segments[0][1]), "axis": segments[0][0].split(":")[-1],
            "total": f"{sign}{_text(total)}", "total_fact_id": total.get("id")}


def locate_text(filing_id: str, token: str, before: str, after: str, item: str | None = None) -> dict | None:
    """The smallest element whose text contains the quoted number with the words around it
    (widest context first; a sentence repeated in another Item is told apart by the Item)."""
    idx = _index(filing_id)
    if idx is None:
        return None
    full, starts, owners = idx["full"], idx["starts"], idx["owners"]
    b, a = before.split(), after.split()
    phrases = [f"{' '.join(b[-k:])} {token} {' '.join(a[:k])}" for k in (8, 4)] + \
              [f"{' '.join(b[-4:])} {token}", f"{token} {' '.join(a[:4])}"]
    for phrase in phrases:
        phrase = " ".join(phrase.split())
        if len(phrase) < len(token) + 8:
            continue
        found, i = [], full.find(phrase)
        while i >= 0 and len(found) < 20:
            first = owners[max(0, bisect_right(starts, i) - 1)]
            last = owners[max(0, bisect_right(starts, i + len(phrase) - 1) - 1)]
            el = _common_ancestor(first, last)          # the smallest element holding the whole phrase
            if el is not None and el not in found:
                found.append(el)
            i = full.find(phrase, i + 1)
        if found:
            el, unique = _best([(int(item is not None and _item_of(idx, e) == item), e) for e in found])
            return {"node": idx["order"][el], "fact_id": None, "concept": None, "method": "text", "unique": unique}
    return None


def _common_ancestor(a, b):
    chain = {a, *a.iterancestors()}
    return b if b in chain else next((x for x in b.iterancestors() if x in chain), None)


def _around(text: str, token: str, n: int = 4) -> tuple[str, str] | None:
    """n words before and after the first standalone occurrence of token in text."""
    m = re.search(rf"(?<![\d,.]){re.escape(token)}(?![\d,]|\.\d)", text)
    if m is None:
        return None
    before = " ".join(text[:m.start()].split()[-n:])
    after = " ".join(text[m.end():].split()[:n])
    return before, after


def link_numbers(numbers: list[dict], passages: list[dict], filing_ids: list[str | None]) -> None:
    """Add a "filing" locator to every cell / text match of locate_numbers() output (in place)."""
    def item(p: dict) -> str | None:
        m = re.search(r"\bItem\s+(\d{1,2}[A-Z]?)\b", p["header"].get("heading_path", ""), re.I)
        return m.group(1).upper() if m else None

    for n in numbers:
        for c in n.get("cells", []):
            fid = filing_ids[c["source"] - 1] if c["source"] - 1 < len(filing_ids) else None
            if not fid:
                continue
            p = passages[c["source"] - 1]
            rows = p["blocks"][c["block"]]["rows"]
            value = cell_value(rows[c["row"]][c["col"]])
            if value is not None:
                table_values = {v for row in rows for cell in row[1:] if (v := cell_value(cell)) is not None}
                c["filing"] = locate_cell(fid, value, c["label"], c["column"], p["header"].get("table_caption", ""),
                                          table_values, item(p))
        for t in n.get("text", []):
            fid = filing_ids[t["source"] - 1] if t["source"] - 1 < len(filing_ids) else None
            ctx = _around(passages[t["source"] - 1]["blocks"][t["block"]]["text"], t["token"], n=8) if fid else None
            if ctx:
                t["filing"] = locate_text(fid, t["token"], *ctx, item(passages[t["source"] - 1]))


# ---------------------------------------------------------------------------- serving
STYLE = """
#%(t)s, .sec-rag-cell { background: #ffe066 !important; outline: 3px solid #e8590c; outline-offset: 1px;
  border-radius: 2px; scroll-margin-top: 45vh; }
tr.sec-rag-row > td { background: #fff8db !important; }
mark.sec-rag-mark { background: #ffe066; outline: 2px solid #e8590c; }
#sec-rag-banner { position: sticky; top: 0; z-index: 99999; background: #0f766e; color: #fff;
  font: 13px/1.45 system-ui, "Segoe UI", sans-serif; padding: 8px 14px; box-shadow: 0 2px 6px rgba(0,0,0,.25); }
#sec-rag-banner a { color: #fff; font-weight: 600; }
""" % {"t": TARGET}


def render_filing(filing_id: str, node: int | None = None, mark: str | None = None) -> str | None:
    """The original filing with element number `node` (document order) marked as #sec-rag-target,
    or None if the filing is unknown."""
    src = raw_path(filing_id)
    if src is None:
        return None
    tree = _parse(src)
    root = tree.getroot()
    note = ""
    if node is not None:
        el = next((e for i, e in enumerate(root.iter()) if i == node), None)  # before sanitising: same order as _index
        if el is not None and not isinstance(el.tag, str):   # a comment or processing instruction
            el = None
        if el is not None:
            el.set("id", TARGET)
            cell = el if el.tag in ("td", "th") else next(el.iterancestors("td", "th"), None)
            if cell is not None:
                cell.set("class", (cell.get("class", "") + " sec-rag-cell").strip())
                tr = next(cell.iterancestors("tr"), None)
                if tr is not None:
                    tr.set("class", (tr.get("class", "") + " sec-rag-row").strip())
            elif mark:
                _mark_token(el, mark)
            note = f" · highlighted: <b>{_esc(mark or _text(el)[:40])}</b>"
        else:
            note = " · the cited spot could not be found in this document"
    _sanitise(root)
    meta = filing_meta(filing_id)
    url = meta.get("document_url", "")
    head = root.find("head")
    if head is None:
        head = etree.SubElement(root, "head")
        root.insert(0, head)
    if url:
        head.insert(0, etree.fromstring(f'<base href="{_esc(url.rsplit("/", 1)[0])}/"/>'))
    style = etree.SubElement(head, "style")
    style.text = STYLE
    body = root.find("body")
    if body is not None:
        banner = lxml.html.fragment_fromstring(
            f'<div id="sec-rag-banner">Original filing · {_esc(meta.get("company", filing_id))} '
            f'{_esc(meta.get("form_type", ""))}, period ended {_esc(meta.get("period_of_report", ""))}{note}'
            + (f' · <a href="{_esc(url)}" target="_blank" rel="noopener">open on sec.gov ↗</a>' if url else "")
            + "</div>")
        body.insert(0, banner)
    return lxml.html.tostring(tree, encoding="unicode", method="html", doctype="<!DOCTYPE html>")


ACTIVE_TAGS = {"script", "noscript", "iframe", "frame", "frameset", "object", "embed", "applet", "link", "meta", "form"}


def _sanitise(root) -> None:
    """Remove active content before serving: every downloaded filing carries a script tag that
    sec.gov's bot protection injected at download time. (The CSP header blocks it too.)"""
    for el in list(root.iter()):
        if not isinstance(el.tag, str):
            continue
        if el.tag.lower() in ACTIVE_TAGS:
            el.drop_tree()
            continue
        for attr in list(el.attrib):
            value = el.attrib[attr].strip().lower()
            if attr.lower().startswith("on") or (attr.lower() in ("href", "src", "action") and value.startswith("javascript:")):
                del el.attrib[attr]


def _mark_token(el, token: str) -> None:
    """Wrap the first occurrence of token inside el's text in <mark class="sec-rag-mark">."""
    for node in [el, *el.iterdescendants()]:
        for attr in ("text", "tail"):
            s = getattr(node, attr) if not (node is el and attr == "tail") else None
            if not s or token not in s:
                continue
            i = s.index(token)
            mark = etree.Element("mark", {"class": "sec-rag-mark"})
            mark.text, mark.tail = token, s[i + len(token):]
            setattr(node, attr, s[:i])
            if attr == "text":
                node.insert(0, mark)
            else:
                node.addnext(mark)
            return


def _esc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;"))


# ---------------------------------------------------------------------------- measurement
def _filing_id_from_header(h: dict) -> str | None:
    """AAPL + 10-Q + 'fiscal 2025 Q3 (quarter ended June 28, 2025)' -> AAPL_10Q_2025-06-28."""
    m = DATE_RE.search(h.get("period", ""))
    if not m or not h.get("ticker"):
        return None
    try:
        d = datetime.strptime(f"{m.group(1)} {m.group(2)} {m.group(3)}", "%b %d %Y").date()
    except ValueError:
        return None
    return f"{h['ticker']}_{h.get('form_type', '').replace('-', '')}_{d.isoformat()}"


def main() -> None:
    from collections import Counter

    from src.evaluation.dataset import EVAL_DIR
    from src.generation.source_view import locate_numbers
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", default=["G5", "G6", "G7", "G8"])
    args = parser.parse_args()
    tally, period_check, same_table = Counter(), Counter(), Counter()
    for run in args.runs:
        for line in open(EVAL_DIR / "results" / f"{run}_ragas_input.jsonl", encoding="utf-8"):
            rec = json.loads(line)
            passages, numbers = locate_numbers(rec["response"], rec["retrieved_contexts"])
            fids = [_filing_id_from_header(p["header"]) for p in passages]
            link_numbers(numbers, passages, fids)
            for n in numbers:
                for kind in ("cells", "text"):
                    for c in n[kind][:1]:  # the match the UI shows first
                        f = c.get("filing")
                        key = "not found" if f is None else f"{f['method']}, {'unique' if f['unique'] else 'ambiguous'}"
                        tally[(kind, key)] += 1
                        if f and kind == "cells":
                            idx = _index(fids[c["source"] - 1])
                            el = idx["elements"][f["node"]]
                            rows = passages[c["source"] - 1]["blocks"][c["block"]]["rows"]
                            tv = {v for row in rows for cell in row[1:] if (v := cell_value(cell)) is not None}
                            same_table[_table_overlap(el, tv, {}) >= 0.8] += 1
                            if f["method"] == "ixbrl":
                                period_check[_period_score(idx["periods"].get(el.get("contextref"), {}),
                                                           c.get("column", "")) > 0] += 1
    total = sum(tally.values())
    print(f"{total} located numbers in {', '.join(args.runs)} (first match of each number):")
    for (kind, key), n in sorted(tally.items()):
        print(f"  {kind:5} {key:22} {n:4}  {n / total:6.1%}")
    print(f"table cells located in an HTML table holding >= 80% of the quoted table's numbers: "
          f"{same_table[True]}/{sum(same_table.values())}")
    print(f"iXBRL facts whose reporting period matches the column header: {period_check[True]}/{sum(period_check.values())}")


if __name__ == "__main__":
    main()
