"""Source passages and number-to-cell tracing for citations.

The LLM sees each source as "Source N:", a metadata header (company, form, period, ...) and the
parent section, whose tables are markdown grids. parse_source() turns that text back into
structure: the header plus text and table blocks. locate_numbers() traces every number of an
answer to the table cell or sentence it was quoted from -- in the source its citation points to --
so the chat UI can show the passage behind a citation with the exact cell highlighted. The
Phase 17 evaluation uses the same parser to classify wrong-cell numbers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from src.generation.numbers import NUM_RE, _value, number_spans

SOURCE_RE = re.compile(r"^Source \d+:\s*$")
HEADER_RE = re.compile(r"^([a-z_]+):\s*(.*)$")
SEP_RE = re.compile(r"^:?-{2,}:?$")
CITE_RE = re.compile(r"(?:\[\d+\])+")
SENTENCE_END_RE = re.compile(r"\.(?:\s|$)|\n")


@dataclass(frozen=True)
class Cell:
    src: int      # index of the source (0-based)
    block: int    # block index inside the parsed source
    table: int    # table number inside the source
    row: int
    col: int
    label: str    # normalised row label (first column)
    value: float  # absolute value


def norm(label: str) -> str:
    return re.sub(r"\s+", " ", label.replace("*", "")).strip(" :").lower()


def matches(stated: float, source: float) -> bool:
    """The stated number is this source value, exactly or rounded from millions to billions."""
    return stated == source or (source >= 1000 and any(round(source / 1000, d) == stated for d in (1, 2)))


def cell_value(text: str) -> float | None:
    toks = NUM_RE.findall(text)
    v = _value(toks[0]) if len(toks) == 1 else None
    return None if v is None else abs(v)


def parse_source(text: str) -> dict:
    """{"header": {key: value}, "blocks": [{"type": "text", "text"} | {"type": "table", "rows": [[cell, ...]]}]}"""
    lines = text.splitlines()
    i = 1 if lines and SOURCE_RE.match(lines[0].strip()) else 0
    header = {}
    while i < len(lines) and (m := HEADER_RE.match(lines[i].strip())):
        header[m.group(1)] = m.group(2)
        i += 1
    blocks, buf, rows = [], [], None
    for line in lines[i:] + [""]:
        s = line.strip()
        if s.startswith("|"):
            if buf and "\n".join(buf).strip():
                blocks.append({"type": "text", "text": "\n".join(buf).strip()})
            buf = []
            rows = [] if rows is None else rows
            parts = [p.strip() for p in s.strip("|").split("|")]
            if not all(SEP_RE.match(p) or not p for p in parts):   # skip separator and empty rows
                rows.append(parts)
            continue
        if rows is not None:
            blocks.append({"type": "table", "rows": rows})
            rows = None
        buf.append(line)
    if "\n".join(buf).strip():
        blocks.append({"type": "text", "text": "\n".join(buf).strip()})
    return {"header": header, "blocks": blocks}


def table_cells(src: int, parsed: dict) -> list[Cell]:
    cells, table = [], -1
    for b, block in enumerate(parsed["blocks"]):
        if block["type"] != "table":
            continue
        table += 1
        for r, row in enumerate(block["rows"]):
            label = norm(row[0]) if row else ""
            for c, text in enumerate(row[1:], 1):
                v = cell_value(text)
                if v is not None:
                    cells.append(Cell(src, b, table, r, c, label, v))
    return cells


def text_numbers(parsed: dict) -> list[tuple[int, str, float]]:
    """(block, token, absolute value) of every number in the running text."""
    out = []
    for b, block in enumerate(parsed["blocks"]):
        if block["type"] == "text":
            for t in NUM_RE.findall(block["text"]):
                v = _value(t)
                if v is not None:
                    out.append((b, t, abs(v)))
    return out


PERIOD_LABEL_RE = re.compile(r"ended|as of|months|weeks|quarter|year", re.I)
UNITS_RE = re.compile(r"\bin (?:millions|thousands|billions)\b|except per share", re.I)


def _is_header_row(row: list[str]) -> bool:
    """Every filled value cell is text or a year: 'Three Months Ended', 'Jan 25, 2026', '2026'."""
    values = [c for c in row[1:] if c.strip()]
    return bool(values) and all(cell_value(c) is None or (cell_value(c).is_integer() and 1900 <= cell_value(c) <= 2100)
                                for c in values)


def column_header(rows: list[list[str]], row: int, col: int) -> str:
    """Header text above a cell: the nearest group of header rows, spanned headers carried to the
    right, and a period label in the first column ('Year Ended June 30,') kept."""
    r = row - 1
    while r >= 0 and not _is_header_row(rows[r]):
        r -= 1
    group = []
    while r >= 0 and _is_header_row(rows[r]):
        group.insert(0, rows[r])
        r -= 1
    parts = []
    for hdr in group:
        text = ""
        for c in range(1, min(col, len(hdr) - 1) + 1):
            text = hdr[c].strip() or text      # a spanned header covers the empty cells to its right
        label = hdr[0].strip()
        if label and PERIOD_LABEL_RE.search(label) and not UNITS_RE.search(label):
            text = f"{label} {text}".strip()
        if text and text not in parts and not UNITS_RE.search(text):   # units rows are not column headers
            parts.append(text)
    return " · ".join(parts)


def _cited_sources(answer: str, start: int, end: int, n_sources: int) -> list[int]:
    """0-based sources cited for the number at [start, end): the first citation after it in the
    same sentence, else the last one before it in the same sentence."""
    sentence_end = next((m.start() for m in SENTENCE_END_RE.finditer(answer, end)), len(answer))
    sentence_start = max((m.end() for m in SENTENCE_END_RE.finditer(answer, 0, start)), default=0)
    groups = [(m.start(), m) for m in CITE_RE.finditer(answer)]
    after = [m for pos, m in groups if end <= pos <= sentence_end]
    before = [m for pos, m in groups if sentence_start <= pos < start]
    chosen = after[0] if after else before[-1] if before else None
    if chosen is None:
        return []
    return [int(n) - 1 for n in re.findall(r"\d+", chosen.group()) if 1 <= int(n) <= n_sources]


def locate_numbers(answer: str, contexts: list[str], calcs: list[dict] | None = None,
                   max_hits: int = 6) -> tuple[list[dict], list[dict]]:
    """(parsed sources, one entry per number in the answer).

    Each number entry: start/end offsets in the answer, token, status ("cell" | "text" |
    "calculated" | "not found"), whether the match is in a cited source, and the cells
    (source, block, row, col, row label, column header) or text blocks it was found in."""
    parsed = [parse_source(c) for c in contexts]
    cells = [cell for s, p in enumerate(parsed) for cell in table_cells(s, p)]
    texts = [(s, b, t, v) for s, p in enumerate(parsed) for b, t, v in text_numbers(p)]
    calc_values = [abs(c["value"]) for c in (calcs or []) if "value" in c]
    out = []
    for start, end, token, v in number_spans(answer, keep_percent=True):
        cited = _cited_sources(answer, start, end, len(parsed))
        entry = {"start": start, "end": end, "token": token, "status": "not found",
                 "in_cited_source": False, "cells": [], "text": []}
        for scope in ([cited] if cited else []) + [range(len(parsed))]:
            hits = [c for c in cells if c.src in scope and matches(v, c.value)]
            text_hits = [(s, b, t) for s, b, t, tv in texts if s in scope and matches(v, tv)]
            if hits or text_hits:
                entry["in_cited_source"] = scope is cited   # False: found only in a source it didn't cite
                entry["status"] = "cell" if hits else "text"
                rows_of = lambda c: parsed[c.src]["blocks"][c.block]["rows"]  # noqa: E731
                entry["cells"] = [{"source": c.src + 1, "block": c.block, "row": c.row, "col": c.col,
                                   "label": rows_of(c)[c.row][0].replace("*", "").strip(),
                                   "column": column_header(rows_of(c), c.row, c.col)} for c in hits[:max_hits]]
                entry["text"] = [{"source": s + 1, "block": b, "token": t} for s, b, t in text_hits[:max_hits]]
                break
        if entry["status"] == "not found" and any(abs(v - cv) < 0.005 or round(cv, 1) == v or round(cv) == v
                                                   for cv in calc_values):
            entry["status"] = "calculated"
        out.append(entry)
    return parsed, out
