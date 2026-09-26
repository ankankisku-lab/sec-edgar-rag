"""Turn SEC HTML tables into clean grids and markdown.

SEC financial tables are typeset, not structured: "$" and ")" live in their own
cells, headers span columns with colspan, and spacer columns sit between
periods. We rebuild the logical grid so every number stays on the same row as
its label and under the right period header.
"""
from __future__ import annotations

import re

from src.parsing.cleaners import normalize

CURRENCY = {"$", "€", "£", "¥"}
_TRAILING = re.compile(r"^(\)|%|\)%|%\)|pts?\)?|bps)$", re.I)
_NUMBER = re.compile(r"^[\(\-–—$€£¥]*\s*\d[\d,]*(\.\d+)?\s*\)?%?$")
_NIL = re.compile(r"^[$€£¥]?(—|–|-|n/?a|nm)$", re.I)
_UNITS = re.compile(r"\b(?:in|dollars in|amounts in|\$ in)\s+(millions|thousands|billions)", re.I)


def expand(table) -> list[list[str]]:
    """Rows of cell texts with colspan expanded (text in the first spanned column)."""
    rows = []
    for tr in table.iter("tr"):
        row = []
        for cell in tr:
            if not isinstance(cell.tag, str) or cell.tag not in ("td", "th"):
                continue
            row.append(normalize(cell.text_content()))
            try:
                span = max(1, min(int(cell.get("colspan", 1)), 50))
            except ValueError:
                span = 1
            row.extend([""] * (span - 1))
        rows.append(row)
    width = max((len(r) for r in rows), default=0)
    return [r + [""] * (width - len(r)) for r in rows]


def _attach_symbols(row: list[str]) -> None:
    """Glue "$" onto the following value and ")" / "%" onto the preceding one."""
    for c, cell in enumerate(row):
        if cell in CURRENCY:
            for n in range(c + 1, min(c + 3, len(row))):
                if row[n]:
                    row[n] = f"{cell}{row[n]}"
                    row[c] = ""
                    break
        elif _TRAILING.match(cell):
            for p in range(c - 1, max(c - 3, -1), -1):
                if row[p]:
                    row[p] = f"{row[p]}{cell}"
                    row[c] = ""
                    break


def _merge_compatible_columns(rows: list[list[str]]) -> list[list[str]]:
    """Merge neighbouring columns that are never both filled in the same row.

    This collapses "$ | 95,359 | )" style cell fragments and colspan headers
    into single logical columns while keeping distinct periods apart (any row
    with values in both columns blocks the merge).
    """
    # Header rows = everything above the first row with a label AND a number.
    # A column that carries header text (a period like "March 31, 2025") belongs
    # to the values, never to the label column, even if they never collide.
    n_header = next((i for i, r in enumerate(rows)
                     if r[0] and any(is_numeric(c) for c in r[1:] if c)), 0)
    cols = [list(col) for col in zip(*rows)]
    merged = [cols[0]]
    for col in cols[1:]:
        prev = merged[-1]
        if len(merged) == 1 and any(col[:n_header]):
            merged.append(col)
        elif all(not (a and b) for a, b in zip(prev, col)):
            merged[-1] = [a or b for a, b in zip(prev, col)]
        else:
            merged.append(col)
    return [list(r) for r in zip(*merged)]


def clean_grid(table) -> list[list[str]]:
    rows = expand(table)
    for row in rows:
        _attach_symbols(row)
    rows = [r for r in rows if any(r)]
    if not rows:
        return []
    keep = [c for c in range(len(rows[0])) if any(r[c] for r in rows)]
    rows = [[r[c] for c in keep] for r in rows]
    return _merge_compatible_columns(rows) if rows and rows[0] else []


def is_numeric(cell: str) -> bool:
    """Numbers, percentages, and the dashes SEC tables use for zero / n/a."""
    cell = cell.replace(" ", "")
    return bool(_NUMBER.match(cell) or _NIL.match(cell))


def classify(grid: list[list[str]]) -> str:
    """'data' for real tables; 'layout' for tables used to position text
    (bullets, footnote markers, two-column headings); 'toc' for the index."""
    if not grid:
        return "empty"
    first_cells = [r[0].lower() for r in grid if r and r[0]]
    item_rows = sum(1 for r in grid if re.match(r"^(item|part)\s+[\divx]", " ".join(c for c in r if c).lower()))
    if item_rows >= 3 and item_rows >= 0.3 * len(grid):
        return "toc"
    n_cols = len(grid[0])
    numeric = sum(is_numeric(c) for r in grid for c in r[1:] if c)
    if n_cols <= 2 and numeric == 0:
        return "layout"
    if len(grid) == 1 and numeric <= 1:
        return "layout"
    if first_cells and all(re.match(r"^(•|●|◦|▪|-|–|\(?\w{1,2}\)|\*+|\d{1,2}\.)$", c) for c in first_cells):
        return "layout"
    return "data"


def header_rows(grid: list[list[str]]) -> list[list[str]]:
    """Leading rows before the first row that carries a number (the column headers)."""
    out = []
    for row in grid:
        if row[0] or any(is_numeric(c) for c in row[1:] if c):  # label or value row
            break
        out.append(row)
    return out[:4]


def has_header(grid: list[list[str]]) -> bool:
    """True if the first row labels the value columns (e.g. periods)."""
    return any(c and not is_numeric(c) for c in grid[0][1:])


def all_percentages(grid: list[list[str]]) -> bool:
    # Only labelled rows: header cells like "2025" are numbers but not values.
    values = [c for r in grid if r[0] for c in r[1:] if c and is_numeric(c)]
    return bool(values) and all(c.endswith("%") or c.endswith("%)") for c in values)


def to_markdown(grid: list[list[str]]) -> str:
    def fmt(row):
        return "| " + " | ".join(c.replace("|", "\\|") for c in row) + " |"
    lines = [fmt(grid[0]), "|" + "---|" * len(grid[0])]
    lines += [fmt(r) for r in grid[1:]]
    return "\n".join(lines)


def to_text(grid: list[list[str]]) -> str:
    """Layout tables read as plain text, row by row."""
    return "\n".join(" ".join(c for c in r if c) for r in grid)


def find_units(*texts: str) -> str | None:
    for t in texts:
        m = _UNITS.search(t or "")
        if m:
            return m.group(1).lower()
    return None
