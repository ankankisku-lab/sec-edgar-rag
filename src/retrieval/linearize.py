"""Table rows as sentences, for models trained on prose (the cross-encoder).

    | | Three Months Ended | | Nine Months Ended | |
    | | June 28, 2025 | June 29, 2024 | June 28, 2025 | June 29, 2024 |
    | Operating income | 28,202 | 25,352 | 100,623 | 93,625 |
->
    Operating income: Three Months Ended June 28, 2025 = 28,202; Three Months Ended
    June 29, 2024 = 25,352; Nine Months Ended June 28, 2025 = 100,623; ...

Only the reranker reads this view; the index and the LLM context are unchanged.
"""
from __future__ import annotations

from src.chunking.hierarchical import parse_markdown
from src.parsing import tables


def column_headers(header: list[list[str]], n_cols: int) -> list[str]:
    """Per-column header text; spanning group headers ("Three Months Ended") are
    carried right over the empty cells they span."""
    names = [""] * n_cols
    for row in header:
        # A period label in the label column applies to every column: "Year Ended June 30," | 2025
        prefix = row[0] if tables.PERIOD_LABEL.match(row[0] or "") else ""
        filled, last = [], ""
        for c in range(n_cols):
            cell = row[c] if c < len(row) else ""
            if c > 0 and cell:
                last = cell
            filled.append(cell or (last if c > 0 and len(header) > 1 and row is header[0] else ""))
        for c in range(1, n_cols):
            if filled[c]:
                names[c] = " ".join(p for p in (names[c], prefix, filled[c]) if p)
    return names


def linearize_table(markdown: str) -> str:
    grid = parse_markdown(markdown)
    if not grid:
        return markdown
    header = tables.header_rows(grid) or grid[:1]
    names = column_headers(header, len(grid[0]))
    lines, group = [], ""
    for row in grid[len(header):]:
        label, values = row[0], row[1:]
        if label and not any(values):          # "Operating expenses:" group row
            group = label.rstrip(":")
            continue
        cells = [f"{names[i + 1] or 'value'} = {v}" for i, v in enumerate(values) if v]
        if not cells:
            continue
        name = f"{group} - {label}" if group and label else (label or group)
        lines.append(f"{name}: " + "; ".join(cells) + ".")
    return "\n".join(lines) or markdown


def rerank_text(node) -> str:
    """Metadata header + chunk, with table chunks linearised."""
    from llama_index.core.schema import MetadataMode
    text = node.get_content(metadata_mode=MetadataMode.EMBED)
    if node.metadata.get("content_type") != "table" or "| " not in node.text:
        return text
    body = node.text
    if body.startswith("Footnotes:"):
        return text
    header_part = text[: len(text) - len(body)]
    return header_part + linearize_table(body)
