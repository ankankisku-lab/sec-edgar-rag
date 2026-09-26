"""Parse raw SEC 10-K / 10-Q HTML into an ordered list of structured elements.

    raw HTML -> blocks (text | table | pagebreak) -> drop page artifacts
             -> section tracking (Part / Item / Note / heading)
             -> elements: heading | paragraph | table | footnote

Each element carries its section labels, so later chunking never has to guess
where a paragraph or table came from. Tables keep a caption, their units
("millions") and the footnotes printed under them.

Usage:
    python -m src.parsing.parser --tickers AAPL MSFT NVDA
    python -m src.parsing.parser --limit 50
    python -m src.parsing.parser --force          # re-parse even if up to date
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path

import lxml.html
import pandas as pd
from tqdm import tqdm

from src.config import CHECKPOINT_DIR, DATA_DIR, FILINGS_CSV, LOG_DIR, METADATA_DIR, PROCESSED_DIR
from src.ingestion.checkpoint import Checkpoint
from src.ingestion.downloader import CHECKPOINT_PATH as DOWNLOAD_CHECKPOINT
from src.parsing import tables
from src.parsing.cleaners import normalize, page_artifact_indices
from src.parsing.structure import SectionTracker

log = logging.getLogger(__name__)

PARSER_VERSION = 3  # bump when output changes, so --force isn't needed to re-parse
PARSE_CHECKPOINT = CHECKPOINT_DIR / "parse_state.json"

BLOCK_TAGS = {"div", "p", "li", "ul", "ol", "table", "center", "blockquote", "section",
              "h1", "h2", "h3", "h4", "h5", "h6", "hr"}
SKIP_TAGS = {"script", "style", "head", "title", "noscript"}
FOOTNOTE_RE = re.compile(r"^(\(\d{1,2}\)|\([a-z]\)|\*{1,3}|\[\d{1,2}\]|†|‡)\s*", re.I)
_BOLD_STYLE = re.compile(r"font-weight:(bold|[6-9]00)")
_FONT_SIZE = re.compile(r"font-size:([\d.]+)(pt|px)")


# --------------------------------------------------------------------------- HTML -> blocks

def load_html(path: Path):
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("cp1252", errors="replace")
    text = re.sub(r"^\s*<\?xml[^>]*\?>", "", text)  # lxml refuses str input with an encoding decl
    root = lxml.html.document_fromstring(text)
    # Line breaks and block boundaries separate words: "June 28,<br>2025" or
    # <td><p>Three Months Ended</p><p>March 31,</p></td> must not run together.
    for el in root.iter("br", "p", "div", "li", "td", "th", "tr"):
        el.tail = " " + (el.tail or "")
    # Hidden iXBRL header facts and other display:none content are not reader-visible.
    for el in root.xpath('//*[contains(translate(@style, " ", ""), "display:none")] | //*[name()="ix:header"]'):
        el.drop_tree()
    return root


def _is_bold(el) -> bool:
    return el.tag in ("b", "strong") or bool(_BOLD_STYLE.search((el.get("style") or "").replace(" ", "").lower()))


def _font_pt(el) -> float | None:
    m = _FONT_SIZE.search((el.get("style") or "").replace(" ", "").lower())
    if not m:
        return None
    size = float(m.group(1))
    return size * 0.75 if m.group(2) == "px" else size


def body_font_pt(root) -> float:
    """Most common font size by amount of text: the document's body text size."""
    sizes: Counter = Counter()
    for el in root.iter():
        if isinstance(el.tag, str) and el.text and el.text.strip():
            size = _font_pt(el)
            if size:
                sizes[size] += len(el.text)
    return sizes.most_common(1)[0][0] if sizes else 10.0


def emphasis_fraction(el, body_pt: float) -> float:
    """Share of the block's text that is emphasised: bold, or clearly larger than
    body text (some filers, e.g. Intel, mark headings with size/colour, not bold)."""
    total = emphasised = 0
    for t in el.xpath(".//text()"):
        n = len(t.strip())
        if not n:
            continue
        node = t.getparent().getparent() if t.is_tail else t.getparent()
        is_bold, size, inside = False, None, True
        while node is not None and (inside or size is None):
            if inside and _is_bold(node):
                is_bold = True
            if size is None:
                size = _font_pt(node)
            if node is el:
                inside = False  # keep climbing only to find an inherited font size
            node = node.getparent()
        total += n
        if is_bold or (size is not None and size >= body_pt + 1.5):
            emphasised += n
    return emphasised / total if total else 0.0


def _containers(root) -> set:
    """Elements that contain a block-level descendant (so we must descend into them)."""
    out: set = set()
    for el in root.iter():
        if isinstance(el.tag, str) and el.tag in BLOCK_TAGS:
            for anc in el.iterancestors():
                if anc in out:
                    break
                out.add(anc)
    return out


def to_blocks(root) -> tuple[list[dict], Counter]:
    containers = _containers(root)
    body_pt = body_font_pt(root)
    blocks: list[dict] = []
    table_kinds: Counter = Counter()

    def text_block(text: str, bold: float) -> None:
        text = normalize(text)
        if text:
            blocks.append({"kind": "text", "text": text, "bold": bold >= 0.8})

    def walk(el) -> None:
        if not isinstance(el.tag, str) or el.tag in SKIP_TAGS:
            return
        style = (el.get("style") or "").replace(" ", "").lower()
        if "page-break-before:always" in style or "break-before:page" in style:
            blocks.append({"kind": "pagebreak"})
        if el.tag == "hr":
            blocks.append({"kind": "pagebreak"})
        elif el.tag == "table":
            grid = tables.clean_grid(el)
            kind = tables.classify(grid)
            table_kinds[kind] += 1
            if kind == "data":
                blocks.append({"kind": "table", "grid": grid})
            elif kind == "layout":
                text_block(tables.to_text(grid), emphasis_fraction(el, body_pt))
        elif el not in containers:
            text_block(el.text_content(), emphasis_fraction(el, body_pt))
        else:
            if el.text and el.text.strip():
                text_block(el.text, 0.0)
            for child in el:
                walk(child)
                if child.tail and child.tail.strip():
                    text_block(child.tail, 0.0)
        if "page-break-after:always" in style or "break-after:page" in style:
            blocks.append({"kind": "pagebreak"})

    walk(root.find("body") if root.find("body") is not None else root)
    return blocks, table_kinds


# --------------------------------------------------------------------------- blocks -> elements

def to_elements(blocks: list[dict], form_type: str) -> tuple[list[dict], dict]:
    artifacts = page_artifact_indices(blocks)
    tracker = SectionTracker(form_type)
    elements: list[dict] = []
    last_table: int | None = None
    in_footnotes = False

    for i, b in enumerate(blocks):
        if b["kind"] == "pagebreak" or i in artifacts:
            continue

        if b["kind"] == "text":
            text = b["text"]
            if in_footnotes and FOOTNOTE_RE.match(text) and len(text) < 1500:
                elements.append({"type": "footnote", "text": text, "table_ref": last_table, **tracker.labels()})
                elements[last_table].setdefault("footnote_refs", []).append(len(elements) - 1)
                continue
            in_footnotes = False
            heading_kind = tracker.observe(text, b["bold"])
            el = {"type": "heading" if heading_kind else "paragraph", "text": text, **tracker.labels()}
            if heading_kind:
                el["heading_kind"] = heading_kind
            elements.append(el)

        else:  # data table
            grid = b["grid"]
            # Caption: the short heading/paragraph lines right above the table
            # (e.g. "CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS", "(In millions)").
            start = (last_table + 1) if last_table is not None else 0
            caption = " | ".join(e["text"] for e in elements[max(start, len(elements) - 3):]
                                 if e["type"] in ("heading", "paragraph") and len(e["text"]) < 250) or None
            units = tables.find_units(caption or "", " ".join(" ".join(r) for r in grid[:3]))
            continued_from = None

            # A table printed directly under another one (nothing in between) is a
            # continuation: inherit its caption/units and, if it has no column
            # headers of its own, its header rows -- otherwise "34.5%" loses its period.
            prev = elements[-1] if elements and elements[-1]["type"] == "table" else None
            if prev is not None:
                continued_from = prev["idx_tmp"]
                caption = caption or prev["caption"]
                units = units or prev["units"]
                if not tables.has_header(grid) and prev["n_cols"] == len(grid[0]):
                    grid = prev["header_rows"] + grid
            if tables.all_percentages(grid):
                units = None

            elements.append({
                "type": "table",
                "text": tables.to_markdown(grid),
                "caption": caption,
                "units": units,
                "n_rows": len(grid), "n_cols": len(grid[0]),
                "continued_from": continued_from,
                "header_rows": tables.header_rows(grid),
                "idx_tmp": len(elements),
                **tracker.labels(),
            })
            last_table = len(elements) - 1
            in_footnotes = True

    for idx, el in enumerate(elements):
        el["idx"] = idx
        el.pop("idx_tmp", None)
        el.pop("header_rows", None)
    mode = "items" if tracker.saw_item_heading else ("titles" if tracker.item else "none")
    return elements, {"page_artifacts_dropped": len(artifacts), "section_mode": mode}


# --------------------------------------------------------------------------- per filing

def filing_stats(elements: list[dict], extra: dict, table_kinds: Counter) -> dict:
    types = Counter(e["type"] for e in elements)
    chars = sum(len(e["text"]) for e in elements)
    cover_chars = sum(len(e["text"]) for e in elements if e["item"] is None)
    data_tables = [e for e in elements if e["type"] == "table"]
    return {
        **{f"n_{t}": types.get(t, 0) for t in ("heading", "paragraph", "table", "footnote")},
        "chars": chars,
        "pct_chars_unsectioned": round(100 * cover_chars / chars, 1) if chars else 0.0,
        "items": list(dict.fromkeys(e["section"] for e in elements if e["item"])),  # document order
        "n_notes": len({e["note"] for e in elements if e["note"]}),
        "tables_with_units": sum(1 for t in data_tables if t["units"]),
        "tables_with_caption": sum(1 for t in data_tables if t["caption"]),
        "layout_tables_as_text": table_kinds.get("layout", 0),
        "toc_tables_dropped": table_kinds.get("toc", 0),
        **extra,
    }


def parse_filing(meta: dict) -> dict:
    root = load_html(DATA_DIR / meta["local_path"])
    blocks, table_kinds = to_blocks(root)
    elements, extra = to_elements(blocks, meta["form_type"])
    return {**meta, "parser_version": PARSER_VERSION,
            "stats": filing_stats(elements, extra, table_kinds), "elements": elements}


def output_path(ticker: str, filing_id: str) -> Path:
    return PROCESSED_DIR / ticker / f"{filing_id}.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    LOG_DIR.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.FileHandler(LOG_DIR / "parse.log", encoding="utf-8")])

    manifest = pd.read_csv(FILINGS_CSV, dtype={"cik": str})
    fiscal = pd.read_csv(METADATA_DIR / "verification.csv", dtype=str).set_index("filing_id")
    downloaded = Checkpoint(DOWNLOAD_CHECKPOINT)
    manifest = manifest[manifest["filing_id"].map(downloaded.is_done)]
    if args.tickers:
        manifest = manifest[manifest["ticker"].isin([t.upper() for t in args.tickers])]
    if args.limit:
        manifest = manifest.head(args.limit)

    ckpt = Checkpoint(PARSE_CHECKPOINT)
    done = failed = skipped = 0
    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="parsing"):
        fid = row["filing_id"]
        out = output_path(row["ticker"], fid)
        state = ckpt.get(fid) or {}
        if (not args.force and state.get("status") == "done"
                and state.get("parser_version") == PARSER_VERSION and out.exists()):
            skipped += 1
            continue
        meta = json.loads(row.to_json())
        if fid in fiscal.index:
            meta["fiscal_year"] = fiscal.at[fid, "fiscal_year"]
            meta["fiscal_period"] = fiscal.at[fid, "fiscal_period"]
        try:
            result = parse_filing(meta)
        except Exception as e:  # one malformed filing must not stop the batch
            log.exception("%s failed", fid)
            ckpt.mark(fid, "failed", error=f"{type(e).__name__}: {e}", parser_version=PARSER_VERSION)
            failed += 1
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        ckpt.mark(fid, "done", error=None, parser_version=PARSER_VERSION, **{
            k: v for k, v in result["stats"].items() if k != "items"})
        log.info("%s ok: %s", fid, result["stats"])
        done += 1

    print(f"parsed {done}, skipped {skipped}, failed {failed} -> {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
