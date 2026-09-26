"""Structure-aware parent-child chunking into LlamaIndex TextNodes.

    parsed elements -> parents  (section text groups | whole tables + footnotes)
                    -> children (sentence-split text | row groups with header rows)

Children are the retrieval units (small, precise); parents are what the LLM
reads after retrieval (complete context). Every node carries filing and section
metadata; the fields listed in EMBED_KEYS are prepended to the text when it is
embedded, so a chunk from Apple's Q3 2025 10-Q is findable as such even though
the chunk text itself never says "Apple".

Usage:
    python -m src.chunking.hierarchical --tickers AAPL MSFT NVDA
    python -m src.chunking.hierarchical            # all parsed filings
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import uuid
from datetime import date
from pathlib import Path

import pandas as pd
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import MetadataMode, NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.core.utils import get_tokenizer
from tqdm import tqdm

from src.config import CHECKPOINT_DIR, DATA_DIR, LOG_DIR, PROCESSED_DIR, load_config
from src.ingestion.checkpoint import Checkpoint
from src.parsing import tables

log = logging.getLogger(__name__)

CHUNKER_VERSION = 2
CHUNKS_DIR = DATA_DIR / "chunks"
CHUNK_CHECKPOINT = CHECKPOINT_DIR / "chunk_state.json"
ID_NAMESPACE = uuid.UUID("5b0c3f4e-8d0a-4b8e-9a51-2f6f3f2c7a10")  # fixed: IDs stay stable across runs

# Metadata the embedder and the LLM see in front of the chunk text.
EMBED_KEYS = ["company", "ticker", "form_type", "period", "heading_path", "table_caption", "units"]
LLM_KEYS = EMBED_KEYS + ["filing_date"]

_tokenizer = get_tokenizer()


def n_tokens(text: str) -> int:
    return len(_tokenizer(text))


# --------------------------------------------------------------------------- metadata

def period_label(doc: dict) -> str:
    """'fiscal 2025 Q3 (quarter ended June 28, 2025)' -- fiscal and calendar, since
    users ask either way and they rarely coincide (NVIDIA's FY2026 Q2 ends July 2025)."""
    end = date.fromisoformat(doc["period_of_report"]).strftime("%B %d, %Y").replace(" 0", " ")
    fy, fp = doc.get("fiscal_year"), doc.get("fiscal_period")
    kind = "fiscal year" if doc["form_type"] == "10-K" else "quarter"
    fiscal = f"fiscal {fy} {'annual' if fp == 'FY' else fp}" if fy and fp else None
    return f"{fiscal} ({kind} ended {end})" if fiscal else f"{kind} ended {end}"


def filing_metadata(doc: dict) -> dict:
    return {
        "filing_id": doc["filing_id"], "company": doc["company"], "ticker": doc["ticker"],
        "cik": doc["cik"], "form_type": doc["form_type"], "filing_date": doc["filing_date"],
        "period_of_report": doc["period_of_report"], "fiscal_year": doc.get("fiscal_year"),
        "fiscal_period": doc.get("fiscal_period"), "period": period_label(doc),
        "accession_number": doc["accession_number"], "source_url": doc["document_url"],
    }


def section_metadata(el: dict) -> dict:
    path = [p for p in (el["section"], el.get("note"), el.get("subsection")) if p]
    return {"section": el["section"], "part": el.get("part"), "item": el.get("item"),
            "note": el.get("note"), "subsection": el.get("subsection"), "heading_path": " > ".join(path)}


# --------------------------------------------------------------------------- tables

def parse_markdown(md: str) -> list[list[str]]:
    rows = []
    for line in md.split("\n"):
        if line.startswith("|---"):
            continue
        cells = re.split(r"(?<!\\) \| ", line[2:-2])
        rows.append([c.replace("\\|", "|") for c in cells])
    return rows


def render_table(header: list[list[str]], body: list[list[str]]) -> str:
    rows = header + body
    return tables.to_markdown(rows) if rows else ""


def table_parts(grid: list[list[str]], max_tokens: int) -> tuple[list[list[str]], list[list[list[str]]]]:
    """Split body rows into groups under `max_tokens`, each to be rendered with the header."""
    header = tables.header_rows(grid) or grid[:1]
    body = grid[len(header):]
    base = n_tokens(render_table(header, []))
    groups, current, size = [], [], base
    for row in body:
        row_tokens = n_tokens(" | ".join(row)) + 4
        if current and size + row_tokens > max_tokens:
            groups.append(current)
            current, size = [], base
        current.append(row)
        size += row_tokens
    if current or not groups:
        groups.append(current)
    return header, groups


# --------------------------------------------------------------------------- parents

def text_groups(elements: list[dict], cfg: dict) -> list[dict]:
    """Group consecutive text elements by heading, then merge small / split large groups."""
    groups: list[dict] = []
    for el in elements:
        if el["type"] not in ("heading", "paragraph"):
            continue
        key = (el["section"], el.get("note"))
        starts_new = (not groups or groups[-1]["key"] != key or el["type"] == "heading"
                      and groups[-1]["has_body"])
        if starts_new:
            groups.append({"key": key, "meta": section_metadata(el), "lines": [], "has_body": False,
                           "first_idx": el["idx"], "subheadings": set()})
        g = groups[-1]
        g["lines"].append(f"## {el['text']}" if el["type"] == "heading" else el["text"])
        g["has_body"] |= el["type"] == "paragraph"
        if el.get("heading_kind") == "heading":
            g["subheadings"].add(el["text"])

    # Merge small neighbours within the same section and note.
    merged: list[dict] = []
    for g in groups:
        g["tokens"] = n_tokens("\n\n".join(g["lines"]))
        prev = merged[-1] if merged else None
        if (prev and prev["key"] == g["key"] and prev["tokens"] < cfg["parent_min_tokens"]
                and prev["tokens"] + g["tokens"] <= cfg["parent_max_tokens"]):
            prev["lines"] += g["lines"]
            prev["tokens"] += g["tokens"]
            prev["subheadings"] |= g["subheadings"]
        else:
            merged.append(g)

    # Split oversized groups at paragraph boundaries (sentence boundaries for huge paragraphs).
    splitter = SentenceSplitter(chunk_size=cfg["parent_max_tokens"], chunk_overlap=0)
    out: list[dict] = []
    for g in merged:
        pieces, current, size = [], [], 0
        for line in g["lines"]:
            t = n_tokens(line)
            parts = splitter.split_text(line) if t > cfg["parent_max_tokens"] else [line]
            for part in parts:
                pt = n_tokens(part)
                if current and size + pt > cfg["parent_max_tokens"]:
                    pieces.append(current)
                    current, size = [], 0
                current.append(part)
                size += pt
        if current:
            pieces.append(current)
        out += [{"kind": "text", "meta": g["meta"], "order": g["first_idx"], "text": "\n\n".join(p),
                 "subheadings": g["subheadings"]} for p in pieces]
    return out


def table_parents(elements: list[dict], cfg: dict) -> list[dict]:
    footnotes: dict[int, list[str]] = {}
    for el in elements:
        if el["type"] == "footnote" and el.get("table_ref") is not None:
            footnotes.setdefault(el["table_ref"], []).append(el["text"])

    footnote_splitter = SentenceSplitter(chunk_size=cfg["child_tokens"], chunk_overlap=0)
    out = []
    for el in elements:
        if el["type"] != "table":
            continue
        grid = parse_markdown(el["text"])
        header, parent_groups = table_parts(grid, cfg["parent_max_tokens"])
        meta = {**section_metadata(el), "table_caption": el.get("caption"), "units": el.get("units")}
        notes = footnotes.get(el["idx"], [])
        for i, rows in enumerate(parent_groups):
            preamble = [f"Table: {el['caption']}" if el.get("caption") else None,
                        f"Units: {el['units']}" if el.get("units") else None,
                        f"(part {i + 1} of {len(parent_groups)})" if len(parent_groups) > 1 else None]
            body = render_table(header, rows)
            text = "\n".join(p for p in preamble if p) + "\n\n" + body
            if notes and i == len(parent_groups) - 1:
                text += "\n\nFootnotes:\n" + "\n".join(notes)
            _, child_groups = table_parts(header + rows, cfg["child_tokens"])
            children = [render_table(header, cg) for cg in child_groups if cg]
            if notes and i == len(parent_groups) - 1:
                # Some tables carry pages of footnotes: keep each piece within the child budget.
                children += [f"Footnotes:\n{piece}" for piece in footnote_splitter.split_text("\n".join(notes))]
            out.append({"kind": "table", "meta": meta, "order": el["idx"], "text": text.strip(),
                        "children": children or [body]})
    return out


# --------------------------------------------------------------------------- nodes

def child_section_metadata(spec: dict, child_text: str) -> dict:
    """A merged text parent can span several subheadings; label each child with
    the subheading in force at its midpoint rather than the parent's first one."""
    parent_text = spec["text"]
    start = max(parent_text.find(child_text[:80]), 0)
    midpoint = start + len(child_text) // 2
    subsection = spec["meta"]["subsection"]
    for m in re.finditer(r"^## (.+)$", parent_text, re.M):
        if m.start() > midpoint:
            break
        if m.group(1) in spec["subheadings"]:
            subsection = m.group(1)
    meta = dict(spec["meta"], subsection=subsection)
    meta["heading_path"] = " > ".join(p for p in (meta["section"], meta.get("note"), subsection) if p)
    return meta


def make_node(node_id: str, text: str, metadata: dict) -> TextNode:
    metadata = {k: v for k, v in metadata.items() if v not in (None, "")}
    return TextNode(
        id_=node_id, text=text, metadata=metadata,
        excluded_embed_metadata_keys=[k for k in metadata if k not in EMBED_KEYS],
        excluded_llm_metadata_keys=[k for k in metadata if k not in LLM_KEYS],
    )


def chunk_filing(doc: dict, cfg: dict) -> tuple[list[TextNode], list[TextNode]]:
    base = filing_metadata(doc)
    acc = doc["accession_number"]
    child_splitter = SentenceSplitter(chunk_size=cfg["text"]["child_tokens"],
                                      chunk_overlap=cfg["text"]["child_overlap_tokens"])

    specs = text_groups(doc["elements"], cfg["text"]) + table_parents(doc["elements"], cfg["tables"])
    specs.sort(key=lambda s: s["order"])  # document order, so prev/next links are meaningful

    parents, children = [], []
    for pi, spec in enumerate(specs):
        pid = str(uuid.uuid5(ID_NAMESPACE, f"{acc}/p{pi}"))
        parent = make_node(pid, spec["text"], {**base, **spec["meta"], "content_type": spec["kind"],
                                               "node_type": "parent"})
        child_texts = spec["children"] if spec["kind"] == "table" else child_splitter.split_text(spec["text"])
        kids = []
        for ci, text in enumerate(child_texts):
            cid = str(uuid.uuid5(ID_NAMESPACE, f"{acc}/p{pi}/c{ci}"))
            meta = spec["meta"] if spec["kind"] == "table" else child_section_metadata(spec, text)
            child = make_node(cid, text, {**base, **meta, "content_type": spec["kind"],
                                          "node_type": "child", "parent_id": pid})
            child.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(node_id=pid)
            child.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=acc)
            kids.append(child)
        parent.relationships[NodeRelationship.CHILD] = [RelatedNodeInfo(node_id=k.node_id) for k in kids]
        parent.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=acc)
        if parents:
            parent.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(node_id=parents[-1].node_id)
            parents[-1].relationships[NodeRelationship.NEXT] = RelatedNodeInfo(node_id=pid)
        parents.append(parent)
        children += kids
    return parents, children


def chunk_stats(parents: list[TextNode], children: list[TextNode]) -> dict:
    def pct(values, q):
        return int(pd.Series(values).quantile(q)) if values else 0
    child_embed = [n_tokens(c.get_content(MetadataMode.EMBED)) for c in children]
    parent_tok = [n_tokens(p.text) for p in parents]
    return {
        "n_parents_text": sum(p.metadata["content_type"] == "text" for p in parents),
        "n_parents_table": sum(p.metadata["content_type"] == "table" for p in parents),
        "n_children": len(children),
        "child_embed_tokens_p50": pct(child_embed, .5), "child_embed_tokens_p95": pct(child_embed, .95),
        "child_embed_tokens_max": max(child_embed, default=0),
        "parent_tokens_p50": pct(parent_tok, .5), "parent_tokens_max": max(parent_tok, default=0),
    }


def output_path(ticker: str, filing_id: str) -> Path:
    return CHUNKS_DIR / ticker / f"{filing_id}.jsonl"


def load_nodes(path: Path) -> list[TextNode]:
    with open(path, encoding="utf-8") as f:
        return [TextNode.from_json(line) for line in f]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    LOG_DIR.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.FileHandler(LOG_DIR / "chunk.log", encoding="utf-8")])
    cfg = load_config("chunking")
    files = sorted(PROCESSED_DIR.glob("*/*.json"))
    if args.tickers:
        wanted = {t.upper() for t in args.tickers}
        files = [f for f in files if f.parent.name in wanted]
    if args.limit:
        files = files[:args.limit]

    ckpt = Checkpoint(CHUNK_CHECKPOINT)
    done = skipped = failed = 0
    for f in tqdm(files, desc="chunking"):
        fid = f.stem
        out = output_path(f.parent.name, fid)
        state = ckpt.get(fid) or {}
        if (not args.force and state.get("status") == "done" and state.get("chunker_version") == CHUNKER_VERSION
                and out.exists() and out.stat().st_mtime >= f.stat().st_mtime):
            skipped += 1
            continue
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
            parents, children = chunk_filing(doc, cfg)
        except Exception as e:  # keep going; the checkpoint records what failed
            log.exception("%s failed", fid)
            ckpt.mark(fid, "failed", error=f"{type(e).__name__}: {e}", chunker_version=CHUNKER_VERSION)
            failed += 1
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            for node in parents + children:
                fh.write(node.to_json() + "\n")
        stats = chunk_stats(parents, children)
        ckpt.mark(fid, "done", error=None, chunker_version=CHUNKER_VERSION, **stats)
        log.info("%s ok: %s", fid, stats)
        done += 1
    print(f"chunked {done}, skipped {skipped}, failed {failed} -> {CHUNKS_DIR}")


if __name__ == "__main__":
    main()
