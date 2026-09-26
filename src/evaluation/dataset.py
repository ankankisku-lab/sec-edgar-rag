"""Retrieval evaluation set with deterministic ground truth.

Two question types:

* numeric   -- generated from financial-statement rows (income statement, balance
               sheet, cash flows), current-period column only. Relevant chunks:
               every child containing that row with that value (the same figure
               re-appears as the comparative column in the next year's filing), or
               a text chunk stating the value next to the line-item name.
* narrative -- hand-written in configs/eval_narrative.yaml with explicit relevance
               rules (filing scope, section, keywords), so labels are auditable.

Usage:
    python -m src.evaluation.dataset --tickers AAPL MSFT NVDA
    -> data/eval/retrieval_v0.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from datetime import date
from pathlib import Path

import yaml

from src.chunking.hierarchical import CHUNKS_DIR, load_nodes, parse_markdown
from src.config import CONFIG_DIR, DATA_DIR
from src.parsing import tables

EVAL_DIR = DATA_DIR / "eval"
SEED = 7

STATEMENTS = {
    "income": re.compile(r"statements?\s+of\s+(operations|income|earnings)|income\s+statements?", re.I),
    "balance": re.compile(r"balance\s+sheets?|statements?\s+of\s+financial\s+position", re.I),
    "cashflow": re.compile(r"statements?\s+of\s+cash\s+flows?|cash\s+flows?\s+statements?", re.I),
}
EXCLUDE_CAPTION = re.compile(r"comprehensive|shareholders|stockholders|equity", re.I)
PERIOD_LABEL = re.compile(r"^(three|six|nine|twelve)\s+months|^years?\s+ended|^quarter|^fiscal\s+year", re.I)
YEAR = re.compile(r"^(19|20)\d\d$")
GENERIC_LABELS = {"products", "services", "service", "product", "other", "basic", "diluted", "total",
                  "net", "service and other", "other, net", "other income (expense), net"}
MONTH_ABBR = {m: m[:3] for m in ["January", "February", "March", "April", "May", "June", "July",
                                 "August", "September", "October", "November", "December"]}


def short_company(name: str) -> str:
    return re.sub(r",?\s+(inc\.?|corp\.?|corporation|incorporated|co\.?|ltd\.?|plc)$", "", name, flags=re.I).strip()


def is_value(cell: str) -> bool:
    return bool(cell) and tables.is_numeric(cell) and not tables._NIL.match(cell.replace(" ", ""))


def norm_value(cell: str) -> str:
    return cell.replace("$", "").replace(" ", "")


def column_period(grid: list[list[str]], col: int = 1) -> tuple[str, int]:
    """Period text for a value column and the index of the first data row.

    Handles "Three Months Ended" / "June 28, 2025" header rows, separate year rows
    ("2025"), and period labels in the label column ("Year Ended June 30," | 2025).
    """
    parts: list[str] = []
    for i, row in enumerate(grid):
        label, cell = row[0], row[col] if col < len(row) else ""
        values = [c for c in row[1:] if c]
        if label and not PERIOD_LABEL.match(label) and any(is_value(c) and not YEAR.match(c) for c in values):
            return " ".join(parts), i  # first real data row
        if PERIOD_LABEL.match(label or ""):
            parts.append(label)
        if cell and (not label or PERIOD_LABEL.match(label) or not values or all(YEAR.match(c) or not is_value(c) for c in values)):
            parts.append(cell)
    return " ".join(parts), len(grid)


def period_matches(phrase: str, period_end: date) -> bool:
    month = period_end.strftime("%B")
    return (str(period_end.year) in phrase and (month in phrase or MONTH_ABBR[month] in phrase)
            and (str(period_end.day) in phrase or "Year" in phrase or "year" in phrase))


def statement_kind(caption: str) -> str | None:
    if not caption or EXCLUDE_CAPTION.search(caption):
        return None
    return next((k for k, rx in STATEMENTS.items() if rx.search(caption)), None)


def lower_label(label: str) -> str:
    return label[0].lower() + label[1:] if len(label) > 1 and label[1].islower() else label


def fiscal_phrase(meta: dict, phrase: str, kind: str) -> str | None:
    fy, fp = meta.get("fiscal_year"), meta.get("fiscal_period")
    if not fy or not fp:
        return None
    if kind == "balance":
        return f"at the end of fiscal {fy}" if fp == "FY" else f"at the end of fiscal {fy} {fp}"
    if fp == "FY" and re.search(r"year", phrase, re.I):
        return f"in fiscal {fy}"
    if fp != "FY" and re.search(r"three months|quarter|13 weeks", phrase, re.I):
        return f"in fiscal {fy} {fp}"
    return None


def numeric_questions(chunk_file: Path, rng: random.Random, per_statement: int = 2) -> list[dict]:
    nodes = load_nodes(chunk_file)
    out = []
    for node in nodes:
        m = node.metadata
        if m["node_type"] != "parent" or m["content_type"] != "table":
            continue
        kind = statement_kind(m.get("table_caption", ""))
        # Primary statements only: they come before Note 1, so they have no note label.
        # "Financial Statement" also matches Item 15 ("...Financial Statement Schedules"),
        # where some filers (e.g. NVIDIA) physically place their statements.
        if kind is None or not re.search(r"Financial Statement", m["section"]) or m.get("note"):
            continue
        md = next((b for b in node.text.split("\n\n") if b.startswith("| ")), None)
        if not md:
            continue
        grid = parse_markdown(md)
        phrase, first_data = column_period(grid)
        period_end = date.fromisoformat(m["period_of_report"])
        if not period_matches(phrase, period_end):
            continue
        phrase = " ".join(phrase.replace(",", ", ").split()).replace(" ,", ",").rstrip(",")
        phrase = re.sub(r"\b(Three|Six|Nine|Twelve|Months|Ended|Years?|Weeks|Quarter|Fiscal)\b",
                        lambda w: w.group(0).lower(), phrase)

        label_counts = Counter(r[0] for r in grid[first_data:])
        group = None
        candidates = []
        for row in grid[first_data:]:
            label, cell = row[0], row[1] if len(row) > 1 else ""
            if label and not any(row[1:]):
                group = label.rstrip(":")
                continue
            if not label or not is_value(cell) or len(label) > 90 or norm_value(cell).strip("()") in ("0", "0.0"):
                continue
            name = label
            if label.lower() in GENERIC_LABELS or label_counts[label] > 1:
                if not group:
                    continue
                name = f"{label} ({group})"
            candidates.append((label, name, cell))
        rng.shuffle(candidates)
        for label, name, cell in candidates[:per_statement]:
            company = short_company(m["company"])
            fiscal = fiscal_phrase(m, phrase, kind)
            if kind == "balance":
                when = f"as of {phrase}" if rng.random() < 0.5 or not fiscal else fiscal
            else:
                when = f"for the {phrase[0].lower() + phrase[1:]}" if rng.random() < 0.5 or not fiscal else fiscal
            out.append({
                "type": "numeric", "statement": kind,
                "question": f"What was {company}'s {lower_label(name)} {when}?",
                "ticker": m["ticker"], "filing_id": m["filing_id"], "period_phrase": phrase,
                "row_label": label, "answer_value": cell, "units": m.get("units"),
            })
    return out


def row_matches(text: str, label: str, value: str) -> bool:
    """Table chunk containing `| label | ... value ...` on one row."""
    target = norm_value(value)
    for line in text.split("\n"):
        if line.startswith(f"| {label} |") and target in [norm_value(c) for c in line.strip("| ").split(" | ")]:
            return True
    return False


def numeric_gold(q: dict, children: list) -> list[str]:
    value, label = norm_value(q["answer_value"]), q["row_label"].lower()
    gold = []
    for c in children:
        if c.metadata["ticker"] != q["ticker"]:
            continue
        if c.metadata["content_type"] == "table" and row_matches(c.text, q["row_label"], q["answer_value"]):
            gold.append(c.node_id)
        elif (c.metadata["content_type"] == "text" and len(value.replace(",", "")) >= 4
              and value in c.text and label in c.text.lower()):
            gold.append(c.node_id)
    return gold


def narrative_gold(rule: dict, children: list) -> list[str]:
    gold = []
    for c in children:
        m = c.metadata
        if rule.get("tickers") and m["ticker"] not in rule["tickers"]:
            continue
        if rule.get("filing_ids") and m["filing_id"] not in rule["filing_ids"]:
            continue
        if rule.get("heading_any") and not any(h.lower() in m.get("heading_path", "").lower() for h in rule["heading_any"]):
            continue
        text = c.text.lower()
        if not all(k.lower() in text for k in rule.get("text_all", [])):
            continue
        if rule.get("text_any") and not any(k.lower() in text for k in rule["text_any"]):
            continue
        gold.append(c.node_id)
    return gold


def build(tickers: list[str] | None, per_statement: int) -> list[dict]:
    rng = random.Random(SEED)
    files = sorted(CHUNKS_DIR.glob("*/*.jsonl"))
    if tickers:
        files = [f for f in files if f.parent.name in {t.upper() for t in tickers}]
    children = [n for f in files for n in load_nodes(f) if n.metadata["node_type"] == "child"]

    questions = []
    for f in files:
        for q in numeric_questions(f, rng, per_statement):
            q["gold_ids"] = numeric_gold(q, children)
            if q["gold_ids"]:
                questions.append(q)

    spec = yaml.safe_load(open(CONFIG_DIR / "eval_narrative.yaml", encoding="utf-8"))
    for item in spec["questions"]:
        gold = narrative_gold(item["relevant"], children)
        questions.append({"type": "narrative", "question": item["question"],
                          "ticker": ",".join(item["relevant"].get("tickers", [])),
                          "rule": item["relevant"], "gold_ids": gold})

    for i, q in enumerate(questions):
        q["qid"] = f"q{i:04d}"
    return questions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", nargs="+")
    parser.add_argument("--per-statement", type=int, default=2)
    parser.add_argument("--out", default="retrieval_v0.jsonl")
    args = parser.parse_args()
    questions = build(args.tickers, args.per_statement)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out = EVAL_DIR / args.out
    with open(out, "w", encoding="utf-8") as fh:
        for q in questions:
            fh.write(json.dumps(q, ensure_ascii=False) + "\n")
    kinds = Counter(q["type"] for q in questions)
    empty = [q["qid"] for q in questions if not q["gold_ids"]]
    print(f"{len(questions)} questions {dict(kinds)} -> {out}")
    print(f"gold chunks per question: median {sorted(len(q['gold_ids']) for q in questions)[len(questions) // 2]}, "
          f"questions with NO gold: {len(empty)} {empty[:10]}")


if __name__ == "__main__":
    main()
