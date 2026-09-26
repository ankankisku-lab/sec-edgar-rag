"""Retrieval / generation evaluation set with deterministic ground truth.

Facts are extracted from the primary financial statements (income statement,
balance sheet, cash flows): line item, current-period value (column 1),
prior-period value (column 2), period text validated against the filing's
period end. Question types built from them:

  numeric         What was X for <period>?                         1 fact
  yoy             How did X for <period> compare with <prior>?     1 fact (both values in one row)
  pct_change      By what percentage did X change ...?             1 fact + calculation
  cross_quarter   Compare X in fiscal FY Qa and fiscal FY Qb.      2 facts, 2 filings
  cross_company   Compare A's and B's X in their latest 10-Ks.     2 facts, 2 companies
  narrative       hand-written, configs/eval_narrative.yaml        rule-based relevance

Each question carries `gold_groups`: one list of relevant chunk ids per fact it
needs. A chunk is relevant to a fact if it contains that row with that value
(the same figure re-appears as the comparative column in later filings) or is a
text chunk stating the value next to the line-item name.

Usage:
    python -m src.evaluation.dataset --tickers AAPL MSFT NVDA        # -> data/eval/retrieval_v1.jsonl
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import re
from collections import Counter, defaultdict
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
PERIOD_LABEL = tables.PERIOD_LABEL
YEAR = tables.YEAR
GENERIC_LABELS = {"products", "services", "service", "product", "other", "basic", "diluted", "total",
                  "net", "service and other", "other, net", "other income (expense), net"}
CHANGE_GROUP = re.compile(r"changes?\s+in\s+(operating\s+)?(assets|liabilities)", re.I)
CURRENT_GROUP = re.compile(r"^current\s+(assets|liabilities)", re.I)
# Rows that close a group: "Total cost of revenue", "Net cash from operations" (Microsoft
# ends its working-capital group this way), "Net change in cash...", "Cash ... end of period".
GROUP_END = re.compile(r"^(total\b|net cash\b|net (increase|decrease|change)\b|cash(,| and) cash equivalents"
                       r"|cash (generated|provided|used)\b)", re.I)
MONTH_ABBR = {m: m[:3] for m in ["January", "February", "March", "April", "May", "June", "July",
                                 "August", "September", "October", "November", "December"]}
CONCEPTS = {  # cross-company comparisons need one concept across different line-item names
    "total revenue": re.compile(r"^(total net sales|total revenues?|revenues?)$", re.I),
    "net income": re.compile(r"^net income$", re.I),
    "operating income": re.compile(r"^(total )?operating income$", re.I),
}


# --------------------------------------------------------------------------- helpers

def short_company(name: str) -> str:
    return re.sub(r",?\s+(inc\.?|corp\.?|corporation|incorporated|co\.?|ltd\.?|plc)$", "", name, flags=re.I).strip()


def is_value(cell: str) -> bool:
    return bool(cell) and tables.is_numeric(cell) and not tables._NIL.match(cell.replace(" ", ""))


def norm_value(cell: str) -> str:
    return cell.replace("$", "").replace(" ", "")


def to_float(cell: str) -> float:
    t = norm_value(cell).replace(",", "")
    neg = t.startswith("(") and t.endswith(")")
    v = float(t.strip("()").rstrip("%"))
    return -v if neg else v


def column_period(grid: list[list[str]], col: int = 1) -> tuple[str, int]:
    """Period text for a value column and the index of the first data row.

    Handles "Three Months Ended" / "June 28, 2025" header rows, separate year rows
    ("2025"), and period labels in the label column ("Year Ended June 30," | 2025).
    Group headers spanning several columns are carried right to the columns they cover.
    """
    parts: list[str] = []
    for i, row in enumerate(grid):
        label = row[0]
        values = [c for c in row[1:] if c]
        if label and not PERIOD_LABEL.match(label) and any(is_value(c) and not YEAR.match(c) for c in values):
            return " ".join(parts), i
        if PERIOD_LABEL.match(label or ""):
            parts.append(label)
        cell = row[col] if col < len(row) else ""
        if not cell and not any(YEAR.match(c) for c in values):   # spanning group header
            cell = next((row[c] for c in range(col - 1, 0, -1) if row[c]), "")
        if cell and (not label or PERIOD_LABEL.match(label) or not values
                     or all(YEAR.match(c) or not is_value(c) for c in values)):
            parts.append(cell)
    return " ".join(parts), len(grid)


def clean_phrase(phrase: str) -> str:
    phrase = " ".join(phrase.replace(",", ", ").split()).replace(" ,", ",").rstrip(",")
    return re.sub(r"\b(Three|Six|Nine|Twelve|Months|Ended|Years?|Weeks|Quarter|Fiscal)\b",
                  lambda w: w.group(0).lower(), phrase)


def period_matches(phrase: str, period_end: date, years_back: int = 0) -> bool:
    month = period_end.strftime("%B")
    return (str(period_end.year - years_back) in phrase and (month in phrase or MONTH_ABBR[month] in phrase)
            and (years_back > 0 or str(period_end.day) in phrase or "year" in phrase.lower()))


def statement_kind(caption: str) -> str | None:
    if not caption or EXCLUDE_CAPTION.search(caption):
        return None
    return next((k for k, rx in STATEMENTS.items() if rx.search(caption)), None)


def lower_label(label: str) -> str:
    return label[0].lower() + label[1:] if len(label) > 1 and label[1].islower() else label


# --------------------------------------------------------------------------- facts

def extract_facts(chunk_file: Path) -> list[dict]:
    """One fact per primary-statement row with a current-period value."""
    facts = []
    for node in load_nodes(chunk_file):
        m = node.metadata
        if m["node_type"] != "parent" or m["content_type"] != "table":
            continue
        kind = statement_kind(m.get("table_caption", ""))
        # Primary statements come before Note 1 (no note label). "Financial Statement" also
        # matches Item 15, where some filers (e.g. NVIDIA) physically place their statements.
        if kind is None or not re.search(r"Financial Statement", m["section"]) or m.get("note"):
            continue
        md = next((b for b in node.text.split("\n\n") if b.startswith("| ")), None)
        if not md:
            continue
        grid = parse_markdown(md)
        period_end = date.fromisoformat(m["period_of_report"])
        phrase, first_data = column_period(grid, 1)
        if not period_matches(phrase, period_end):
            continue
        prior, _ = column_period(grid, 2)
        prior_ok = period_matches(prior, period_end, years_back=1) and prior != phrase

        label_counts = Counter(r[0] for r in grid[first_data:])
        group = None
        for row in grid[first_data:]:
            label = row[0]
            if label and not any(row[1:]):
                group = label.rstrip(":")
                continue
            cell = row[1] if len(row) > 1 else ""
            closes_group = bool(GROUP_END.match(label or ""))
            if label and is_value(cell) and len(label) <= 90 and norm_value(cell).strip("()") not in ("0", "0.0"):
                if kind == "cashflow" and group and CHANGE_GROUP.search(group) and not closes_group:
                    name = f"change in {lower_label(label)}"
                elif label.lower() in GENERIC_LABELS or label_counts[label] > 1:
                    name = f"{lower_label(label)} ({group})" if group else None
                elif kind == "balance" and group and CURRENT_GROUP.match(group) and "current" not in label.lower():
                    name = f"{lower_label(label)} (current)"
                else:
                    name = lower_label(label)
                if name:
                    prior_cell = row[2] if prior_ok and len(row) > 2 and is_value(row[2]) else None
                    facts.append({**{k: m.get(k) for k in ("ticker", "company", "filing_id", "form_type",
                                                          "fiscal_year", "fiscal_period", "units")},
                                  "statement": kind, "row_label": label, "name": name, "group": group,
                                  "value": cell, "phrase": clean_phrase(phrase),
                                  "prior_value": prior_cell, "prior_phrase": clean_phrase(prior) if prior_cell else None})
            if GROUP_END.match(label or ""):
                group = None  # a group ends at its total / subtotal row
    return facts


def fiscal_phrase(f: dict) -> str | None:
    fy, fp = f.get("fiscal_year"), f.get("fiscal_period")
    if not fy or not fp:
        return None
    if f["statement"] == "balance":
        return f"at the end of fiscal {fy}" if fp == "FY" else f"at the end of fiscal {fy} {fp}"
    if fp == "FY" and "year" in f["phrase"]:
        return f"in fiscal {fy}"
    if fp != "FY" and re.search(r"three months|quarter|13 weeks", f["phrase"]):
        return f"in fiscal {fy} {fp}"
    return None


def when(f: dict, rng: random.Random) -> str:
    fiscal = fiscal_phrase(f)
    if fiscal and rng.random() < 0.5:
        return fiscal
    return f"as of {f['phrase']}" if f["statement"] == "balance" else f"for the {f['phrase']}"


# --------------------------------------------------------------------------- question builders

def q_numeric(facts: list[dict], rng: random.Random, per_statement: int) -> list[dict]:
    out = []
    by_table = defaultdict(list)
    for f in facts:
        by_table[(f["filing_id"], f["statement"])].append(f)
    for group in by_table.values():
        rng.shuffle(group)
        for f in group[:per_statement]:
            out.append({"type": "numeric", "facts": [f],
                        "question": f"What was {short_company(f['company'])}'s {f['name']} {when(f, rng)}?",
                        "answer_values": [f["value"]]})
    return out


def q_yoy(facts: list[dict], rng: random.Random, per_filing: int) -> list[dict]:
    out = []
    by_filing = defaultdict(list)
    for f in facts:
        if f["statement"] != "balance" and f["prior_value"]:
            by_filing[f["filing_id"]].append(f)
    for group in by_filing.values():
        rng.shuffle(group)
        for i, f in enumerate(group[:per_filing]):
            cur, prev = to_float(f["value"]), to_float(f["prior_value"])
            co = short_company(f["company"])
            fiscal = fiscal_phrase(f)
            if i % 2 == 0:
                q = (f"How did {co}'s {f['name']} for the {f['phrase']} compare with the {f['prior_phrase']}?"
                     if rng.random() < 0.5 or not fiscal else
                     f"How did {co}'s {f['name']} {fiscal} compare with the same period a year earlier?")
                out.append({"type": "yoy", "facts": [f], "question": q,
                            "answer_values": [f["value"], f["prior_value"]], "expected_change": cur - prev})
            elif prev > 0 and cur > 0 and not f["name"].startswith("change in"):
                # % change only for positive levels: a % change of a change (or across a sign flip) is meaningless
                q = (f"By what percentage did {co}'s {f['name']} change from the {f['prior_phrase']} "
                     f"to the {f['phrase']}?")
                out.append({"type": "pct_change", "facts": [f], "question": q,
                            "answer_values": [f["value"], f["prior_value"]],
                            "expected_pct": (cur - prev) / abs(prev) * 100})
    return out


def q_cross_quarter(facts: list[dict], rng: random.Random, max_q: int) -> list[dict]:
    """Same company, same fiscal year, two different quarterly filings."""
    by_key = defaultdict(dict)
    for f in facts:
        if (f["statement"] == "income" and f["fiscal_period"] in ("Q1", "Q2", "Q3")
                and re.search(r"three months|quarter|13 weeks", f["phrase"])
                and any(rx.match(f["row_label"]) for rx in CONCEPTS.values())):
            by_key[(f["ticker"], f["fiscal_year"], f["row_label"])][f["fiscal_period"]] = f
    out = []
    for (ticker, fy, label), quarters in by_key.items():
        for a, b in itertools.combinations(sorted(quarters), 2):
            fa, fb = quarters[a], quarters[b]
            out.append({"type": "cross_quarter", "facts": [fa, fb],
                        "question": (f"Compare {short_company(fa['company'])}'s {fa['name']} in fiscal {fy} {a} "
                                     f"with fiscal {fy} {b}. Which quarter was higher, and by how much?"),
                        "answer_values": [fa["value"], fb["value"]],
                        "expected_change": to_float(fb["value"]) - to_float(fa["value"])})
    rng.shuffle(out)
    return out[:max_q]


def q_cross_company(facts: list[dict], rng: random.Random) -> list[dict]:
    """Latest 10-K of each company, same concept."""
    latest = {}
    for f in facts:
        if f["form_type"] == "10-K" and f["statement"] == "income":
            cur = latest.get(f["ticker"])
            if cur is None or f["fiscal_year"] > cur:
                latest[f["ticker"]] = f["fiscal_year"]
    by_concept = defaultdict(dict)
    for f in facts:
        if f["form_type"] == "10-K" and f["statement"] == "income" and latest.get(f["ticker"]) == f["fiscal_year"]:
            for concept, rx in CONCEPTS.items():
                if rx.match(f["row_label"]) and f["ticker"] not in by_concept[concept]:
                    by_concept[concept][f["ticker"]] = f
    out = []
    for concept, per_ticker in by_concept.items():
        for ta, tb in itertools.combinations(sorted(per_ticker), 2):
            fa, fb = per_ticker[ta], per_ticker[tb]
            ca, cb = short_company(fa["company"]), short_company(fb["company"])
            out.append({"type": "cross_company", "facts": [fa, fb],
                        "question": (f"Compare {ca}'s and {cb}'s {concept} in their most recent annual reports "
                                     f"(fiscal {fa['fiscal_year']} and fiscal {fb['fiscal_year']}). Which was larger?"),
                        "answer_values": [fa["value"], fb["value"]]})
    return out


# --------------------------------------------------------------------------- gold

def row_matches(text: str, label: str, value: str) -> bool:
    target = norm_value(value)
    for line in text.split("\n"):
        if line.startswith(f"| {label} |") and target in [norm_value(c) for c in line.strip("| ").split(" | ")]:
            return True
    return False


def fact_gold(f: dict, children: list) -> list[str]:
    value, label = norm_value(f["value"]), f["row_label"].lower()
    gold = []
    for c in children:
        if c.metadata["ticker"] != f["ticker"]:
            continue
        if c.metadata["content_type"] == "table" and row_matches(c.text, f["row_label"], f["value"]):
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


# --------------------------------------------------------------------------- build

def build(tickers: list[str] | None, per_statement: int = 2, yoy_per_filing: int = 2,
          max_cross_quarter: int = 24) -> list[dict]:
    rng = random.Random(SEED)
    files = sorted(CHUNKS_DIR.glob("*/*.jsonl"))
    if tickers:
        files = [f for f in files if f.parent.name in {t.upper() for t in tickers}]
    children = [n for f in files for n in load_nodes(f) if n.metadata["node_type"] == "child"]
    facts = [fact for f in files for fact in extract_facts(f)]

    questions = (q_numeric(facts, rng, per_statement) + q_yoy(facts, rng, yoy_per_filing)
                 + q_cross_quarter(facts, rng, max_cross_quarter) + q_cross_company(facts, rng))
    for q in questions:
        q["gold_groups"] = [fact_gold(f, children) for f in q["facts"]]
        q["ticker"] = ",".join(dict.fromkeys(f["ticker"] for f in q["facts"]))
        q["statement"] = q["facts"][0]["statement"]
        q["units"] = q["facts"][0]["units"]
        q["facts"] = [{k: f[k] for k in ("filing_id", "row_label", "value", "phrase", "prior_value")} for f in q["facts"]]
    questions = [q for q in questions if all(q["gold_groups"])]

    spec = yaml.safe_load(open(CONFIG_DIR / "eval_narrative.yaml", encoding="utf-8"))
    for item in spec["questions"]:
        questions.append({"type": "narrative", "question": item["question"],
                          "ticker": ",".join(item["relevant"].get("tickers", []) or
                                             sorted({f.split("_")[0] for f in item["relevant"].get("filing_ids", [])})),
                          "rule": item["relevant"], "gold_groups": [narrative_gold(item["relevant"], children)]})

    for i, q in enumerate(questions):
        q["qid"] = f"q{i:04d}"
        q["gold_ids"] = sorted({g for grp in q["gold_groups"] for g in grp})  # union, for single-list metrics
    return questions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", nargs="+")
    parser.add_argument("--out", default="retrieval_v1.jsonl")
    args = parser.parse_args()
    questions = build(args.tickers)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out = EVAL_DIR / args.out
    with open(out, "w", encoding="utf-8") as fh:
        for q in questions:
            fh.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"{len(questions)} questions {dict(Counter(q['type'] for q in questions))} -> {out}")
    empty = [q["qid"] for q in questions if not all(q["gold_groups"])]
    print(f"questions with an empty gold group: {len(empty)} {empty[:10]}")


if __name__ == "__main__":
    main()
