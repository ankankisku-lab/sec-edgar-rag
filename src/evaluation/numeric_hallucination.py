"""Numerical-hallucination evaluation (Phase 17).

verify_numbers() accepts a number when it appears anywhere in the sources. That catches invented
numbers, but a figure read from the wrong cell of the right table is in the sources and passes.
This module measures both, on generation runs whose exact LLM contexts were saved
(<run>_ragas_input.jsonl next to <run>_generation.csv):

1. audit  -- every answer-keyed question gets one outcome: correct / refused / wrong and flagged by
             verification / wrong and NOT flagged (a silent numerical error). Each number of a wrong
             answer is located in the source tables: the right cell, another column in the same row,
             another row in the same column, the same line item in another table, text, or nowhere.
2. stress -- plants known errors in correct answers and records whether verify_numbers() flags them:
             invented numbers (last digit +1, +/-10%, transposed digits) vs wrong-cell numbers taken
             from the sources (same row other column, same column other row, same line item in
             another table or filing).
3. arithmetic -- for runs made without <calc> (src.evaluation.generation --no-calc), the model's own
             differences and percentages are checked against the answer key.

Usage:
    python -m src.evaluation.numeric_hallucination --runs G5 G6 G7 G8
    -> data/eval/results/phase17_audit.csv, phase17_numbers.csv, phase17_stress.csv
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter

import pandas as pd

from src.evaluation.dataset import EVAL_DIR
from src.generation.numbers import NUM_RE, _value, number_tokens, verify_numbers
from src.generation.source_view import Cell, matches, norm, parse_source, table_cells, text_numbers

RESULTS_DIR = EVAL_DIR / "results"


# ---------------------------------------------------------------------------- source tables
def parse_sources(contexts: list[str]) -> tuple[list[Cell], set[float], list[tuple[str, str]]]:
    """Numeric table cells of every source, the numbers in running text, and each source's
    (company, period) from the metadata header the LLM sees (same parser as the chat UI)."""
    parsed = [parse_source(c) for c in contexts]
    cells = [cell for s, p in enumerate(parsed) for cell in table_cells(s, p)]
    prose = {v for p in parsed for _, _, v in text_numbers(p)}
    meta = [(p["header"].get("company", ""), p["header"].get("period", "")) for p in parsed]
    return cells, prose, meta


def locate(v: float, cells: list[Cell], prose: set[float], gold: list[Cell], gold_labels: set[str],
           meta: list[tuple[str, str]]) -> str:
    """Where a stated number comes from, relative to the cells that hold the right answer."""
    hits = [c for c in cells if matches(v, c.value)]
    if any(c in gold for c in hits):
        return "right cell"
    rows = {(g.src, g.table, g.row) for g in gold}
    cols = {(g.src, g.table, g.col) for g in gold}
    tables = {(g.src, g.table) for g in gold}
    if any((c.src, c.table, c.row) in rows for c in hits):
        return "same row, other column"
    same_item = [c for c in hits if c.label in gold_labels]
    if same_item:
        gold_meta = {meta[g.src] for g in gold}
        if gold_meta and all(meta[c.src] not in gold_meta for c in same_item):
            return "same line item, other filing or period"
        return "same line item, other table"
    if any((c.src, c.table, c.col) in cols for c in hits):
        return "same column, other row"
    if any((c.src, c.table) in tables for c in hits):
        return "same table, other cell"
    if hits:
        return "other table"
    if v in prose:
        return "text"
    return "not in sources"


def gold_cells(q: dict, cells: list[Cell]) -> tuple[list[Cell], set[str]]:
    """Cells holding the answer-key values: the fact's row label with the fact's value."""
    labels = {norm(f["row_label"]) for f in q.get("facts", [])}
    keys = {abs(_value(v)) for v in q.get("answer_values", []) if _value(v) is not None}
    return [c for c in cells if c.label in labels and c.value in keys], labels


# ---------------------------------------------------------------------------- run loading
def load_run(run: str) -> list[dict]:
    """Answer-keyed questions of a run with their answer, verification result and exact contexts."""
    questions = {q["qid"]: q for q in map(json.loads, open(EVAL_DIR / "retrieval_v1.jsonl", encoding="utf-8"))}
    gen = pd.read_csv(RESULTS_DIR / f"{run}_generation.csv").set_index("qid")
    rows = []
    for rec in map(json.loads, open(RESULTS_DIR / f"{run}_ragas_input.jsonl", encoding="utf-8")):
        q = questions[rec["qid"]]
        if q["type"] == "narrative" or rec["qid"] not in gen.index:
            continue
        g = gen.loc[rec["qid"]]
        rows.append({"run": run, "q": q, "answer": rec["response"], "contexts": rec["retrieved_contexts"],
                     "correct": bool(g.correct), "refused": bool(g.refused), "values_ok": bool(g.values_ok),
                     "calc_ok": None if pd.isna(g.calc_ok) else bool(g.calc_ok),
                     "unverified": str(g.unverified_numbers).split() if g.unverified > 0 else [],
                     "facts_in_context": bool(g.all_facts_in_context)})
    return rows


def key_values(q: dict) -> list[float]:
    return [abs(_value(v)) for v in q.get("answer_values", []) if _value(v) is not None]


def is_key(v: float, q: dict) -> bool:
    """A number the answer key expects: a quoted value, the change or the percentage change."""
    if any(matches(v, k) for k in key_values(q)):
        return True
    if "expected_change" in q and (abs(v - abs(q["expected_change"])) < 0.5 or matches(v, abs(q["expected_change"]))):
        return True
    return "expected_pct" in q and abs(v - abs(q["expected_pct"])) <= 0.06


# ---------------------------------------------------------------------------- 1. audit
def audit(rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_q, per_num = [], []
    for r in rows:
        q, sources = r["q"], "\n".join(r["contexts"])
        cells, prose, meta = parse_sources(r["contexts"])
        gold, labels = gold_cells(q, cells)
        allowed = {abs(v) for v in (_value(t) for t in NUM_RE.findall(sources + " " + q["question"])) if v is not None}
        flagged = bool(r["unverified"])
        checked = Counter(number_tokens(r["answer"]))
        for token, v in number_tokens(r["answer"], keep_percent=True):
            if checked[(token, v)] > 0:
                checked[(token, v)] -= 1
                in_sources = v in allowed or any(matches(v, a) for a in allowed)
                origin = "unverified" if token in r["unverified"] else "source" if in_sources else "calculation"
            else:
                origin = "unchecked percentage"  # <= 31%: skipped by verification's day-of-month rule
            per_num.append({"run": r["run"], "qid": q["qid"], "type": q["type"], "token": token, "value": v,
                            "key": is_key(v, q), "origin": origin, "answer_correct": r["correct"],
                            "location": "calculation" if origin == "calculation"
                            else locate(v, cells, prose, gold, labels, meta)})
        if r["correct"]:
            outcome, cause = "correct", ""
        elif r["refused"]:
            outcome, cause = "refused", "facts not in context" if not r["facts_in_context"] else "refused with facts in context"
        else:
            outcome = "wrong, flagged" if flagged else "wrong, not flagged"
            wrong = [n for n in per_num if n["run"] == r["run"] and n["qid"] == q["qid"] and not n["key"]
                     and n["origin"] in ("source", "unverified")]
            if r["values_ok"] and r["calc_ok"] is False:
                cause = "values right, change missing or wrong"
            elif not r["facts_in_context"]:
                cause = "facts not in context"
            elif wrong:
                cause = wrong[0]["location"]
            else:
                cause = "value missing from answer"
        per_q.append({"run": r["run"], "qid": q["qid"], "type": q["type"], "outcome": outcome, "cause": cause,
                      "flagged": flagged, "facts_in_context": r["facts_in_context"], "gold_cells_found": bool(gold),
                      "question": q["question"], "expected": " | ".join(q.get("answer_values", [])), "answer": r["answer"]})
    return pd.DataFrame(per_q), pd.DataFrame(per_num)


# ---------------------------------------------------------------------------- 2. stress test
def fmt_like(original: str, value: float) -> str:
    """Write a planted number in the same style as the token it replaces ($, commas, decimals, parentheses)."""
    body = original.strip("$()").replace(",", "")
    decimals = len(body.split(".")[1]) if "." in body else 0
    text = f"{value:,.{decimals}f}" if ("," in original or value >= 1000) else f"{value:.{decimals}f}"
    if original.lstrip("$").startswith("("):
        text = f"({text})"
    return "$" + text if original.startswith("$") else text


def transposed(value: float) -> float | None:
    digits = list(str(int(value)))
    for i in range(len(digits) - 1, 0, -1):
        if digits[i] != digits[i - 1]:
            digits[i], digits[i - 1] = digits[i - 1], digits[i]
            return float("".join(digits))
    return None


def plants(v: float, target: Cell | None, cells: list[Cell], forbidden: set[float]) -> list[tuple[str, str, float]]:
    """(kind, name, planted value) for one correctly quoted value."""
    step = 1.0 if v.is_integer() else 0.01
    out = [("invented", "last digit +1", v + step), ("invented", "+10%", round(v * 1.1, 0 if v.is_integer() else 2)),
           ("invented", "-10%", round(v * 0.9, 0 if v.is_integer() else 2))]
    t = transposed(v) if v.is_integer() else None
    if t is not None:
        out.append(("invented", "transposed digits", t))
    if target is not None:
        ok = lambda c: c.value != v and c.value not in forbidden and c.value > 0  # noqa: E731
        same_row = [c for c in cells if (c.src, c.table, c.row) == (target.src, target.table, target.row) and ok(c)]
        same_col = sorted((c for c in cells if (c.src, c.table, c.col) == (target.src, target.table, target.col)
                           and c.row != target.row and ok(c)), key=lambda c: abs(c.row - target.row))
        same_label = [c for c in cells if c.label == target.label and (c.src, c.table) != (target.src, target.table) and ok(c)]
        for name, pool in (("same row, other column", same_row), ("same column, other row", same_col),
                           ("same line item, other table", same_label)):
            if pool:
                out.append(("wrong cell", name, pool[0].value))
    return [(k, n, p) for k, n, p in out if p > 0 and p != v]


def stress(rows: list[dict]) -> pd.DataFrame:
    out = []
    for r in rows:
        if not r["correct"] or r["unverified"]:
            continue
        q, sources = r["q"], "\n".join(r["contexts"])
        # Numbers the live run accepted but the sources don't contain are <calc> results.
        calcs = [{"value": abs(_value(t))} for t in verify_numbers(r["answer"], sources, q["question"])]
        if verify_numbers(r["answer"], sources, q["question"], calcs):
            continue
        cells, _, _ = parse_sources(r["contexts"])
        gold, _ = gold_cells(q, cells)
        keys = set(key_values(q))
        for k in keys:
            m = next((m for m in NUM_RE.finditer(r["answer"]) if _value(m.group()) is not None
                      and abs(_value(m.group())) == k), None)
            if m is None:
                continue  # the value was quoted in billions; plant only into exact quotes
            target = next((g for g in gold if g.value == k), None)
            for kind, name, p in plants(k, target, cells, keys):
                planted = r["answer"][:m.start()] + fmt_like(m.group(), p) + r["answer"][m.end():]
                flagged = bool(verify_numbers(planted, sources, q["question"], calcs))
                out.append({"run": r["run"], "qid": q["qid"], "type": q["type"], "kind": kind, "plant": name,
                            "original": m.group(), "planted": fmt_like(m.group(), p), "flagged": flagged})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------- 3. self-arithmetic
def consistent(d: float, operands: list[float]) -> bool:
    """d is the difference or % change of two numbers the answer itself quotes. Operands are absolute
    values, so a change across zero (9,541 vs (1,165)) is their sum; $4.6 billion - $2.2 billion may be
    stated as 2,400 million."""
    for a in operands:
        for b in operands:
            if a == b:
                continue
            for diff in (abs(a - b), a + b):
                if abs(d - diff) < 0.5 or matches(d, diff) or abs(d - diff * 1000) < 0.5 \
                        or (b and abs(d - diff / b * 100) <= 0.06):
                    return True
    return False


def arithmetic(numbers: pd.DataFrame, rows: list[dict]) -> pd.DataFrame:
    """Per question that asks for a change or % change: how the stated change relates to the key."""
    out = []
    for r in rows:
        q = r["q"]
        if "expected_change" not in q and "expected_pct" not in q:
            continue
        exp = abs(q.get("expected_pct", q.get("expected_change")))
        nums = numbers[(numbers.run == r["run"]) & (numbers.qid == q["qid"])]
        derived = nums[nums.origin.isin(["calculation", "unverified", "unchecked percentage"])]
        operands = list(nums[nums.origin == "source"].value)
        if r["refused"]:
            status = "refused"
        elif r["calc_ok"]:
            status = "right"
        elif not len(derived):
            status = "no change stated"
        elif r["values_ok"]:
            # The right values are quoted, so a wrong change is either rounding, a lost sign (the
            # change of the unsigned values: 9,792 - 4,886 for (4,886) vs 9,792) or the arithmetic.
            if any(abs(v - exp) <= 0.05 * exp for v in derived.value):
                status = "rounded or approximate"
            elif any(consistent(v, operands) for v in derived.value):
                status = "sign lost"
            else:
                status = "arithmetic error"
        elif any(consistent(v, operands) for v in derived.value):
            status = "arithmetic right, values wrong"
        elif any(abs(v - exp) <= 0.05 * exp for v in derived.value):
            status = "rounded or approximate"
        else:
            status = "arithmetic error"
        out.append({"run": r["run"], "qid": q["qid"], "type": q["type"], "expected": exp, "status": status,
                    "derived_numbers": " ".join(derived.token), "flagged_by_verification": bool(r["unverified"])})
    return pd.DataFrame(out)


def mcnemar(a: pd.Series, b: pd.Series) -> tuple[int, int, float]:
    fixed, broke = int((~a & b).sum()), int((a & ~b).sum())
    n, k = fixed + broke, min(fixed, broke)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return fixed, broke, p


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", default=["G5", "G6", "G7", "G8"])
    parser.add_argument("--pair", nargs=2, default=["G7", "G8"], metavar=("CALC_RUN", "NO_CALC_RUN"),
                        help="the <calc> ablation pair (same questions and index, prompt differs)")
    args = parser.parse_args()
    runs = [r for r in args.runs if (RESULTS_DIR / f"{r}_ragas_input.jsonl").exists()]
    rows = [row for run in runs for row in load_run(run)]
    audit_df, numbers = audit(rows)
    stress_df = stress(rows)
    arith = arithmetic(numbers, rows)
    audit_df.to_csv(RESULTS_DIR / "phase17_audit.csv", index=False)
    numbers.to_csv(RESULTS_DIR / "phase17_numbers.csv", index=False)
    stress_df.to_csv(RESULTS_DIR / "phase17_stress.csv", index=False)
    arith.to_csv(RESULTS_DIR / "phase17_arithmetic.csv", index=False)

    pd.set_option("display.width", 200)
    print("Outcome per answer-keyed question:")
    print(pd.crosstab(audit_df.run, audit_df.outcome, margins=True).to_string(), "\n")
    wrong = audit_df[audit_df.outcome.str.startswith("wrong")]
    if len(wrong):
        print("Cause of wrong answers:")
        print(pd.crosstab([wrong.run, wrong.outcome], wrong.cause).to_string(), "\n")
    if len(stress_df):
        print("Planted errors flagged by verify_numbers():")
        s = stress_df.groupby(["kind", "plant"]).flagged.agg(n="size", flagged="mean")
        print(s.to_string(float_format=lambda x: f"{x:.1%}"), "\n")
    if len(arith):
        print("Stated change / % change on questions that ask for one:")
        print(pd.crosstab(arith.run, arith.status, margins=True).to_string(), "\n")
    unchecked = numbers[numbers.origin == "unchecked percentage"]
    print(f"Percentages <= 31% that verification never checks: {len(unchecked)} in "
          f"{unchecked.groupby(['run', 'qid']).ngroups} answers "
          f"({', '.join(f'{r}: {n}' for r, n in unchecked.groupby('run').size().items()) or 'none'})\n")
    a, b = args.pair
    if {a, b} <= set(runs):
        ja = audit_df[audit_df.run == a].set_index("qid").outcome.eq("correct")
        jb = audit_df[audit_df.run == b].set_index("qid").outcome.eq("correct")
        j = pd.concat([ja, jb], axis=1, join="inner", keys=[a, b])
        fixed, broke, p = mcnemar(j[a], j[b])
        print(f"{a} -> {b}: correct {j[a].mean():.1%} -> {j[b].mean():.1%} "
              f"({fixed} fixed, {broke} broken, McNemar p = {p:.4f})")


if __name__ == "__main__":
    main()
