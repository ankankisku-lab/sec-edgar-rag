"""End-to-end answer accuracy on questions with known answers (all types but narrative).

Per question: did every required fact reach the LLM context, are the expected
values stated, is the requested calculation right (change or % change, computed
by <calc> in Python), refusals, unverified numbers, citations. Splitting "the
context had the facts" from "the answer is right" separates retrieval failures
from generation failures.

Usage:
    python -m src.evaluation.generation --version G2
"""
from __future__ import annotations

import argparse
import json
import random
import re

import pandas as pd

from src.chunking.hierarchical import CHUNKS_DIR, load_nodes
from src.evaluation.dataset import EVAL_DIR
from src.generation.generator import answer, ollama_running
from src.generation.numbers import NUM_RE, _value

RESULTS_DIR = EVAL_DIR / "results"
REFUSAL = "do not contain this information"
SAMPLE = {"numeric": 12, "yoy": 8, "pct_change": 6, "cross_quarter": 8, "cross_company": 6}


def answer_numbers(text: str) -> list[float]:
    vals = (_value(t) for t in NUM_RE.findall(re.sub(r"\[\d+\]", " ", text)))
    return [abs(v) for v in vals if v is not None]


def value_stated(answer_text: str, value: str) -> bool:
    """The expected table value appears in the answer (exact, or rounded to billions)."""
    target = _value(value)
    if target is None:
        return False
    target = abs(target)
    return any(v == target or any(round(target / 1000, d) == v for d in (1, 2)) for v in answer_numbers(answer_text))


def change_stated(answer_text: str, change: float) -> bool:
    c = abs(change)
    return any(abs(v - c) < 0.5 or any(round(c / 1000, d) == v for d in (1, 2)) for v in answer_numbers(answer_text))


def pct_stated(answer_text: str, pct: float) -> bool:
    return any(abs(v - abs(pct)) <= 0.06 for v in answer_numbers(answer_text))


def score_answer(q: dict, text: str) -> dict:
    values_ok = all(value_stated(text, v) for v in q["answer_values"])
    if q["type"] == "pct_change":
        calc_ok = pct_stated(text, q["expected_pct"])
    elif "expected_change" in q:
        calc_ok = change_stated(text, q["expected_change"])
    else:
        calc_ok = None
    correct = calc_ok if q["type"] == "pct_change" else (values_ok and calc_ok is not False)
    return {"values_ok": values_ok, "calc_ok": calc_ok, "correct": bool(correct)}


def sample_questions(questions: list[dict], plan: dict[str, int], seed: int = 11) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for qtype, n in plan.items():
        pool = [q for q in questions if q["type"] == qtype]
        out += rng.sample(pool, min(n, len(pool)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", default="G2")
    parser.add_argument("--dataset", default="retrieval_v1.jsonl")
    args = parser.parse_args()

    questions = [json.loads(line) for line in open(EVAL_DIR / args.dataset, encoding="utf-8")]
    sample = sample_questions(questions, SAMPLE)
    tickers = {t for q in sample for t in q["ticker"].split(",")}
    child_parent = {n.node_id: n.metadata["parent_id"] for f in CHUNKS_DIR.glob("*/*.jsonl") if f.parent.name in tickers
                    for n in load_nodes(f) if n.metadata["node_type"] == "child"}

    # Load and exercise the embedder + reranker BEFORE the LLM is loaded, so Ollama sizes
    # its GPU layers around their steady-state memory instead of being squeezed later.
    from src.retrieval.reranker import retrieve_reranked
    retrieve_reranked("warm-up: what was Apple's net income?")
    rows = []
    with ollama_running():
        answer("warm-up: what was Apple's net income?")
        for i, q in enumerate(sample, 1):
            r = answer(q["question"])
            in_ctx = {s["node_id"] for s in r["sources"]}
            facts_in_ctx = [bool(in_ctx & (set(g) | {child_parent[c] for c in g if c in child_parent}))
                            for g in q["gold_groups"]]
            row = {"qid": q["qid"], "type": q["type"], "ticker": q["ticker"],
                   "all_facts_in_context": all(facts_in_ctx), **score_answer(q, r["answer"]),
                   "refused": REFUSAL in r["answer"].lower(), "unverified": len(r["unverified_numbers"]),
                   "n_calcs": len(r["calculations"]), "cited": bool(re.search(r"\[\d+\]", r["answer"])),
                   "latency_ms": r["latency_ms"], "question": q["question"],
                   "expected": " | ".join(q["answer_values"]), "answer": r["answer"],
                   "unverified_numbers": " ".join(r["unverified_numbers"])}
            rows.append(row)
            print(f"{i:3}/{len(sample)} {'OK ' if row['correct'] else 'ERR'} {q['type']:13} ctx={int(row['all_facts_in_context'])} "
                  f"{r['latency_ms']:6d}ms | {q['question'][:60]} -> {r['answer'][:90]!r}", flush=True)

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / f"{args.version}_generation.csv", index=False)
    by_type = df.groupby("type").agg(n=("qid", "size"), correct=("correct", "mean"),
                                     facts_in_ctx=("all_facts_in_context", "mean"), values_ok=("values_ok", "mean"),
                                     calc_ok=("calc_ok", lambda s: s.dropna().astype(float).mean() if s.notna().any() else float("nan")),
                                     unverified=("unverified", lambda s: (s > 0).mean()), p50_ms=("latency_ms", "median"))
    print(f"\n{args.version}: {len(df)} questions\n{by_type.round(3).to_string()}")
    ctx = df[df.all_facts_in_context]
    print(f"\n  correct overall            {df.correct.mean():.1%}")
    print(f"  all facts reached context  {df.all_facts_in_context.mean():.1%}")
    print(f"  correct | facts in context {ctx.correct.mean():.1%}   (generation accuracy)")
    print(f"  refused                    {df.refused.mean():.1%}   (with facts in context: {(df.refused & df.all_facts_in_context).sum()})")
    print(f"  answers w/ unverified num  {(df.unverified > 0).mean():.1%}")
    print(f"  cited                      {df.cited.mean():.1%}")
    print(f"  latency p50 / p95          {df.latency_ms.median():.0f} / {df.latency_ms.quantile(.95):.0f} ms")


if __name__ == "__main__":
    main()
