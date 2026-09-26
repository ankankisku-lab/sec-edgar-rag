"""End-to-end answer accuracy on numeric questions with known answers.

Per question: did the gold chunk (or its parent) reach the LLM context, does the
answer state the correct value, did the model refuse, are there unverified
numbers, is it cited. Splitting "context had the answer" from "answer correct"
separates retrieval failures from generation failures.

Usage:
    python -m src.evaluation.generation --n 30
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd

from src.chunking.hierarchical import CHUNKS_DIR, load_nodes
from src.evaluation.dataset import EVAL_DIR
from src.generation.generator import answer, ollama_running
from src.generation.numbers import NUM_RE, _value

RESULTS_DIR = EVAL_DIR / "results"
REFUSAL = "do not contain this information"


def value_stated(answer_text: str, value: str) -> bool:
    """The expected table value appears in the answer (exact, or rounded to billions)."""
    target = _value(value)
    if target is None:
        return False
    target = abs(target)
    for token in NUM_RE.findall(re.sub(r"\[\d+\]", " ", answer_text)):
        v = _value(token)
        if v is None:
            continue
        v = abs(v)
        if v == target or any(round(target / 1000, d) == v for d in (1, 2)):
            return True
    return False


def gold_parent_ids(q: dict, child_meta: dict[str, dict]) -> set[str]:
    return {child_meta[g]["parent_id"] for g in q["gold_ids"] if g in child_meta}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--version", default="G1")
    args = parser.parse_args()

    questions = [json.loads(line) for line in open(EVAL_DIR / "retrieval_v0.jsonl", encoding="utf-8")]
    numeric = [q for q in questions if q["type"] == "numeric"]
    rng = random.Random(11)
    by_stratum: dict[tuple, list] = {}
    for q in numeric:
        by_stratum.setdefault((q["ticker"], q["statement"]), []).append(q)
    sample = []
    while len(sample) < args.n and any(by_stratum.values()):
        for stratum in list(by_stratum):
            if by_stratum[stratum] and len(sample) < args.n:
                sample.append(by_stratum[stratum].pop(rng.randrange(len(by_stratum[stratum]))))

    child_meta = {n.node_id: n.metadata for f in CHUNKS_DIR.glob("*/*.jsonl") for n in load_nodes(f)
                  if n.metadata["node_type"] == "child" and f.parent.name in {q["ticker"] for q in sample}}

    rows = []
    # Load and exercise the embedder + reranker BEFORE the LLM is loaded, so Ollama sizes
    # its GPU layers around their steady-state memory instead of being squeezed later.
    from src.retrieval.reranker import retrieve_reranked
    retrieve_reranked("warm-up: what was Apple's net income?")
    with ollama_running():
        answer("warm-up: what was Apple's net income?")
        for i, q in enumerate(sample, 1):
            r = answer(q["question"])
            source_ids = {s["node_id"] for s in r["sources"]}
            gold_ctx = bool(source_ids & (set(q["gold_ids"]) | gold_parent_ids(q, child_meta)))
            row = {"qid": q["qid"], "ticker": q["ticker"], "statement": q["statement"],
                   "gold_in_context": gold_ctx, "correct": value_stated(r["answer"], q["answer_value"]),
                   "refused": REFUSAL in r["answer"].lower(), "unverified": len(r["unverified_numbers"]),
                   "cited": bool(re.search(r"\[\d+\]", r["answer"])), "latency_ms": r["latency_ms"],
                   "question": q["question"], "expected": q["answer_value"], "answer": r["answer"],
                   "unverified_numbers": " ".join(r["unverified_numbers"])}
            rows.append(row)
            print(f"{i:3}/{len(sample)} {'OK ' if row['correct'] else 'ERR'} ctx={int(gold_ctx)} "
                  f"{r['latency_ms']:6d}ms | {q['question'][:70]} -> {r['answer'][:80]!r}", flush=True)

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / f"{args.version}_generation.csv", index=False)
    ctx = df[df.gold_in_context]
    print(f"\n{args.version}: {len(df)} numeric questions")
    print(f"  answer correct            {df.correct.mean():.1%}")
    print(f"  gold reached the context  {df.gold_in_context.mean():.1%}")
    print(f"  correct | gold in context {ctx.correct.mean():.1%}   (generation accuracy)")
    print(f"  refused                   {df.refused.mean():.1%}   (of which gold was in context: {(df.refused & df.gold_in_context).sum()})")
    print(f"  answers w/ unverified num {(df.unverified > 0).mean():.1%}")
    print(f"  cited                     {df.cited.mean():.1%}")
    print(f"  latency p50 / p95         {df.latency_ms.median():.0f} / {df.latency_ms.quantile(.95):.0f} ms")


if __name__ == "__main__":
    main()
