"""Retrieval metrics on the evaluation set, logged to an experiments table.

Metrics (binary relevance, per question, then averaged):
    precision@3  |top3 & gold| / 3
    recall@5     |top5 & gold| / min(|gold|, 5)   -- capped, so a question with 12
                                                     equally-correct chunks isn't
                                                     penalised for "missing" 7
    hit@k        any gold chunk in the top k (k = 1, 3, 5, 10)
    mrr@10       1 / rank of the first gold chunk (0 if none in the top 10)
    ndcg@10      binary-gain nDCG
  multi-fact questions (cross-quarter, cross-company) have one gold group per fact:
    all_hit@k    every fact has a relevant chunk in the top k (k = 5, 10)
    fact_recall@10  share of facts with a relevant chunk in the top 10

Usage:
    python -m src.evaluation.retrieval --retriever dense --version V1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from datetime import datetime

import pandas as pd

from src.config import load_config
from src.evaluation.dataset import EVAL_DIR

RESULTS_DIR = EVAL_DIR / "results"
EXPERIMENTS_CSV = RESULTS_DIR / "experiments.csv"
K = 10


def results_dir(dataset: str):
    """v0 results live directly in results/ (history); later datasets get a subfolder."""
    return RESULTS_DIR if dataset == "retrieval_v0.jsonl" else RESULTS_DIR / dataset.rsplit(".", 1)[0]


def score(ranked: list[str], gold: set[str] | list[list[str]]) -> dict:
    """`gold` is a set of relevant ids, or a list of groups (one per required fact)."""
    groups = [set(g) for g in gold] if isinstance(gold, list) else [set(gold)]
    gold = set().union(*groups)
    rel = [nid in gold for nid in ranked[:K]]
    first = next((i + 1 for i, r in enumerate(rel) if r), None)
    dcg = sum(1 / math.log2(i + 2) for i, r in enumerate(rel) if r)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(gold), K)))
    facts_hit = {k: [bool(set(ranked[:k]) & g) for g in groups] for k in (5, 10)}
    return {
        "precision@3": sum(rel[:3]) / 3,
        "recall@5": sum(rel[:5]) / min(len(gold), 5),
        **{f"hit@{k}": float(any(rel[:k])) for k in (1, 3, 5, 10)},
        "mrr@10": 1 / first if first else 0.0,
        "ndcg@10": dcg / idcg if idcg else 0.0,
        **{f"all_hit@{k}": float(all(h)) for k, h in facts_hit.items()},
        "fact_recall@10": sum(facts_hit[10]) / len(groups),
        "first_rank": first,
    }


def get_retriever(name: str, **options):
    """dense | bm25 | hybrid (options: fusion=relative|rrf, alpha=float)."""
    if name == "dense":
        from src.retrieval.dense import retrieve
        return retrieve
    if name == "bm25":
        from src.retrieval.hybrid import retrieve_bm25
        return retrieve_bm25
    if name == "rerank":
        from src.retrieval.reranker import retrieve_reranked
        return lambda q, top_k=K: retrieve_reranked(q, top_k=top_k, **options)
    if name == "hybrid":
        from src.retrieval.hybrid import retrieve_hybrid
        return lambda q, top_k=K: retrieve_hybrid(q, top_k=top_k, **options)
    raise ValueError(f"unknown retriever {name!r}")


def split_of(qid: str) -> str:
    """Deterministic 50/50 dev/test split: tune settings on dev, report on test."""
    return "dev" if int(hashlib.md5(qid.encode()).hexdigest(), 16) % 2 == 0 else "test"


def evaluate(retriever_name: str, dataset: str, **options) -> tuple[pd.DataFrame, list[dict]]:
    questions = [json.loads(line) for line in open(EVAL_DIR / dataset, encoding="utf-8")]
    retrieve = get_retriever(retriever_name, **options)
    retrieve("warm-up query")  # load models / open connections outside the timed loop
    rows, traces = [], []
    for q in questions:
        t0 = time.perf_counter()
        hits = retrieve(q["question"], top_k=K)
        ms = (time.perf_counter() - t0) * 1000
        ranked = [h.node.node_id for h in hits]
        rows.append({"qid": q["qid"], "type": q["type"], "split": split_of(q["qid"]),
                     "ticker": q["ticker"], "latency_ms": ms,
                     **score(ranked, q.get("gold_groups") or [q["gold_ids"]])})
        traces.append({"qid": q["qid"], "question": q["question"], "ranked": ranked,
                       "scores": [round(float(h.score), 4) for h in hits], "n_gold": len(q["gold_ids"])})
    return pd.DataFrame(rows), traces


METRICS = ["precision@3", "recall@5", "hit@1", "hit@3", "hit@5", "hit@10", "mrr@10", "ndcg@10",
           "all_hit@5", "all_hit@10", "fact_recall@10"]


def summarize(df: pd.DataFrame) -> dict:
    out = {m: round(df[m].mean(), 3) for m in METRICS}
    for t, g in df.groupby("type"):
        for m in ("precision@3", "mrr@10", "hit@10", "all_hit@10"):
            out[f"{t}_{m}"] = round(g[m].mean(), 3)
    out["latency_p50_ms"] = round(df["latency_ms"].median(), 1)
    out["latency_p95_ms"] = round(df["latency_ms"].quantile(0.95), 1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--retriever", default="dense")
    parser.add_argument("--version", required=True, help="experiment label, e.g. V1")
    parser.add_argument("--dataset", default="retrieval_v1.jsonl")
    parser.add_argument("--notes", default="")
    parser.add_argument("--fusion", choices=["relative", "rrf"])
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--pool", help="rerank candidate pool name (configs/retrieval.yaml)")
    parser.add_argument("--no-linearize", action="store_true", help="rerank on markdown tables")
    args = parser.parse_args()

    options = {k: v for k, v in (("fusion", args.fusion), ("alpha", args.alpha), ("pool", args.pool))
               if v is not None}
    if args.retriever == "rerank":
        options["linearize"] = not args.no_linearize
    df, traces = evaluate(args.retriever, args.dataset, **options)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run = f"{args.version}_{args.retriever}"
    out_dir = results_dir(args.dataset)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{run}_per_query.csv", index=False)
    with open(out_dir / f"{run}_traces.jsonl", "w", encoding="utf-8") as fh:
        for t in traces:
            fh.write(json.dumps(t, ensure_ascii=False) + "\n")

    summary = {"version": args.version, "retriever": args.retriever, "dataset": args.dataset, **options,
               "embedding": load_config("embedding")["model_name"], "n_questions": len(df),
               "timestamp": datetime.now().isoformat(timespec="seconds"), **summarize(df), "notes": args.notes}
    log = pd.read_csv(EXPERIMENTS_CSV) if EXPERIMENTS_CSV.exists() else pd.DataFrame()
    if len(log):
        same = (log["version"] == args.version) & (log["retriever"] == args.retriever) &                (log["dataset"].fillna("retrieval_v0.jsonl") == args.dataset)
        log = log[~same]
    pd.concat([log, pd.DataFrame([summary])], ignore_index=True).to_csv(EXPERIMENTS_CSV, index=False)

    print(f"\n{run} on {len(df)} questions ({df.type.value_counts().to_dict()})")
    print(pd.DataFrame([{k: v for k, v in summarize(df).items()}]).T.to_string(header=False))


if __name__ == "__main__":
    main()
