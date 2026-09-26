"""Choose the hybrid fusion method on the dev split, report it on the test split.

Usage:
    python -m src.evaluation.fusion_sweep
"""
from __future__ import annotations

import pandas as pd

from src.evaluation.dataset import EVAL_DIR
from src.evaluation.retrieval import evaluate

VARIANTS = [
    {"fusion": "relative", "alpha": 0.3},
    {"fusion": "relative", "alpha": 0.5},
    {"fusion": "relative", "alpha": 0.7},
    {"fusion": "rrf"},
]
SELECT_BY = "mrr@10"


def pool_recall() -> pd.DataFrame:
    """How often the candidate pool handed to a reranker contains a relevant chunk."""
    import json
    from src.retrieval.dense import retrieve as dense
    from src.retrieval.hybrid import retrieve_bm25

    rows = []
    for line in open(EVAL_DIR / "retrieval_v0.jsonl", encoding="utf-8"):
        q = json.loads(line)
        gold = set(q["gold_ids"])
        d = [h.node.node_id for h in dense(q["question"], top_k=60)]
        b = [h.node.node_id for h in retrieve_bm25(q["question"], top_k=60)]
        for name, pool in [("dense top-60", d), ("bm25 top-60", b),
                           ("union dense30+bm25-30", list(dict.fromkeys(d[:30] + b[:30])))]:
            rows.append({"pool": name, "type": q["type"], "size": len(pool), "hit": float(bool(gold & set(pool))),
                         "recall": len(gold & set(pool)) / min(len(gold), len(pool))})
    df = pd.DataFrame(rows)
    table = df.groupby("pool").agg(avg_size=("size", "mean"), hit=("hit", "mean"), recall=("recall", "mean"))
    by_type = df.pivot_table(index="pool", columns="type", values="hit", aggfunc="mean").add_prefix("hit_")
    return table.join(by_type).round(3)


RERANK_VARIANTS = [{"pool": p, "linearize": lin} for lin in (False, True)
                   for p in ("bm25_60", "union_30_30", "union_45_15")]


def main() -> None:
    import sys
    if "--pools" in sys.argv:
        table = pool_recall()
        print(table.to_string())
        table.to_csv(EVAL_DIR / "results" / "candidate_pools.csv")
        return
    rerank = "--rerank" in sys.argv
    retriever, variants = ("rerank", RERANK_VARIANTS) if rerank else ("hybrid", VARIANTS)
    rows = []
    for v in variants:
        df, _ = evaluate(retriever, "retrieval_v0.jsonl", **v)
        for split, g in [("dev", df[df.split == "dev"]), ("test", df[df.split == "test"])]:
            rows.append({**v, "split": split, "n": len(g),
                         **{m: round(g[m].mean(), 3) for m in ("precision@3", "recall@5", "mrr@10", "hit@10")},
                         "numeric_p@3": round(g[g.type == "numeric"]["precision@3"].mean(), 3),
                         "narrative_p@3": round(g[g.type == "narrative"]["precision@3"].mean(), 3),
                         "latency_p50_ms": round(g["latency_ms"].median())})
    table = pd.DataFrame(rows)
    if "alpha" in table:
        table["alpha"] = table["alpha"].fillna("-")
    print(table.to_string(index=False))
    dev = table[table.split == "dev"]
    best = dev.loc[dev[SELECT_BY].idxmax()]
    keys = dict.fromkeys(k for v in variants for k in v)
    print(f"\nselected on dev by {SELECT_BY}: " + ", ".join(f"{k}={best[k]}" for k in keys))
    table.to_csv(EVAL_DIR / "results" / ("rerank_sweep.csv" if rerank else "fusion_sweep.csv"), index=False)


if __name__ == "__main__":
    main()
