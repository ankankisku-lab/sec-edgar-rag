"""Phase 16: scale the index 24 -> 56 -> 104 -> 254 -> 726 filings and re-measure.

Each stage adds the next companies (AAPL, MSFT, NVDA -- the eval companies -- first,
then by ticker; see data/eval/scaling_stages.json), embeds and indexes only what is new
(the embedder and indexer skip finished filings), records index size / memory / time,
and re-runs retrieval experiments on the unchanged eval set. Every added filing is a
potential distractor for the eval questions, so this measures whether retrieval
quality holds as the corpus grows 30x.

Usage:
    python -m src.pipeline.scale                  # all stages
    python -m src.pipeline.scale --stages S50 S100
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime

import pandas as pd

from src.config import DATA_DIR, load_config
from src.evaluation.dataset import EVAL_DIR

RESULTS = EVAL_DIR / "results" / "scaling.csv"
EXPERIMENTS = [("bm25", "V2", []), ("rerank", "V4", ["--pool", "bm25_60"]), ("decomposed", "V5", [])]


def sh(args: list[str]) -> tuple[float, str]:
    t0 = time.perf_counter()
    out = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if out.returncode != 0:
        raise RuntimeError(f"{' '.join(args[:4])} failed:\n{out.stdout[-1500:]}\n{out.stderr[-1500:]}")
    return time.perf_counter() - t0, out.stdout


def qdrant_footprint() -> dict:
    import requests
    cfg = load_config("qdrant")
    info = requests.get(f"{cfg['url']}/collections/{cfg['collection']}", timeout=30).json()["result"]
    mem = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", "sec-rag-qdrant"],
                         capture_output=True, text=True).stdout.strip().split(" / ")[0]
    disk = subprocess.run(["docker", "exec", "sec-rag-qdrant", "du", "-sh", "/qdrant/storage"],
                          capture_output=True, text=True).stdout.split()[:1]
    parents_db = DATA_DIR / "index" / "parents.sqlite"
    return {"points": info["points_count"], "segments": info.get("segments_count"),
            "qdrant_mem": mem, "qdrant_disk": disk[0] if disk else None,
            "parents_db_mb": round(parents_db.stat().st_size / 2**20, 1) if parents_db.exists() else None}


def run_stage(name: str, tickers: list[str], n_filings: int) -> dict:
    py = sys.executable
    print(f"\n===== {name}: {n_filings} filings, {len(tickers)} companies", flush=True)
    embed_s, _ = sh([py, "-m", "src.embeddings.embedder", "--tickers", *tickers])
    index_s, index_out = sh([py, "-m", "src.retrieval.qdrant_index", "--tickers", *tickers])
    row = {"stage": name, "filings": n_filings, "companies": len(tickers),
           "embed_s": round(embed_s), "index_s": round(index_s), **qdrant_footprint(),
           "timestamp": datetime.now().isoformat(timespec="seconds")}
    print(f"   embedded in {embed_s:.0f}s, indexed in {index_s:.0f}s | {index_out.strip().splitlines()[-1][:160]}", flush=True)
    for retriever, version, extra in EXPERIMENTS:
        _, out = sh([py, "-m", "src.evaluation.retrieval", "--retriever", retriever, "--version", f"{version}@{name}",
                     *extra, "--notes", f"Phase 16 scaling stage {name}: {n_filings} filings"])
        metrics = dict(line.split(maxsplit=1) for line in out.splitlines()
                       if line.split() and line.split()[0] in ("mrr@10", "precision@3", "all_hit@10", "latency_p50_ms"))
        for k, v in metrics.items():
            row[f"{version}_{k}"] = float(v)
        print(f"   {version}@{name}: MRR {metrics.get('mrr@10')}  P@3 {metrics.get('precision@3')}  "
              f"all-hit@10 {metrics.get('all_hit@10')}  p50 {metrics.get('latency_p50_ms')} ms", flush=True)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stages", nargs="+", help="subset of stage names, e.g. S50 S100")
    args = parser.parse_args()
    plan = json.load(open(EVAL_DIR / "scaling_stages.json", encoding="utf-8"))
    rows = pd.read_csv(RESULTS).to_dict("records") if RESULTS.exists() else []
    for stage in plan["stages"]:
        if args.stages and stage["name"] not in args.stages:
            continue
        row = run_stage(stage["name"], plan["order"][:stage["companies"]], stage["filings"])
        rows = [r for r in rows if r["stage"] != stage["name"]] + [row]
        pd.DataFrame(rows).to_csv(RESULTS, index=False)
    print(f"\nsaved {RESULTS}")


if __name__ == "__main__":
    main()
