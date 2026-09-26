"""Per-stage latency of recent questions, read back from Phoenix's REST API.

For each `rag.answer` trace, sums span durations by stage:
    decompose   query.decompose (router + LLM or cache)
    pool        retrieve.candidate_pool (BM25 top-60 in Qdrant)
    rerank      rerank.cross_encoder (BGE, fp16)
    llm         OpenInference LLM spans (Qwen3-4B via Ollama)
    calc_verify answer.calc_and_verify
    other       remainder of the root span (parent expansion, prompt assembly, ...)
Nested spans of the same stage are not double counted (only the outermost is summed).

Usage:
    python -m src.observability.latency_report --last 5
"""
from __future__ import annotations

import argparse
from datetime import datetime

import pandas as pd
import requests

from src.config import load_config

STAGES = {"query.decompose": "decompose", "retrieve.candidate_pool": "pool", "rerank.cross_encoder": "rerank",
          "answer.calc_and_verify": "calc_verify"}


def fetch_spans(limit: int = 2000) -> pd.DataFrame:
    cfg = load_config("observability")["tracing"]
    base = cfg["endpoint"].rsplit("/v1/traces", 1)[0]
    rows, cursor = [], None
    while len(rows) < limit:
        params = {"limit": min(500, limit - len(rows))}
        if cursor:
            params["cursor"] = cursor
        r = requests.get(f"{base}/v1/projects/{cfg['project_name']}/spans", params=params, timeout=30)
        r.raise_for_status()
        body = r.json()
        rows += body["data"]
        cursor = body.get("next_cursor")
        if not cursor:
            break
    df = pd.DataFrame([{"trace": s["context"]["trace_id"], "span": s["context"]["span_id"], "parent": s["parent_id"],
                        "name": s["name"], "kind": s["span_kind"],
                        "start": datetime.fromisoformat(s["start_time"]), "end": datetime.fromisoformat(s["end_time"]),
                        "input": (s.get("attributes") or {}).get("input.value")} for s in rows])
    df["ms"] = (df.end - df.start).dt.total_seconds() * 1000
    return df


def stage_of(row) -> str | None:
    if row["name"] in STAGES:
        return STAGES[row["name"]]
    if row["kind"] == "LLM":
        return "llm"
    return None


def breakdown(df: pd.DataFrame, last: int) -> pd.DataFrame:
    roots = df[df.name == "rag.answer"].sort_values("start").tail(last)
    out = []
    for _, root in roots.iterrows():
        spans = df[df.trace == root.trace].copy()
        spans["stage"] = spans.apply(stage_of, axis=1)
        by_id = spans.set_index("span")
        totals = {s: 0.0 for s in ("decompose", "pool", "rerank", "llm", "calc_verify")}
        for _, s in spans[spans.stage.notna()].iterrows():
            # count only outermost stage spans: an LLM call inside query.decompose belongs to
            # "decompose", and a nested Ollama.chat under Ollama.predict is not counted twice
            p, nested = s.parent, False
            while p in by_id.index:
                if pd.notna(by_id.loc[p, "stage"]):
                    nested = True
                    break
                p = by_id.loc[p, "parent"]
            if not nested:
                totals[s.stage] += s.ms
        totals["other"] = max(root.ms - sum(totals.values()), 0.0)
        out.append({"question": (root.input or "")[:60], "total_ms": round(root.ms), **{k: round(v) for k, v in totals.items()}})
    return pd.DataFrame(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--last", type=int, default=5)
    args = parser.parse_args()
    table = breakdown(fetch_spans(), args.last)
    pd.set_option("display.width", 200)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
