"""Evidence layer for the architecture report: every number in the report is computed here
from files in the repository (no hand-typed results). Read-only."""
from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "data" / "eval" / "results"
EVAL = ROOT / "data" / "eval"
V1_DIR = RES / "retrieval_v1"


def rel(p: Path | str) -> str:
    return str(Path(p).relative_to(ROOT)).replace("\\", "/")


# ------------------------------------------------------------------ git
def git_log() -> pd.DataFrame:
    out = subprocess.run(["git", "log", "--reverse", "--date=format:%Y-%m-%d %H:%M", "--format=%h|%ad|%s"],
                         cwd=ROOT, capture_output=True, text=True, encoding="utf-8").stdout
    rows = [l.split("|", 2) for l in out.strip().splitlines()]
    return pd.DataFrame(rows, columns=["hash", "date", "subject"])


def head_commit() -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()


# ------------------------------------------------------------------ corpus
def corpus() -> dict:
    filings = pd.read_csv(ROOT / "data/metadata/filings.csv")
    companies = pd.read_csv(ROOT / "data/metadata/companies.csv")
    verif = pd.read_csv(ROOT / "data/metadata/verification.csv")
    no_filings = companies[(companies.filter(like="n_10").sum(axis=1)) == 0]
    parse = json.load(open(ROOT / "data/checkpoints/parse_state.json"))
    chunk = json.load(open(ROOT / "data/checkpoints/chunk_state.json"))
    ps, cs = pd.DataFrame(parse.values()), pd.DataFrame(chunk.values())
    raw_bytes = sum(f.stat().st_size for f in (ROOT / "data/raw").rglob("*.htm"))
    return {
        "n_companies_listed": len(companies), "n_filings": len(filings),
        "forms": filings.form_type.value_counts().to_dict(),
        "n_filers_with_filings": int((companies.filter(like="n_10").sum(axis=1) > 0).sum()),
        "foreign": sorted(no_filings.ticker.tolist()),
        "verified_ok": int((verif.status == "ok").sum()), "verified_total": len(verif),
        "filing_dates": (filings.filing_date.min(), filings.filing_date.max()),
        "raw_gb": raw_bytes / 1e9,
        "parser_versions": ps.parser_version.value_counts().to_dict(),
        "section_mode": ps.section_mode.value_counts().to_dict() if "section_mode" in ps else {},
        "n_tables": int(ps.n_table.sum()), "n_footnotes": int(ps.n_footnote.sum()),
        "chunker_versions": cs.chunker_version.value_counts().to_dict(),
        "children": int(cs.n_children.sum()), "parents_text": int(cs.n_parents_text.sum()),
        "parents_table": int(cs.n_parents_table.sum()), "child_tok_max": int(cs.child_embed_tokens_max.max()),
        "child_tok_p50": float(cs.child_embed_tokens_p50.median()),
    }


def eval_sets() -> dict:
    out = {}
    for name in ("retrieval_v0.jsonl", "retrieval_v1.jsonl"):
        qs = [json.loads(l) for l in open(EVAL / name, encoding="utf-8")]
        out[name] = {"n": len(qs), "types": pd.Series([q["type"] for q in qs]).value_counts().to_dict(),
                     "median_gold": float(np.median([len(q["gold_ids"]) for q in qs]))}
    return out


# ------------------------------------------------------------------ retrieval
def experiments() -> pd.DataFrame:
    return pd.read_csv(RES / "experiments.csv")


def per_query(run: str, dataset: str) -> pd.DataFrame:
    folder = RES if dataset == "retrieval_v0.jsonl" else V1_DIR
    return pd.read_csv(folder / f"{run}_per_query.csv").set_index("qid")


def bootstrap(run_a: str, run_b: str, metric: str, dataset: str, subset: str | None = None,
              n: int = 10_000, seed: int = 0) -> dict:
    """Same method as src/evaluation/compare.py (paired, 10k resamples, 95% CI)."""
    a, b = per_query(run_a, dataset), per_query(run_b, dataset)
    j = a[[metric, "type"]].join(b[[metric]], rsuffix="_b", how="inner")
    if subset:
        j = j[j.type == subset]
    diff = (j[f"{metric}_b"] - j[metric]).to_numpy()
    rng = np.random.default_rng(seed)
    samples = diff[rng.integers(0, len(diff), size=(n, len(diff)))].mean(axis=1)
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return {"n": len(j), "a": j[metric].mean(), "b": j[f"{metric}_b"].mean(), "diff": diff.mean(),
            "lo": lo, "hi": hi, "sig": not (lo <= 0 <= hi)}


def sweeps() -> dict:
    return {k: pd.read_csv(RES / f"{k}.csv") for k in ("fusion_sweep", "rerank_sweep", "candidate_pools")}


def scaling() -> pd.DataFrame:
    return pd.read_csv(RES / "scaling.csv")


# ------------------------------------------------------------------ generation
def gen_summary(run: str) -> dict:
    d = pd.read_csv(RES / f"{run}_generation.csv")
    ctx_col = "all_facts_in_context" if "all_facts_in_context" in d else "gold_in_context"
    d = d[d["type"] != "narrative"] if "type" in d else d
    ctx = d[d[ctx_col].astype(bool)]
    by_type = d.groupby("type")["correct"].mean().to_dict() if "type" in d else {}
    return {"n": len(d), "correct": d.correct.astype(bool).mean(), "ctx": d[ctx_col].astype(bool).mean(),
            "correct_given_ctx": ctx.correct.astype(bool).mean(),
            "refused": d.refused.astype(bool).mean(),
            "bad_refusals": int((d.refused.astype(bool) & d[ctx_col].astype(bool)).sum()),
            "unverified": (d.unverified > 0).mean(), "cited": d.cited.astype(bool).mean(),
            "p50_s": d.latency_ms.median() / 1000, "p95_s": d.latency_ms.quantile(0.95) / 1000, "by_type": by_type}


def mcnemar(run_a: str, run_b: str) -> dict:
    a = pd.read_csv(RES / f"{run_a}_generation.csv").set_index("qid")
    b = pd.read_csv(RES / f"{run_b}_generation.csv").set_index("qid")
    j = a[["correct"]].join(b[["correct"]], rsuffix="_b", how="inner").dropna()
    fixed = int((~j.correct.astype(bool) & j.correct_b.astype(bool)).sum())
    broke = int((j.correct.astype(bool) & ~j.correct_b.astype(bool)).sum())
    n, k = fixed + broke, min(fixed, broke)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return {"n": len(j), "fixed": fixed, "broke": broke, "p": p}


def phase17() -> dict:
    """Numerical-hallucination evaluation outputs (src/evaluation/numeric_hallucination.py)."""
    return {k: pd.read_csv(RES / f"phase17_{k}.csv") for k in ("audit", "numbers", "stress", "arithmetic")}


def example_questions() -> pd.DataFrame:
    """80 questions asked through the live API and checked against the filings' iXBRL facts."""
    return pd.read_csv(RES / "example_questions.csv")


def test_count() -> int:
    out = subprocess.run([str(ROOT / ".venv" / "Scripts" / "python"), "-m", "pytest", "--collect-only", "-q"],
                         cwd=ROOT, capture_output=True, text=True).stdout
    return sum(1 for line in out.splitlines() if "::" in line)


def feasibility() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data/benchmarks/llm_feasibility.csv")


def judge_agreement() -> dict:
    a = pd.read_csv(RES / "G5_ragas_groq-gpt-oss-120b.csv").set_index("qid")
    b = pd.read_csv(RES / "G5_ragas_local-llama3.1-8b.csv").set_index("qid")
    j = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
    j = j[~j.refused_a.astype(bool) & ~j.refused_b.astype(bool)].dropna(subset=["faithfulness_a", "faithfulness_b"])
    out = {"n": len(j), "refused": int(a.refused.astype(bool).sum())}
    for m in ("faithfulness", "answer_relevancy"):
        x, y = j[f"{m}_a"], j[f"{m}_b"]
        out[m] = {"a": x.mean(), "b": y.mean(), "mad": (x - y).abs().mean(), "pearson": x.corr(y),
                  "spearman": x.corr(y, method="spearman"), "same_side": ((x >= .5) == (y >= .5)).mean()}
    out["flips"] = j[(j.faithfulness_a >= .5) != (j.faithfulness_b >= .5)].index.tolist()
    return out


def configs() -> dict:
    import yaml
    return {p.stem: yaml.safe_load(open(p, encoding="utf-8")) for p in (ROOT / "configs").glob("*.yaml")}
