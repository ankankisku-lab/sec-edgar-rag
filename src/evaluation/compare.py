"""Is run B really better than run A? Paired bootstrap over questions.

Resamples the question set 10,000 times (same questions for both runs) and
reports the mean difference with a 95% confidence interval. If the interval
contains 0, the difference is not distinguishable from noise on this eval set.

Usage:
    python -m src.evaluation.compare V2_bm25 V4_rerank
    python -m src.evaluation.compare V4_rerank V4h_rerank --metric precision@3
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.evaluation.retrieval import RESULTS_DIR


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int = 10_000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    diff = b - a
    idx = rng.integers(0, len(diff), size=(n, len(diff)))
    samples = diff[idx].mean(axis=1)
    return float(diff.mean()), float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def compare(run_a: str, run_b: str, metric: str) -> dict:
    a = pd.read_csv(RESULTS_DIR / f"{run_a}_per_query.csv").set_index("qid")
    b = pd.read_csv(RESULTS_DIR / f"{run_b}_per_query.csv").set_index("qid")
    joined = a[[metric, "type"]].join(b[[metric]], rsuffix="_b", how="inner")
    out = {}
    for name, g in [("all", joined), *joined.groupby("type")]:
        mean, lo, hi = paired_bootstrap(g[metric].to_numpy(), g[f"{metric}_b"].to_numpy())
        out[name] = {"n": len(g), "A": round(g[metric].mean(), 3), "B": round(g[f"{metric}_b"].mean(), 3),
                     "diff": round(mean, 3), "ci95": (round(lo, 3), round(hi, 3)),
                     "significant": not (lo <= 0 <= hi)}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_a")
    parser.add_argument("run_b")
    parser.add_argument("--metric", default="mrr@10")
    args = parser.parse_args()
    for name, r in compare(args.run_a, args.run_b, args.metric).items():
        print(f"{args.metric:12} {name:10} n={r['n']:3}  A={r['A']:.3f}  B={r['B']:.3f}  "
              f"diff={r['diff']:+.3f}  95% CI [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]  "
              f"{'significant' if r['significant'] else 'not significant'}")


if __name__ == "__main__":
    main()
