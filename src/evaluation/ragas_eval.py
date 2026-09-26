"""Ragas generation metrics with a free-tier judge (Groq, gpt-oss-120b).

Metrics (ragas 0.4 collections, scored one sample at a time):
    faithfulness       share of the answer's claims supported by the retrieved contexts
    answer_relevancy   how directly the answer addresses the question (local bge-small similarity)
Retrieval quality is not re-judged here: it is already measured deterministically
against gold chunks (src.evaluation.retrieval), which is stronger evidence than an
LLM judge and costs no API calls.

Free-tier guards (configs/evaluation.yaml): the judge key is on a free plan with no
billing attached. Every call goes through CallGuard, which paces requests, stops at a
hard per-run call cap, waits out per-minute rate limits (using the provider's
"try again in Ns" hint) and stops cleanly on daily limits. All judge responses are
cached on disk, so re-running or resuming after a stop never repeats a call.

Usage:
    python -m src.evaluation.ragas_eval --input G6 --limit 5      # pilot: shows calls/sample
    python -m src.evaluation.ragas_eval --input G5 G6             # full, resumable
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time

import pandas as pd

from src.config import PROJECT_ROOT, load_config
from src.evaluation.dataset import EVAL_DIR

RESULTS_DIR = EVAL_DIR / "results"
PER_DAY = re.compile(r"per day|\((?:rpd|tpd)\)|daily|quota", re.I)
RETRY_IN = re.compile(r"try again in\s+(?:(\d+)m)?([\d.]+)s", re.I)


class BudgetExhausted(RuntimeError):
    pass


class CallGuard:
    """Installs free-tier guards on a ragas InstructorLLM *in place* (ragas type-checks the
    judge, so it can't be wrapped): pacing, hard call cap, per-minute waits, daily stop.

    Counts every generate call, including ones later served from the disk cache and
    retries after a per-minute limit: the cap is therefore conservative."""

    def __init__(self, llm, rpm: int, max_calls: int, max_waits: int):
        self.min_interval, self.max_calls, self.max_waits = 60.0 / rpm, max_calls, max_waits
        self.calls = self.waits = 0
        self._last = 0.0
        self.stopped: str | None = None
        async_generate = llm.agenerate

        async def agenerate(*a, **kw):
            for _ in range(self.max_waits + 1):
                self._before()
                try:
                    return await async_generate(*a, **kw)
                except Exception as e:  # noqa: BLE001
                    await self._handle(e)
            self.stopped = f"still rate-limited after {self.max_waits} waits"
            raise BudgetExhausted(self.stopped)

        object.__setattr__(llm, "agenerate", agenerate)

    def _before(self):
        if self.stopped:
            raise BudgetExhausted(self.stopped)
        if self.calls >= self.max_calls:
            self.stopped = f"call cap reached ({self.max_calls})"
            raise BudgetExhausted(self.stopped)
        wait = self.min_interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()
        self.calls += 1

    async def _handle(self, e: Exception):
        text = str(e)
        if "429" not in text and "rate limit" not in text.lower():
            raise e
        if PER_DAY.search(text):
            self.stopped = f"free-tier daily limit reached: {text[:200]}"
            raise BudgetExhausted(self.stopped) from e
        m = RETRY_IN.search(text)
        delay = (int(m.group(1) or 0) * 60 + float(m.group(2)) + 1.0) if m else 20.0
        self.waits += 1
        await asyncio.sleep(min(delay, 120.0))


def build_metrics(cfg: dict):
    from openai import AsyncOpenAI
    from ragas.cache import DiskCacheBackend
    from ragas.embeddings import HuggingFaceEmbeddings
    from ragas.llms import llm_factory
    from ragas.metrics.collections import AnswerRelevancy, Faithfulness

    judge_cfg = cfg["judge"]
    key = os.environ.get(judge_cfg["api_key_env"])
    if not key:
        raise SystemExit(f"{judge_cfg['api_key_env']} missing from .env")
    # OpenAI-compatible endpoint (Groq); async, as ragas' collections metrics require.
    # max_retries=0: rate-limit handling lives in CallGuard, where every retry is counted.
    client = AsyncOpenAI(api_key=key, base_url=judge_cfg["base_url"], max_retries=0)
    extra = {"reasoning_effort": judge_cfg["reasoning_effort"]} if judge_cfg.get("reasoning_effort") else {}
    base = llm_factory(judge_cfg["model"], provider="openai", client=client, adapter="instructor",
                       cache=DiskCacheBackend(str(PROJECT_ROOT / cfg["cache_dir"])), temperature=0.0, **extra)
    guard = CallGuard(base, cfg["requests_per_minute"], cfg["max_calls_per_run"], cfg["rate_limit_max_waits"])
    # CPU: a handful of short texts per sample; keeps the GPU free for the local LLM stack.
    embeddings = HuggingFaceEmbeddings(model=cfg["embeddings_model"], device="cpu")
    return guard, {
        "faithfulness": Faithfulness(llm=base),
        "answer_relevancy": AnswerRelevancy(llm=base, embeddings=embeddings,
                                            strictness=cfg["answer_relevancy_strictness"]),
    }


async def score_sample(metrics: dict, s: dict) -> dict:
    """Awaited inside ONE event loop for the whole run: the async HTTP client keeps its
    connections bound to the loop it first ran in (a fresh asyncio.run per call breaks it)."""
    out = {}
    for name, metric in metrics.items():
        if name == "faithfulness":
            r = await metric.ascore(user_input=s["user_input"], response=s["response"],
                                    retrieved_contexts=s["retrieved_contexts"])
        else:
            r = await metric.ascore(user_input=s["user_input"], response=s["response"])
        out[name] = float(r.value) if r.value is not None else float("nan")
    return out


async def run(runs: list[str], limit: int | None) -> None:
    cfg = load_config("evaluation")["ragas"]
    judge, metrics = build_metrics(cfg)
    for run_name in runs:
        samples = [json.loads(line) for line in open(RESULTS_DIR / f"{run_name}_ragas_input.jsonl", encoding="utf-8")]
        samples = samples[:limit] if limit else samples
        rows = []
        for i, s in enumerate(samples, 1):
            calls_before, t0 = judge.calls, time.perf_counter()
            try:
                scores = await score_sample(metrics, s)
            except BudgetExhausted as e:
                print(f"stopped at {run_name} sample {i}: {e}. Re-run later; cached calls are not repeated.")
                break
            rows.append({"qid": s["qid"], "type": s["type"], **scores,
                         "judge_calls": judge.calls - calls_before, "seconds": round(time.perf_counter() - t0, 1)})
            print(f"{run_name} {i:3}/{len(samples)} {s['type']:13} faith={scores['faithfulness']:.2f} "
                  f"relev={scores['answer_relevancy']:.2f} calls={judge.calls - calls_before} "
                  f"({time.perf_counter() - t0:.1f}s)", flush=True)
        df = pd.DataFrame(rows)
        if len(df):
            suffix = "_pilot" if limit else ""
            df.to_csv(RESULTS_DIR / f"{run_name}_ragas{suffix}.csv", index=False)
            print(f"\n{run_name}: {len(df)}/{len(samples)} scored | faithfulness {df.faithfulness.mean():.3f} | "
                  f"answer_relevancy {df.answer_relevancy.mean():.3f} | judge calls {df.judge_calls.sum()} "
                  f"({df.judge_calls.mean():.1f}/sample, {df.seconds.mean():.0f}s/sample)")
            print(df.groupby("type")[["faithfulness", "answer_relevancy"]].mean().round(3).to_string())
        if judge.stopped:
            break
    print(f"\ntotal judge API calls this run: {judge.calls} (cap {cfg['max_calls_per_run']}), "
          f"per-minute rate-limit waits: {judge.waits}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", nargs="+", required=True, help="generation run names, e.g. G5 G6")
    parser.add_argument("--limit", type=int, help="score only the first N samples per run (pilot)")
    args = parser.parse_args()
    asyncio.run(run(args.input, args.limit))


if __name__ == "__main__":
    main()
