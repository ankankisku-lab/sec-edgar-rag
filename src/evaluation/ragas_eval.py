"""Ragas generation metrics with FREE judges only.

Metrics (ragas 0.4 collections, scored one sample at a time):
    faithfulness       share of the answer's claims supported by the retrieved contexts
    answer_relevancy   how directly the answer addresses the question (local bge-small similarity)
Retrieval quality is not re-judged here: it is already measured deterministically
against gold chunks (src.evaluation.retrieval), which is stronger evidence than an
LLM judge and costs no API calls.

Judges (configs/evaluation.yaml): gpt-oss-120b on Groq's free plan (no billing,
~200K tokens/day), or a local open-weight model through Ollama (no quota). Each judge
has its own cache and result files, so verdicts never mix. A local judge is only
trusted after `--agreement` shows it reproduces the strong judge on the same answers.

Every call goes through CallGuard: pacing, hard per-run call cap, waits on per-minute
rate limits (using the provider's "try again in Ns" hint), clean stop on daily limits.
All judge responses are cached on disk, so re-runs and resumes never repeat a call.
Refusals are not judged (no claims to verify; refusal rates come from the generation eval).

Usage:
    python -m src.evaluation.ragas_eval --input G5 --limit 5                         # pilot
    python -m src.evaluation.ragas_eval --input G5 G6 --judge groq-gpt-oss-120b      # resumable
    python -m src.evaluation.ragas_eval --input G5 --judge local-llama3.1-8b --same-as groq-gpt-oss-120b
    python -m src.evaluation.ragas_eval --input G5 --agreement groq-gpt-oss-120b local-llama3.1-8b
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
REFUSAL = "do not contain this information"
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


def build_metrics(cfg: dict, judge_name: str):
    from openai import AsyncOpenAI
    from ragas.cache import DiskCacheBackend
    from ragas.embeddings import HuggingFaceEmbeddings
    from ragas.llms import llm_factory
    from ragas.metrics.collections import AnswerRelevancy, Faithfulness

    judge_cfg = cfg["judges"][judge_name]
    if judge_cfg.get("api_key_env"):
        key = os.environ.get(judge_cfg["api_key_env"])
        if not key:
            raise SystemExit(f"{judge_cfg['api_key_env']} missing from .env")
    else:
        key = "local"  # Ollama ignores the key
    # OpenAI-compatible endpoint (Groq or local Ollama); async, as ragas' collections metrics need.
    # max_retries=0: rate-limit handling lives in CallGuard, where every retry is counted.
    client = AsyncOpenAI(api_key=key, base_url=judge_cfg["base_url"], max_retries=0)
    extra = {"reasoning_effort": judge_cfg["reasoning_effort"]} if judge_cfg.get("reasoning_effort") else {}
    # One cache per judge: a shared cache could return one judge's verdicts to another.
    cache = DiskCacheBackend(str(PROJECT_ROOT / cfg["cache_dir"] / judge_name))
    base = llm_factory(judge_cfg["model"], provider="openai", client=client, adapter="instructor",
                       cache=cache, temperature=0.0, **extra)
    if judge_cfg.get("instructor_mode"):
        # ragas hard-codes Mode.JSON for OpenAI-style endpoints; small local models drift out of
        # free JSON on long prompts (the 8B echoed the schema). JSON_SCHEMA makes Ollama constrain
        # decoding to the schema, so only valid verdicts can be produced.
        import instructor
        base.client = instructor.from_openai(client, mode=getattr(instructor.Mode, judge_cfg["instructor_mode"]))
    guard = CallGuard(base, judge_cfg["requests_per_minute"], judge_cfg["max_calls_per_run"],
                      cfg["rate_limit_max_waits"])
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


def result_path(run_name: str, judge_name: str, pilot: bool = False):
    return RESULTS_DIR / f"{run_name}_ragas_{judge_name}{'_pilot' if pilot else ''}.csv"


async def run(runs: list[str], judge_name: str, limit: int | None, same_as: str | None) -> None:
    cfg = load_config("evaluation")["ragas"]
    judge_cfg = cfg["judges"][judge_name]
    server = None
    if judge_cfg.get("local_ollama"):
        from src.generation import ollama_server
        server = None if ollama_server.is_up() else ollama_server.start("q8_0")
    try:
        judge, metrics = build_metrics(cfg, judge_name)
        for run_name in runs:
            samples = [json.loads(line) for line in open(RESULTS_DIR / f"{run_name}_ragas_input.jsonl", encoding="utf-8")]
            if same_as:  # score exactly the samples another judge scored (agreement check)
                keep = set(pd.read_csv(result_path(run_name, same_as))["qid"])
                samples = [x for x in samples if x["qid"] in keep]
            samples = samples[:limit] if limit else samples
            rows = []
            for i, s in enumerate(samples, 1):
                if REFUSAL in s["response"].lower():
                    # A refusal has no claims to verify: ragas scores it 0, which says nothing
                    # about faithfulness. Refusal rates are reported by the generation eval.
                    rows.append({"qid": s["qid"], "type": s["type"], "refused": True})
                    continue
                calls_before, t0 = judge.calls, time.perf_counter()
                try:
                    scores = await score_sample(metrics, s)
                except BudgetExhausted as e:
                    print(f"stopped at {run_name} sample {i}: {e}. Re-run later; cached calls are not repeated.")
                    break
                except Exception as e:  # noqa: BLE001 -- e.g. unparseable judge output: record and go on
                    rows.append({"qid": s["qid"], "type": s["type"], "refused": False,
                                 "error": f"{type(e).__name__}: {str(e)[:120]}"})
                    print(f"{run_name} {i:3}/{len(samples)} {s['type']:13} JUDGE ERROR {type(e).__name__}", flush=True)
                    pd.DataFrame(rows).to_csv(result_path(run_name, judge_name, pilot=bool(limit)), index=False)
                    continue
                rows.append({"qid": s["qid"], "type": s["type"], "refused": False, **scores,
                             "judge_calls": judge.calls - calls_before, "seconds": round(time.perf_counter() - t0, 1)})
                pd.DataFrame(rows).to_csv(result_path(run_name, judge_name, pilot=bool(limit)), index=False)
                print(f"{run_name} {i:3}/{len(samples)} {s['type']:13} faith={scores['faithfulness']:.2f} "
                      f"relev={scores['answer_relevancy']:.2f} calls={judge.calls - calls_before} "
                      f"({time.perf_counter() - t0:.1f}s)", flush=True)
            df = pd.DataFrame(rows)
            if len(df):
                df.to_csv(result_path(run_name, judge_name, pilot=bool(limit)), index=False)
                judged = df[~df.refused.astype(bool)]
                if "error" in judged:
                    n_err = int(judged["error"].notna().sum())
                    judged = judged[judged["error"].isna()]
                    if n_err:
                        print(f"{run_name}: {n_err} answers could not be judged (see 'error' column)")
                print(f"\n{run_name} [{judge_name}]: {len(judged)} judged, {int(df.refused.sum())} refusals skipped | "
                      f"faithfulness {judged.faithfulness.mean():.3f} | answer_relevancy {judged.answer_relevancy.mean():.3f}"
                      f" | {judged.judge_calls.mean():.1f} calls, {judged.seconds.mean():.0f}s per answer")
                print(judged.groupby("type")[["faithfulness", "answer_relevancy"]].mean().round(3).to_string())
            if judge.stopped:
                break
        print(f"\ntotal judge calls this run: {judge.calls} (cap {judge_cfg['max_calls_per_run']}), "
              f"per-minute rate-limit waits: {judge.waits}")
    finally:
        if server is not None:
            from src.generation import ollama_server
            ollama_server.stop(server)


def agreement(run_name: str, judge_a: str, judge_b: str) -> None:
    """How well does judge B reproduce judge A's verdicts on the same answers?"""
    a = pd.read_csv(result_path(run_name, judge_a)).set_index("qid")
    b = pd.read_csv(result_path(run_name, judge_b)).set_index("qid")
    j = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
    j = j[~j.refused_a.astype(bool) & ~j.refused_b.astype(bool)].dropna(subset=["faithfulness_a", "faithfulness_b"])
    print(f"{run_name}: {len(j)} answers judged by both {judge_a} (A) and {judge_b} (B)")
    for m in ("faithfulness", "answer_relevancy"):
        x, y = j[f"{m}_a"], j[f"{m}_b"]
        same_side = ((x >= 0.5) == (y >= 0.5)).mean()
        print(f"  {m:17} mean A {x.mean():.3f}  B {y.mean():.3f} | mean |A-B| {(x - y).abs().mean():.3f} | "
              f"Pearson {x.corr(y):.2f}  Spearman {x.corr(y, method='spearman'):.2f} | "
              f"same side of 0.5: {same_side:.0%}")
    flips = j[(j.faithfulness_a >= 0.5) != (j.faithfulness_b >= 0.5)]
    for qid, r in flips.iterrows():
        print(f"    disagree {qid} ({r.type_a}): A {r.faithfulness_a:.2f} vs B {r.faithfulness_b:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", nargs="+", required=True, help="generation run names, e.g. G5 G6")
    parser.add_argument("--judge", help="judge name from configs/evaluation.yaml")
    parser.add_argument("--limit", type=int, help="score only the first N samples per run (pilot)")
    parser.add_argument("--same-as", help="score only the samples this other judge scored")
    parser.add_argument("--agreement", nargs=2, metavar=("JUDGE_A", "JUDGE_B"), help="compare two judges' results")
    args = parser.parse_args()
    if args.agreement:
        for run_name in args.input:
            agreement(run_name, *args.agreement)
        return
    judge = args.judge or load_config("evaluation")["ragas"]["default_judge"]
    asyncio.run(run(args.input, judge, args.limit, args.same_as))


if __name__ == "__main__":
    main()
