"""Can a 3-4B local LLM serve this RAG system on a 4 GB GPU?

For every candidate x context length x KV-cache type, send one real RAG prompt
(Apple FY2025 Q3 10-Q parents, income statement first) and record:
VRAM used / layers spilled to CPU, runner RAM, load time, prompt and generation
speed, and whether the answer states the right numbers. This is a feasibility
and speed benchmark only; answer quality is benchmarked later on the eval set.

Usage:
    python -m src.generation.llm_feasibility --pull      # download candidates first
    python -m src.generation.llm_feasibility
    python -m src.generation.llm_feasibility --coresident qwen3-4b --context 8192
"""
from __future__ import annotations

import argparse
import re
import subprocess
import time
from pathlib import Path

import pandas as pd
import psutil
import requests

from src.chunking.hierarchical import CHUNKS_DIR, load_nodes, n_tokens
from src.config import DATA_DIR, load_config
from src.generation import ollama_server

RESULTS_DIR = DATA_DIR / "benchmarks"
FILING = "AAPL_10Q_2025-06-28"
QUESTION = ("What was Apple's operating income for the three months ended June 28, 2025, "
            "and how did it compare with the same quarter a year earlier?")
EXPECTED = {"2025": ("28,202", "28.2"), "2024": ("25,352", "25.4")}
SYSTEM = ("You are a financial analyst. Answer only from the provided SEC filing excerpts. "
          "Quote exact figures with their units and periods. If the excerpts do not contain "
          "the answer, say so.")


def build_prompt(target_tokens: int) -> str:
    nodes = [n for n in load_nodes(CHUNKS_DIR / "AAPL" / f"{FILING}.jsonl") if n.metadata["node_type"] == "parent"]
    # Income statement first (it holds the answer), then the rest in document order as filler.
    nodes.sort(key=lambda n: "STATEMENTS OF OPERATIONS" not in n.metadata.get("table_caption", ""))
    parts, used = [], n_tokens(QUESTION) + 80
    for i, n in enumerate(nodes, 1):
        block = f"[SOURCE {i}] {n.metadata['heading_path']}\n{n.text}"
        t = n_tokens(block)
        if used + t > target_tokens:
            continue
        parts.append(block)
        used += t
    return "Context:\n\n" + "\n\n".join(parts) + f"\n\nQuestion: {QUESTION}"


def vram_used_mib() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return int(out.strip().splitlines()[0])


def runner_ram_mib() -> int:
    return sum(p.info["memory_info"].rss for p in psutil.process_iter(["name", "memory_info"])
               if (p.info["name"] or "").lower().startswith(("ollama", "llama"))) // 2**20


NUMBER_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")  # "28,202", "11.2"; no trailing "." or ","
CORRECT_DERIVED = {"2,850", "2.85", "11.2", "11.24", "11.3"}          # 28,202 - 25,352 and its % change


def answer_check(answer: str, prompt: str) -> dict:
    found = {year: any(v in answer for v in values) for year, values in EXPECTED.items()}
    # Numbers the model wrote that are neither in the context nor a correct derivation:
    # a cheap hallucination signal (misread values, wrong arithmetic).
    context_numbers = set(NUMBER_RE.findall(prompt))
    numbers = {n for n in NUMBER_RE.findall(answer)
               if len(n.replace(",", "").split(".")[0]) >= 3 or "." in n}   # skip days like "28"
    unsupported = sorted(n for n in numbers - context_numbers - CORRECT_DERIVED if not re.fullmatch(r"(19|20)\d\d", n))
    return {"has_2025_value": found["2025"], "has_2024_value": found["2024"],
            "computed_change_correctly": bool(numbers & CORRECT_DERIVED),
            "unsupported_numbers": " ".join(unsupported)}


def run_once(tag: str, num_ctx: int, prompt: str, max_tokens: int) -> dict:
    url = ollama_server.base_url()
    before = vram_used_mib()
    t0 = time.perf_counter()
    r = requests.post(url + "/api/chat", timeout=900, json={
        "model": tag, "stream": False, "keep_alive": "5m",
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        "options": {"num_ctx": num_ctx, "temperature": 0, "seed": 0, "num_predict": max_tokens},
    })
    r.raise_for_status()
    wall = time.perf_counter() - t0
    body = r.json()
    ps = next(m for m in requests.get(url + "/api/ps").json()["models"] if m["name"] == tag or m["model"] == tag)
    row = {
        "vram_used_mib": vram_used_mib(), "vram_before_mib": before,
        "model_total_mib": ps["size"] // 2**20, "model_on_gpu_mib": ps["size_vram"] // 2**20,
        "fully_on_gpu": ps["size_vram"] >= ps["size"],
        "runner_ram_mib": runner_ram_mib(),
        "load_s": round(body.get("load_duration", 0) / 1e9, 1),
        "prompt_tokens": body.get("prompt_eval_count", 0),
        "prompt_tok_s": round(body.get("prompt_eval_count", 0) / max(body.get("prompt_eval_duration", 1) / 1e9, 1e-9)),
        "gen_tokens": body.get("eval_count", 0),
        "gen_tok_s": round(body.get("eval_count", 0) / max(body.get("eval_duration", 1) / 1e9, 1e-9), 1),
        "wall_s": round(wall, 1),
        "answer": body["message"]["content"].strip(),
    }
    row.update(answer_check(row["answer"], prompt))
    requests.post(url + "/api/generate", json={"model": tag, "keep_alive": 0})  # unload for a clean next run
    time.sleep(2)
    return row


def pull(tags: list[str]) -> None:
    proc = ollama_server.start()
    try:
        for tag in tags:
            print(f"pulling {tag} ...", flush=True)
            r = requests.post(ollama_server.base_url() + "/api/pull", json={"model": tag, "stream": False}, timeout=3600)
            print(f"  {r.status_code} {r.json()}")
    finally:
        ollama_server.stop(proc)


def feasibility(cfg: dict) -> pd.DataFrame:
    f = cfg["feasibility"]
    prompts = {ctx: build_prompt(int(ctx * f["prompt_fill"])) for ctx in f["contexts"]}
    rows = []
    for kv in f["kv_cache_types"]:
        proc = ollama_server.start(kv)
        try:
            for cand in cfg["candidates"]:
                for ctx in f["contexts"]:
                    print(f"{cand['name']:12} ctx={ctx:5} kv={kv:5} ...", end=" ", flush=True)
                    try:
                        row = run_once(cand["tag"], ctx, prompts[ctx], f["max_output_tokens"])
                    except Exception as e:  # e.g. out of memory: record it, keep going
                        row = {"error": f"{type(e).__name__}: {e}"[:200]}
                    rows.append({"model": cand["name"], "context": ctx, "kv_cache": kv, **row})
                    print({k: row.get(k) for k in ("fully_on_gpu", "model_on_gpu_mib", "vram_used_mib",
                                                    "prompt_tok_s", "gen_tok_s", "has_2025_value", "error") if k in row})
        finally:
            ollama_server.stop(proc)
    return pd.DataFrame(rows)


def coresident(cfg: dict, name: str, num_ctx: int) -> None:
    """Embedder + reranker loaded on the GPU first (as in the live API), then the LLM."""
    import torch
    from sentence_transformers import CrossEncoder

    from src.embeddings.embedder import embed_query

    tag = next(c["tag"] for c in cfg["candidates"] if c["name"] == name)
    embed_query("warm up")
    reranker = CrossEncoder("BAAI/bge-reranker-base", device="cuda", model_kwargs={"torch_dtype": torch.float16})
    prompt = build_prompt(int(num_ctx * cfg["feasibility"]["prompt_fill"]))
    passages = [p for p in prompt.split("[SOURCE")[1:40]]
    reranker.predict([(QUESTION, p[:2000]) for p in passages], batch_size=16)
    torch_mib = torch.cuda.max_memory_reserved() // 2**20
    print(f"embedder + reranker: {torch_mib} MiB reserved by torch; GPU used {vram_used_mib()} MiB")

    proc = ollama_server.start("q8_0")
    try:
        row = run_once(tag, num_ctx, prompt, cfg["feasibility"]["max_output_tokens"])
        t0 = time.perf_counter()
        reranker.predict([(QUESTION, p[:2000]) for p in passages], batch_size=16)
        print(f"reranker still works after LLM load: {1000 * (time.perf_counter() - t0):.0f} ms for {len(passages)} pairs")
    finally:
        ollama_server.stop(proc)
    print({k: row[k] for k in ("fully_on_gpu", "model_on_gpu_mib", "model_total_mib", "vram_used_mib",
                               "prompt_tok_s", "gen_tok_s", "wall_s", "has_2025_value", "has_2024_value")})
    print("answer:", row["answer"][:600])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pull", action="store_true")
    parser.add_argument("--coresident", metavar="MODEL_NAME")
    parser.add_argument("--context", type=int, default=8192)
    args = parser.parse_args()
    cfg = load_config("llm")
    if args.pull:
        pull([c["tag"] for c in cfg["candidates"]])
    elif args.coresident:
        coresident(cfg, args.coresident, args.context)
    else:
        df = feasibility(cfg)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out = RESULTS_DIR / "llm_feasibility.csv"
        df.to_csv(out, index=False)
        print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
