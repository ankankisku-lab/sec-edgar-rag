"""Dense embeddings for child chunks, cached on disk per model.

    data/chunks/<T>/<filing>.jsonl -> data/embeddings/<model>/<T>/<filing>.npz
                                      (ids: child node ids, vectors: float32 [n, dim])

The text embedded is exactly what LlamaIndex would embed (MetadataMode.EMBED):
the metadata header (company, form, fiscal period, heading path, caption,
units) followed by the chunk. Vectors are cached so indexing (Qdrant), model
comparisons and the quantization experiment never recompute them.

Usage:
    python -m src.embeddings.embedder --tickers AAPL MSFT NVDA
    python -m src.embeddings.embedder --tickers AAPL MSFT NVDA --smoke-test
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.config import CHECKPOINT_DIR, DATA_DIR, LOG_DIR, load_config  # sets HF_HOME before HF imports
from src.chunking.hierarchical import CHUNKS_DIR, load_nodes
from src.ingestion.checkpoint import Checkpoint

log = logging.getLogger(__name__)

EMBEDDINGS_DIR = DATA_DIR / "embeddings"


def model_slug(model_name: str) -> str:
    return model_name.split("/")[-1]


@lru_cache(maxsize=2)
def get_embed_model(model_name: str | None = None):
    """LlamaIndex HuggingFaceEmbedding on GPU (fp16 if configured)."""
    import torch
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    cfg = load_config("embedding")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        log.warning("CUDA not available: embedding on CPU will be slow")
    model_kwargs = {"torch_dtype": torch.float16} if cfg["fp16"] and device == "cuda" else {}
    return HuggingFaceEmbedding(
        model_name=model_name or cfg["model_name"],
        max_length=cfg["max_length"],
        normalize=cfg["normalize"],
        embed_batch_size=cfg["batch_size"],
        query_instruction=cfg["query_instruction"],
        text_instruction="",
        device=device,
        model_kwargs=model_kwargs,
    )


def embed_texts(texts: list[str]) -> np.ndarray:
    vectors = get_embed_model().get_text_embedding_batch(texts)
    return np.asarray(vectors, dtype=np.float32)


def embed_query(query: str) -> np.ndarray:
    return np.asarray(get_embed_model().get_query_embedding(query), dtype=np.float32)


def child_nodes(path: Path):
    return [n for n in load_nodes(path) if n.metadata["node_type"] == "child"]


def output_path(model_name: str, ticker: str, filing_id: str) -> Path:
    return EMBEDDINGS_DIR / model_slug(model_name) / ticker / f"{filing_id}.npz"


def load_embeddings(model_name: str, ticker: str, filing_id: str) -> tuple[list[str], np.ndarray]:
    data = np.load(output_path(model_name, ticker, filing_id))
    return list(data["ids"]), data["vectors"]


# --------------------------------------------------------------------------- smoke test

SMOKE_QUERIES = [
    "What was Apple's operating income in the third quarter of fiscal 2025?",
    "Why did Microsoft's gross margin change?",
    "NVIDIA data center revenue",
    "What are the main risks from export controls on NVIDIA's products?",
    "How much did Apple spend on share repurchases?",
]


def smoke_test(model_name: str, files: list[Path], k: int = 3) -> None:
    """Brute-force cosine search over the embedded chunks: a preview of the
    Phase 7 baseline, run before any vector database exists."""
    texts, ids, vecs = {}, [], []
    for f in files:
        for n in child_nodes(f):
            texts[n.node_id] = n
        fid_ids, v = load_embeddings(model_name, f.parent.name, f.stem)
        ids += fid_ids
        vecs.append(v)
    matrix = np.vstack(vecs)
    for q in SMOKE_QUERIES:
        scores = matrix @ embed_query(q)
        print(f"\nQ: {q}")
        for i in np.argsort(-scores)[:k]:
            n = texts[ids[i]]
            m = n.metadata
            snippet = " ".join(n.text.split())[:110]
            print(f"  {scores[i]:.3f}  {m['ticker']} {m['form_type']} {m['period']} | {m['heading_path'][-60:]}\n"
                  f"         {snippet}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help="run example queries after embedding")
    args = parser.parse_args()

    LOG_DIR.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.FileHandler(LOG_DIR / "embed.log", encoding="utf-8")])
    import torch
    from llama_index.core.schema import MetadataMode

    cfg = load_config("embedding")
    model_name = cfg["model_name"]
    files = sorted(CHUNKS_DIR.glob("*/*.jsonl"))
    if args.tickers:
        wanted = {t.upper() for t in args.tickers}
        files = [f for f in files if f.parent.name in wanted]
    if args.limit:
        files = files[:args.limit]

    ckpt = Checkpoint(CHECKPOINT_DIR / f"embed_state_{model_slug(model_name)}.json")
    tokenizer = get_embed_model()._model.tokenizer
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    n_chunks = n_truncated = 0
    t0 = time.perf_counter()
    for f in files:
        fid = f.stem
        out = output_path(model_name, f.parent.name, fid)
        if (not args.force and ckpt.is_done(fid) and out.exists()
                and out.stat().st_mtime >= f.stat().st_mtime):
            continue
        nodes = child_nodes(f)
        texts = [n.get_content(MetadataMode.EMBED) for n in nodes]
        lengths = [len(ids) for ids in tokenizer(texts, add_special_tokens=True)["input_ids"]]
        truncated = sum(l > cfg["max_length"] for l in lengths)
        start = time.perf_counter()
        vectors = embed_texts(texts)
        seconds = time.perf_counter() - start
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, ids=np.array([n.node_id for n in nodes]), vectors=vectors)
        ckpt.mark(fid, "done", error=None, n=len(nodes), dim=int(vectors.shape[1]), truncated=truncated,
                  max_tokens=max(lengths, default=0), seconds=round(seconds, 2))
        log.info("%s: %d chunks in %.1fs (%d truncated)", fid, len(nodes), seconds, truncated)
        print(f"{fid}: {len(nodes):5d} chunks  {len(nodes) / seconds:6.0f}/s  max {max(lengths)} tokens"
              f"{f'  ({truncated} truncated)' if truncated else ''}")
        n_chunks += len(nodes)
        n_truncated += truncated

    elapsed = time.perf_counter() - t0
    if n_chunks:
        peak = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0
        print(f"\nembedded {n_chunks:,} chunks in {elapsed:.1f}s ({n_chunks / elapsed:.0f}/s), "
              f"truncated {n_truncated}, peak VRAM {peak:.0f} MiB, model {model_name}")
    else:
        print("nothing to embed (all up to date)")
    if args.smoke_test:
        smoke_test(model_name, files)


if __name__ == "__main__":
    main()
