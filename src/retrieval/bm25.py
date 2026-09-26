"""BM25 as Qdrant sparse vectors.

Documents: fastembed's Qdrant/bm25 turns each child's embedding text (metadata
header + chunk) into {token_id: BM25 term-frequency weight}, with length
normalisation against our corpus's real average length (configs/retrieval.yaml).
IDF is applied by Qdrant at query time (collection sparse vector uses the IDF
modifier), so it always reflects the whole collection, including filings added
later. Queries: each unique stemmed token with weight 1.

These functions are passed to LlamaIndex's QdrantVectorStore as
sparse_doc_fn / sparse_query_fn, so indexing and querying go through LlamaIndex.

Usage (measure avg_len for the config):
    python -m src.retrieval.bm25 --avg-len --tickers AAPL MSFT NVDA
"""
from __future__ import annotations

import argparse
import os
from functools import lru_cache

from src.config import load_config


@lru_cache(maxsize=1)
def get_bm25():
    from fastembed import SparseTextEmbedding
    cfg = load_config("retrieval")["bm25"]
    return SparseTextEmbedding(cfg["model"], cache_dir=os.environ["FASTEMBED_CACHE_PATH"],
                               k=cfg["k"], b=cfg["b"], avg_len=cfg["avg_len"])


def sparse_doc_fn(texts: list[str]) -> tuple[list[list[int]], list[list[float]]]:
    embs = list(get_bm25().embed(texts, batch_size=256))
    return [e.indices.tolist() for e in embs], [e.values.tolist() for e in embs]


def sparse_query_fn(texts: list[str]) -> tuple[list[list[int]], list[list[float]]]:
    embs = list(get_bm25().query_embed(texts))
    return [e.indices.tolist() for e in embs], [e.values.tolist() for e in embs]


def bm25_length(text: str) -> int:
    """Document length exactly as fastembed's BM25 counts it (stemmed, no stopwords)."""
    from fastembed.sparse.utils.tokenizer import SimpleTokenizer  # noqa: F401  (import check)
    from fastembed.sparse.bm25 import remove_non_alphanumeric
    model = get_bm25().model
    return len(model._stem(model.tokenizer.tokenize(remove_non_alphanumeric(text))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--avg-len", action="store_true")
    parser.add_argument("--tickers", nargs="+")
    args = parser.parse_args()
    if args.avg_len:
        from llama_index.core.schema import MetadataMode
        from src.chunking.hierarchical import CHUNKS_DIR, load_nodes
        files = sorted(CHUNKS_DIR.glob("*/*.jsonl"))
        if args.tickers:
            files = [f for f in files if f.parent.name in {t.upper() for t in args.tickers}]
        lengths = [bm25_length(n.get_content(MetadataMode.EMBED)) for f in files for n in load_nodes(f)
                   if n.metadata["node_type"] == "child"]
        print(f"{len(lengths):,} chunks: mean BM25 length {sum(lengths) / len(lengths):.1f} tokens, "
              f"min {min(lengths)}, max {max(lengths)}")


if __name__ == "__main__":
    main()
