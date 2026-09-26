"""Load child nodes + cached embeddings into Qdrant, parents into the parent store.

Per filing: delete its existing points (by doc_id = accession number), upsert
the children with their dense vectors via LlamaIndex's QdrantVectorStore,
replace its parents in SQLite, checkpoint. Re-running never duplicates.

Usage:
    docker compose up -d
    python -m src.retrieval.qdrant_index --tickers AAPL MSFT NVDA
    python -m src.retrieval.qdrant_index --recreate     # drop and rebuild the collection
"""
from __future__ import annotations

import argparse
import logging
from functools import lru_cache

from src.config import CHECKPOINT_DIR, LOG_DIR, load_config
from src.chunking.hierarchical import CHUNKS_DIR, load_nodes
from src.embeddings.embedder import load_embeddings, output_path as embedding_path
from src.ingestion.checkpoint import Checkpoint
from src.retrieval.parent_store import ParentStore

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_client():
    from qdrant_client import QdrantClient
    return QdrantClient(url=load_config("qdrant")["url"], timeout=60)


def ensure_collection(recreate: bool = False) -> None:
    """Named dense + (reserved) sparse vectors, on-disk payload, payload indexes."""
    from qdrant_client import models

    cfg, emb = load_config("qdrant"), load_config("embedding")
    client, name = get_client(), cfg["collection"]
    if recreate and client.collection_exists(name):
        client.delete_collection(name)
    if client.collection_exists(name):
        return
    client.create_collection(
        collection_name=name,
        vectors_config={cfg["dense_vector_name"]: models.VectorParams(size=emb["dim"], distance=models.Distance.COSINE)},
        # BM25 needs IDF computed over the whole collection; Qdrant does that with the IDF modifier.
        sparse_vectors_config={cfg["sparse_vector_name"]: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        on_disk_payload=True,
    )
    for field in cfg["keyword_indexes"]:
        client.create_payload_index(name, field, models.PayloadSchemaType.KEYWORD)
    for field in cfg["datetime_indexes"]:
        client.create_payload_index(name, field, models.PayloadSchemaType.DATETIME)
    log.info("created collection %s", name)


def get_vector_store(fusion_fn=None):
    """LlamaIndex QdrantVectorStore with dense + BM25 sparse vectors.

    `fusion_fn` overrides how hybrid queries merge the dense and sparse result
    lists (LlamaIndex default: relative score fusion)."""
    from llama_index.vector_stores.qdrant import QdrantVectorStore
    from src.retrieval.bm25 import sparse_doc_fn, sparse_query_fn
    cfg = load_config("qdrant")
    kwargs = {"hybrid_fusion_fn": fusion_fn} if fusion_fn else {}
    return QdrantVectorStore(
        collection_name=cfg["collection"], client=get_client(),
        dense_vector_name=cfg["dense_vector_name"], sparse_vector_name=cfg["sparse_vector_name"],
        batch_size=cfg["upsert_batch_size"],
        enable_hybrid=True, sparse_doc_fn=sparse_doc_fn, sparse_query_fn=sparse_query_fn,
        **kwargs,
    )


def index_filing(chunk_file, vector_store, parents: ParentStore, model_name: str) -> int:
    nodes = load_nodes(chunk_file)
    children = [n for n in nodes if n.metadata["node_type"] == "child"]
    ids, vectors = load_embeddings(model_name, chunk_file.parent.name, chunk_file.stem)
    if ids != [c.node_id for c in children]:
        raise ValueError("embeddings are stale for this chunk file; re-run the embedder")
    for child, vector in zip(children, vectors):
        child.embedding = vector.tolist()

    accession = children[0].ref_doc_id
    vector_store.delete(ref_doc_id=accession)  # idempotent re-index
    vector_store.add(children)
    parents.replace_filing(chunk_file.stem, [n for n in nodes if n.metadata["node_type"] == "parent"])
    return len(children)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", nargs="+")
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    LOG_DIR.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.FileHandler(LOG_DIR / "index.log", encoding="utf-8")])
    cfg, model_name = load_config("qdrant"), load_config("embedding")["model_name"]
    ensure_collection(recreate=args.recreate)
    ckpt_path = CHECKPOINT_DIR / f"index_state_{cfg['collection']}.json"
    if args.recreate and ckpt_path.exists():
        ckpt_path.unlink()
    ckpt = Checkpoint(ckpt_path)
    vector_store, parents = get_vector_store(), ParentStore()

    files = sorted(CHUNKS_DIR.glob("*/*.jsonl"))
    if args.tickers:
        wanted = {t.upper() for t in args.tickers}
        files = [f for f in files if f.parent.name in wanted]
    done = skipped = missing = 0
    for f in files:
        emb = embedding_path(model_name, f.parent.name, f.stem)
        if not emb.exists():
            missing += 1
            continue
        if not args.force and ckpt.is_done(f.stem) and ckpt.get(f.stem).get("embedding_mtime") == emb.stat().st_mtime:
            skipped += 1
            continue
        n = index_filing(f, vector_store, parents, model_name)
        ckpt.mark(f.stem, "done", error=None, n_children=n, embedding_mtime=emb.stat().st_mtime)
        done += 1

    info = get_client().get_collection(cfg["collection"])
    print(f"indexed {done}, skipped {skipped}, missing embeddings {missing} | "
          f"collection '{cfg['collection']}': {info.points_count:,} points, status {info.status} | "
          f"parents: {parents.count():,}")


if __name__ == "__main__":
    main()
