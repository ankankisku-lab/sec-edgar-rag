from llama_index.core.schema import TextNode

from src.retrieval.parent_store import ParentStore


def test_parent_store_replace_is_idempotent(tmp_path):
    store = ParentStore(tmp_path / "p.sqlite")
    nodes = [TextNode(id_=f"p{i}", text=f"parent {i}") for i in range(3)]
    store.replace_filing("F1", nodes)
    store.replace_filing("F1", nodes[:2])          # re-index with fewer parents
    store.replace_filing("F2", [TextNode(id_="q0", text="other")])
    assert store.count() == 3
    got = store.get(["p0", "p2", "q0", "missing"])
    assert set(got) == {"p0", "q0"} and got["p0"].text == "parent 0"


def test_retrieval_metrics():
    from src.evaluation.retrieval import score
    s = score(["x", "g1", "y", "g2", "z"], {"g1", "g2", "g3"})
    assert s["precision@3"] == 1 / 3
    assert s["recall@5"] == 2 / 3            # min(|gold|=3, 5) = 3
    assert s["hit@1"] == 0 and s["hit@3"] == 1
    assert s["mrr@10"] == 0.5 and s["first_rank"] == 2
    miss = score(["a", "b"], {"g"})
    assert miss["mrr@10"] == 0 and miss["ndcg@10"] == 0 and miss["first_rank"] is None
    many = score(["g1", "g2", "g3", "g4", "g5"], {f"g{i}" for i in range(1, 13)})
    assert many["recall@5"] == 1.0           # capped recall: 12 gold, all top-5 correct


def test_rrf_fusion_rewards_agreement():
    from llama_index.core.vector_stores.types import VectorStoreQueryResult
    from src.retrieval.hybrid import rrf_fusion
    nodes = {i: TextNode(id_=i, text=i) for i in "abcd"}
    dense = VectorStoreQueryResult(nodes=[nodes["a"], nodes["b"], nodes["c"]], similarities=[0.9, 0.8, 0.7])
    sparse = VectorStoreQueryResult(nodes=[nodes["c"], nodes["d"]], similarities=[12.0, 3.0])
    fused = rrf_fusion(dense, sparse, top_k=3)
    assert fused.ids[0] == "c"               # ranked by both lists
    assert len(fused.ids) == 3 and set(fused.ids) <= set("abcd")


def test_multi_fact_metrics():
    from src.evaluation.retrieval import score
    s = score(["a1", "x", "y", "z", "q", "b1"], [["a1", "a2"], ["b1"]])
    assert s["all_hit@5"] == 0 and s["all_hit@10"] == 1     # fact b only found at rank 6
    assert s["fact_recall@10"] == 1.0 and s["hit@1"] == 1
    assert score(["a1"], [["a1"], ["b1"]])["fact_recall@10"] == 0.5


def test_parent_store_readonly_immutable_after_checkpoint(tmp_path):
    rw = ParentStore(tmp_path / "p.sqlite")
    rw.replace_filing("F1", [TextNode(id_="p0", text="parent 0")])
    rw.checkpoint()
    ro = ParentStore(tmp_path / "p.sqlite", readonly=True)
    assert ro.get(["p0"])["p0"].text == "parent 0"


def test_env_overrides_service_addresses(monkeypatch):
    from src.config import load_config
    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")
    monkeypatch.setenv("OLLAMA_HOST", "ollama:11434")
    assert load_config("qdrant")["url"] == "http://qdrant:6333"
    assert load_config("llm")["ollama"]["host"] == "ollama:11434"
    monkeypatch.delenv("QDRANT_URL")
    assert load_config("qdrant")["url"] == "http://127.0.0.1:6333"


def test_bm25_offline_snapshot_only_when_offline(tmp_path, monkeypatch):
    from src.retrieval.bm25 import _offline_snapshot
    repo = tmp_path / "models--Qdrant--bm25"
    (repo / "refs").mkdir(parents=True)
    (repo / "refs" / "main").write_text("abc123\n")
    (repo / "snapshots" / "abc123").mkdir(parents=True)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    assert _offline_snapshot("Qdrant/bm25", str(tmp_path)) is None       # online: fastembed's normal path
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert _offline_snapshot("Qdrant/bm25", str(tmp_path)) == str(repo / "snapshots" / "abc123")
    assert _offline_snapshot("Qdrant/other", str(tmp_path)) is None      # not cached
