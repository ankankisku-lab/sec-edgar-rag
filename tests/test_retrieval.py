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
