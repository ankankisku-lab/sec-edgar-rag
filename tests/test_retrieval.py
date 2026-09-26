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
