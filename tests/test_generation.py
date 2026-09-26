import pytest

from src.generation.numbers import apply_calcs, format_number, safe_eval, verify_numbers
from src.evaluation.generation import value_stated


def test_calc_is_evaluated_in_python():
    text, log = apply_calcs("Up $<calc>28202 - 25352</calc> million, or <calc>(28202-25352)/25352*100</calc>% [1].")
    assert text == "Up $2,850 million, or 11.24% [1]."
    assert [c["text"] for c in log] == ["2,850", "11.24"]


def test_calc_rejects_anything_but_arithmetic():
    with pytest.raises(ValueError):
        safe_eval("__import__('os').system('dir')")
    text, log = apply_calcs("<calc>1/0</calc>")
    assert "calculation failed" in text and "error" in log[0]


def test_format_number():
    assert format_number(2850.0) == "2,850"
    assert format_number(11.2421) == "11.24"
    assert format_number(-171.0) == "-171"


def test_verify_flags_numbers_not_in_sources():
    sources = "| Operating income | 28,202 | 25,352 |"
    assert verify_numbers("Operating income was $28,202 million [1].", sources) == []
    assert verify_numbers("It was $27,346 million [1].", sources) == ["$27,346"]
    # rounding a millions table to billions is fine; years, days and citations are ignored
    assert verify_numbers("About $28.2 billion in Q3 2025, quarter ended June 28, 2025 [1][2].", sources) == []


def test_value_stated():
    assert value_stated("Operating income was $28,202 million [1].", "28,202")
    assert value_stated("about $28.2 billion", "$28,202")
    assert value_stated("a loss of (171) million", "(171)")
    assert not value_stated("It was $25,352 million.", "28,202")


def test_score_answer_by_type():
    from src.evaluation.generation import score_answer
    yoy = {"type": "yoy", "answer_values": ["28,202", "25,352"], "expected_change": 2850.0}
    assert score_answer(yoy, "$28,202 million vs $25,352 million, up $2,850 million [1].")["correct"]
    assert not score_answer(yoy, "$28,202 million vs $25,352 million, up $850 million [1].")["correct"]
    pct = {"type": "pct_change", "answer_values": ["31,653", "27,032"], "expected_pct": 17.0946}
    assert score_answer(pct, "It rose 17.09% [1].")["correct"]
    assert score_answer(pct, "It rose 17.1% [1].")["correct"]
    assert not score_answer(pct, "It rose 16.2% [1].")["correct"]
    cc = {"type": "cross_company", "answer_values": ["416,161", "$331,839"]}
    assert score_answer(cc, "Apple $416,161 million [1] vs Microsoft $331,839 million [2].")["correct"]
    assert not score_answer(cc, "Apple $416,161 million [1].")["correct"]


def test_decomposition_parse_and_interleave():
    from llama_index.core.schema import NodeWithScore, TextNode
    from src.query.decomposition import parse
    from src.retrieval.decomposed import interleave
    q = "Compare A and B."
    assert parse('{"sub_questions": ["What was A?", "What was B?"]}', q) == ["What was A?", "What was B?"]
    assert parse('{"sub_questions": ["What was X revenue?"]}', "What drove X growth?") == ["What drove X growth?"]
    assert parse("not json", q) == [q]
    n = lambda i: NodeWithScore(node=TextNode(id_=i, text=i), score=1.0)
    merged = interleave([[n("a1"), n("a2"), n("s")], [n("b1"), n("s"), n("b2")]], top_k=5)
    assert [m.node.node_id for m in merged] == ["a1", "b1", "a2", "s", "b2"]


def test_router_only_sends_multi_fact_questions_to_the_llm():
    from src.query.decomposition import needs_decomposition
    assert needs_decomposition("Compare Apple's net income in fiscal 2025 Q1 with fiscal 2025 Q3.")
    assert needs_decomposition("By what percentage did Apple's revenue change from 2024 to 2025?")
    assert needs_decomposition("How did X compare with the same period a year earlier?")
    assert needs_decomposition("How has Nvidia's revenue trended over the last three quarters?")
    assert not needs_decomposition("What was Apple's operating income in fiscal 2025 Q3?")
    assert not needs_decomposition("What drove NVIDIA's Data Center revenue growth in fiscal 2027 Q2?")
