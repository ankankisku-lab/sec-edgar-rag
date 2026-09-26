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
