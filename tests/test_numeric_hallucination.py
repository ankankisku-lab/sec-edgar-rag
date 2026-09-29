from src.evaluation.numeric_hallucination import consistent, fmt_like, gold_cells, locate, parse_sources, plants
from src.generation.numbers import number_tokens, verify_numbers

SOURCES = [
    "company: Apple\nperiod: fiscal 2025 Q3 (quarter ended June 28, 2025)\n\n"
    "|  | Three Months Ended |  |\n|---|---|---|\n|  | June 28, 2025 | June 29, 2024 |\n"
    "| Operating income | $28,202 | $25,352 |\n| Net income | $23,434 | $21,448 |\n",
    "company: Apple\nperiod: fiscal 2025 Q2 (quarter ended March 29, 2025)\n\n"
    "|  | Three Months Ended |  |\n|---|---|---|\n| Operating income | $29,589 | $27,900 |\n"
    "Operating income increased 11% year over year.",
]
Q = {"facts": [{"row_label": "Operating income", "value": "28,202", "prior_value": "25,352"}], "answer_values": ["28,202"]}


def test_percentages_up_to_31_are_invisible_to_verification():
    assert [t for t, _ in number_tokens("It rose 14% to $28,202 million on June 28.")] == ["$28,202"]
    assert [t for t, _ in number_tokens("It rose 14% on June 28.", keep_percent=True)] == ["14"]
    assert verify_numbers("It rose 14% [1].", "no such number") == []  # the gap Phase 17 measures


def test_parse_sources_reads_cells_and_headers():
    cells, prose, meta = parse_sources(SOURCES)
    assert meta == [("Apple", "fiscal 2025 Q3 (quarter ended June 28, 2025)"),
                    ("Apple", "fiscal 2025 Q2 (quarter ended March 29, 2025)")]
    assert {(c.label, c.value) for c in cells if c.src == 0 and c.label == "operating income"} == {
        ("operating income", 28202.0), ("operating income", 25352.0)}
    assert 11.0 in prose and 28202.0 not in prose


def test_locate_names_the_wrong_cell():
    cells, prose, meta = parse_sources(SOURCES)
    gold, labels = gold_cells(Q, cells)
    where = lambda v: locate(v, cells, prose, gold, labels, meta)  # noqa: E731
    assert where(28202) == "right cell"
    assert where(28.2) == "right cell"                      # rounded to billions
    assert where(25352) == "same row, other column"         # prior-year column
    assert where(23434) == "same column, other row"         # net income instead of operating income
    assert where(29589) == "same line item, other filing or period"
    assert where(11) == "text"
    assert where(27346) == "not in sources"


def test_planted_errors_keep_the_original_style():
    assert fmt_like("$28,202", 28203) == "$28,203"
    assert fmt_like("$(1,165)", 1282) == "$(1,282)"
    assert fmt_like("1.57", 1.73) == "1.73"
    cells, _, _ = parse_sources(SOURCES)
    gold, _ = gold_cells(Q, cells)
    kinds = {name: v for _, name, v in plants(28202.0, gold[0], cells, {28202.0})}
    assert kinds["last digit +1"] == 28203 and kinds["transposed digits"] == 28220
    assert kinds["same row, other column"] == 25352
    assert kinds["same column, other row"] == 23434         # net income, same period column
    assert kinds["same line item, other table"] == 29589    # operating income in the Q2 filing


def test_consistent_arithmetic_on_quoted_values():
    assert consistent(2850, [28202, 25352])                 # difference
    assert consistent(11.24, [28202, 25352])                # % change
    assert consistent(10706, [9541, 1165])                  # across zero: 9,541 vs (1,165)
    assert consistent(2400, [4.6, 2.2])                     # $4.6 billion - $2.2 billion in millions
    assert not consistent(3000, [28202, 25352])
