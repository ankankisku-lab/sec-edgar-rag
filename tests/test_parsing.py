import lxml.html

from src.parsing import tables
from src.parsing.cleaners import page_artifact_indices
from src.parsing.structure import SectionTracker


def _track(form, lines):
    """Feed (text, bold) lines; return the section label after each."""
    t = SectionTracker(form)
    out = []
    for text, bold in lines:
        t.observe(text, bold)
        out.append(t.labels()["section"])
    return out


# ------------------------------------------------------------------ sections

def test_standard_10k_items_and_running_headers():
    s = _track("10-K", [
        ("PART II", True), ("ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS", True),
        ("Item 7", False),          # running page header: ignored
        ("Item 1A", False),         # cross-reference header: ignored
        ("ITEM 7A. QUANTITATIVE AND QUALITATIVE DISCLOSURES", True),
    ])
    assert s[1].startswith("Item 7. Management")
    assert s[2] == s[1] and s[3] == s[1]
    assert s[4].startswith("Item 7A.")


def test_item_numbers_must_increase():
    s = _track("10-K", [("Item 8. Financial Statements", True), ("Item 1A. Risk Factors", True)])
    assert s[1] == s[0]


def test_10q_part_i_without_item_1_heading_is_financial_statements():
    s = _track("10-Q", [("PART I. FINANCIAL INFORMATION", True), ("CONSOLIDATED BALANCE SHEETS", True),
                        ("ITEM 2. MANAGEMENT'S DISCUSSION", True), ("PART II. OTHER INFORMATION", True),
                        ("ITEM 1A. RISK FACTORS", True)])
    assert s[0] == s[1] == "Part I, Item 1. Financial Statements"
    assert s[2].startswith("Part I, Item 2.")
    assert s[4] == "Part II, Item 1A. Risk Factors"


def test_10q_statements_without_any_part_or_item_heading():
    s = _track("10-Q", [("Some cover text", False), ("Condensed Consolidated Statements of Operations", True)])
    assert s[0] == "Cover Page"
    assert s[1] == "Part I, Item 1. Financial Statements"


def test_part_and_item_on_one_line_and_bare_item_with_period():
    assert _track("10-Q", [("PART I. FINANCIAL INFORMATION ITEM 1. FINANCIAL STATEMENTS", True)])[0] \
        == "Part I, Item 1. Financial Statements"
    s = _track("10-Q", [("PART I", True), ("ITEM 2.", False)])
    assert s[1].startswith("Part I, Item 2. Management")


def test_combined_items_keep_printed_title():
    s = _track("10-K", [("ITEMS 1 AND 2. BUSINESS AND PROPERTIES", True)])
    assert s[0] == "Item 1. BUSINESS AND PROPERTIES"


def test_title_fallback_only_for_distinctive_titles():
    s = _track("10-K", [("Legal Proceedings", True),
                        ("Management's Discussion and Analysis of Financial Condition and Results of Operations", True)])
    assert s[0] == "Cover Page"
    assert s[1].startswith("Item 7.")


# ------------------------------------------------------------------ tables

def _grid(html):
    return tables.clean_grid(lxml.html.fragment_fromstring(html))


def test_currency_and_parentheses_cells_are_merged_with_values():
    g = _grid("""<table>
      <tr><td></td><td colspan="3">2025</td><td colspan="3">2024</td></tr>
      <tr><td>Net income</td><td>$</td><td>1,234</td><td></td><td>$</td><td>(56</td><td>)</td></tr>
      <tr><td>Other</td><td></td><td>7</td><td></td><td></td><td>8</td><td></td></tr>
    </table>""")
    assert g == [["", "2025", "2024"], ["Net income", "$1,234", "$(56)"], ["Other", "7", "8"]]


def test_spanned_period_header_is_not_merged_into_label_column():
    # Microsoft balance sheet layout: dates sit in colspan cells left of the values.
    g = _grid("""<table>
      <tr><td></td><td></td><td colspan="2">March 31, 2025</td><td></td><td colspan="2">June 30, 2024</td></tr>
      <tr><td>Cash</td><td></td><td>$</td><td>28,828</td><td></td><td>$</td><td>18,315</td></tr>
    </table>""")
    assert g == [["", "March 31, 2025", "June 30, 2024"], ["Cash", "$28,828", "$18,315"]]


def test_classify():
    assert tables.classify([["Item 1.", "Business", "3"], ["Item 1A.", "Risk Factors", "9"],
                            ["Item 2.", "Properties", "20"]]) == "toc"
    assert tables.classify([["•", "We sell products worldwide."]]) == "layout"
    assert tables.classify([["", "2025", "2024"], ["Revenue", "$10", "$9"]]) == "data"


def test_units_and_percentages():
    assert tables.find_units("(In millions, except per share amounts)") == "millions"
    assert tables.all_percentages([["", "2025"], ["Margin", "46.5%"]])
    assert not tables.has_header([["Gross margin percentage:", "", ""], ["Products", "34.5%", "35.3%"]])


# ------------------------------------------------------------------ page artifacts

def test_running_footers_and_page_numbers_are_dropped():
    blocks = []
    for page in range(1, 5):
        blocks += [{"kind": "text", "text": f"Real content on page {page}. " * 5},
                   {"kind": "text", "text": f"Apple Inc. | Q3 2025 Form 10-Q | {page}"},
                   {"kind": "pagebreak"}]
    dropped = page_artifact_indices(blocks)
    assert all(blocks[i]["text"].startswith("Apple Inc. |") for i in dropped)
    assert len(dropped) == 4


def test_nonstandard_10k_statements_become_item_8():
    s = _track("10-K", [("MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS", True),
                        ("MARKET FOR REGISTRANT'S COMMON EQUITY, RELATED STOCKHOLDER MATTERS AND ISSUER PURCHASES", True),
                        ("CONSOLIDATED STATEMENT OF OPERATIONS", True)])
    assert s[0].startswith("Item 7.") and s[1].startswith("Item 5.")
    assert s[2].startswith("Item 8.")


def test_shortened_title_fallback():
    assert _track("10-K", [("Management's Discussion and Analysis", True)])[0].startswith("Item 7.")


def test_emphasis_counts_large_font_as_heading():
    from src.parsing.parser import emphasis_fraction
    el = lxml.html.fragment_fromstring(
        '<div style="font-size:9pt"><span style="font-size:14pt;font-weight:400">Market Trends</span></div>')
    assert emphasis_fraction(el, body_pt=9.0) == 1.0
    assert emphasis_fraction(el, body_pt=14.0) == 0.0
