from llama_index.core.schema import MetadataMode, NodeRelationship

from src.chunking.hierarchical import chunk_filing, n_tokens, parse_markdown
from src.parsing import tables

CFG = {"text": {"child_tokens": 64, "child_overlap_tokens": 8, "parent_min_tokens": 40, "parent_max_tokens": 200},
       "tables": {"child_tokens": 80, "parent_max_tokens": 400}}
SECTION = {"section": "Part I, Item 2. MD&A", "part": "I", "item": "2", "note": None}


def _doc(elements):
    for i, e in enumerate(elements):
        e["idx"] = i
    return {"filing_id": "TEST_10Q_2025-06-28", "company": "Test Co", "ticker": "TEST", "cik": "0000000001",
            "form_type": "10-Q", "filing_date": "2025-08-01", "period_of_report": "2025-06-28",
            "fiscal_year": "2025", "fiscal_period": "Q3", "accession_number": "0000000001-25-000001",
            "document_url": "https://example.test", "elements": elements}


def _table_doc(n_rows=40, n_footnotes=0):
    grid = [["", "June 28, 2025", "June 29, 2024"]] + [[f"Line item {i}", f"{i},000", f"{i},500"] for i in range(n_rows)]
    els = [{"type": "table", "text": tables.to_markdown(grid), "caption": "STATEMENTS OF OPERATIONS",
            "units": "millions", "subsection": None, **SECTION}]
    els += [{"type": "footnote", "text": f"({k}) " + "Long footnote sentence about adjustments. " * 12,
             "table_ref": 0, "subsection": None, **SECTION} for k in range(n_footnotes)]
    return _doc(els)


def test_every_table_child_repeats_period_header():
    parents, children = chunk_filing(_table_doc(), CFG)
    # 40 rows exceed the 400-token parent budget: two parents, each self-describing.
    assert len(parents) == 2 and len(children) > 4
    assert "(part 1 of 2)" in parents[0].text and "Line item 39" in parents[1].text
    for node in parents + children:
        table_md = node.text[node.text.index("|"):].split("\n\nFootnotes")[0]
        assert parse_markdown(table_md)[0] == ["", "June 28, 2025", "June 29, 2024"]
        assert node.metadata["units"] == "millions" and node.metadata["table_caption"] == "STATEMENTS OF OPERATIONS"


def test_footnotes_are_split_within_budget():
    _, children = chunk_filing(_table_doc(n_rows=3, n_footnotes=6), CFG)
    notes = [c for c in children if c.text.startswith("Footnotes:")]
    assert len(notes) > 1
    assert all(n_tokens(c.text) <= CFG["tables"]["child_tokens"] + 5 for c in notes)


def test_children_link_to_parents_and_ids_are_deterministic():
    doc = _table_doc()
    parents, children = chunk_filing(doc, CFG)
    parents2, children2 = chunk_filing(doc, CFG)
    assert [c.node_id for c in children] == [c.node_id for c in children2]
    for parent in parents:
        kids = {r.node_id for r in parent.relationships[NodeRelationship.CHILD]}
        assert kids and all(c.relationships[NodeRelationship.PARENT].node_id == parent.node_id
                            for c in children if c.node_id in kids)
    all_kids = set().union(*({r.node_id for r in p.relationships[NodeRelationship.CHILD]} for p in parents))
    assert all_kids == {c.node_id for c in children}


def test_embed_text_carries_company_and_period_but_not_ids():
    _, children = chunk_filing(_table_doc(), CFG)
    embedded = children[0].get_content(MetadataMode.EMBED)
    assert "Test Co" in embedded and "fiscal 2025 Q3 (quarter ended June 28, 2025)" in embedded
    assert "parent_id" not in embedded and "accession_number" not in embedded


def test_merged_text_parent_labels_children_by_their_own_subheading():
    body = "Revenue increased because of higher unit volumes across regions. " * 6
    els = [{"type": "heading", "heading_kind": "heading", "text": "Segment Performance", "subsection": "Segment Performance", **SECTION},
           {"type": "paragraph", "text": "Short intro.", "subsection": "Segment Performance", **SECTION},
           {"type": "heading", "heading_kind": "heading", "text": "Gross Margin", "subsection": "Gross Margin", **SECTION},
           {"type": "paragraph", "text": body, "subsection": "Gross Margin", **SECTION}]
    parents, children = chunk_filing(_doc(els), CFG)
    assert len(parents) == 1  # small groups merged
    assert children[-1].metadata["subsection"] == "Gross Margin"
    assert children[-1].metadata["heading_path"].endswith("> Gross Margin")
