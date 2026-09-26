"""Section detection: Part / Item / Note headings -> canonical section labels."""
from __future__ import annotations

import re

ITEMS_10K = {
    "1": "Business", "1A": "Risk Factors", "1B": "Unresolved Staff Comments", "1C": "Cybersecurity",
    "2": "Properties", "3": "Legal Proceedings", "4": "Mine Safety Disclosures",
    "5": "Market for Registrant's Common Equity, Related Stockholder Matters and Issuer Purchases of Equity Securities",
    "6": "[Reserved]", "7": "Management's Discussion and Analysis of Financial Condition and Results of Operations",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9": "Changes in and Disagreements with Accountants on Accounting and Financial Disclosure",
    "9A": "Controls and Procedures", "9B": "Other Information",
    "9C": "Disclosure Regarding Foreign Jurisdictions that Prevent Inspections",
    "10": "Directors, Executive Officers and Corporate Governance", "11": "Executive Compensation",
    "12": "Security Ownership of Certain Beneficial Owners and Management and Related Stockholder Matters",
    "13": "Certain Relationships and Related Transactions, and Director Independence",
    "14": "Principal Accountant Fees and Services", "15": "Exhibits and Financial Statement Schedules",
    "16": "Form 10-K Summary",
}
ITEMS_10Q = {
    ("I", "1"): "Financial Statements",
    ("I", "2"): "Management's Discussion and Analysis of Financial Condition and Results of Operations",
    ("I", "3"): "Quantitative and Qualitative Disclosures About Market Risk",
    ("I", "4"): "Controls and Procedures",
    ("II", "1"): "Legal Proceedings", ("II", "1A"): "Risk Factors",
    ("II", "2"): "Unregistered Sales of Equity Securities and Use of Proceeds",
    ("II", "3"): "Defaults Upon Senior Securities", ("II", "4"): "Mine Safety Disclosures",
    ("II", "5"): "Other Information", ("II", "6"): "Exhibits",
}

_SEP = r"\s*[.:\-–—]\s*"
# "Item 7. Management's Discussion..." or a bold bare "Item 7." -- but not running
# headers like "Item 7" or cross-reference lists like "Item 9B, 9C, 10".
# "Items 1 and 2. Business and Properties" covers two items; we label it with the first.
ITEM_RE = re.compile(
    rf"^items?\s+(\d{{1,2}}[a-c]?)((?:\s*(?:and|&|,)\s*\d{{1,2}}[a-c]?)*)(?:{_SEP}(\S.*)|\s*(\.?))$", re.I)
EMBEDDED_ITEM_RE = re.compile(r"item\s+(\d{1,2}[a-c]?)", re.I)
AUDITOR_RE = re.compile(r"report\s+of\s+independent\s+registered\s+public\s+accounting\s+firm", re.I)
# 10-Q financial statements that start without an "Item 1" heading.
STATEMENT_RE = re.compile(
    r"(condensed\s+)?consolidated\s+(statements?|balance\s+sheets?)|"
    r"statements?\s+of\s+(operations|income|earnings|financial\s+position|cash\s+flows)", re.I)
PART_RE = re.compile(r"^part\s+(iv|i{1,3})\b(?!\s*,)", re.I)
NOTE_RE = re.compile(rf"^note\s+(\d{{1,2}}){_SEP}(\S.*)$", re.I)
MAX_HEADING_CHARS = 200


def item_sort_key(code: str) -> tuple[int, str]:
    m = re.match(r"(\d+)([A-C]?)", code)
    return int(m.group(1)), m.group(2)


def canonical_item_title(form_type: str, part: str | None, code: str) -> str | None:
    if form_type == "10-K":
        return ITEMS_10K.get(code)
    return ITEMS_10Q.get((part or "I", code))


def _norm_title(text: str) -> str:
    text = re.sub(r"\(unaudited\)", "", text.lower().replace("’", "'"))
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


# Fallback for filers that never write "Item N" (e.g. Intel, AEP): exact section titles.
TITLES_10K = {_norm_title(t): (None, code) for code, t in ITEMS_10K.items() if code != "6"}
TITLES_10Q = {_norm_title(t): key for key, t in ITEMS_10Q.items()}


def match_item_title(form_type: str, text: str) -> tuple[str | None, str] | None:
    norm = _norm_title(text)
    titles = TITLES_10K if form_type == "10-K" else TITLES_10Q
    # Only long, distinctive titles: short ones ("Legal Proceedings", "Risk Factors")
    # also appear as subheadings inside notes and would jump to the wrong section.
    for title, key in titles.items():
        if len(title) >= 25 and len(norm) >= 25 and (norm.startswith(title) or title.startswith(norm)):
            return key
    return None


class SectionTracker:
    """Walks headings in document order and keeps the current section state.

    Item numbers must increase within a part; a lower number is a cross
    reference or running header, not a new section.
    """

    def __init__(self, form_type: str):
        self.form_type = form_type
        self.part: str | None = None
        self.item: str | None = None
        self.item_title: str | None = None
        self.note: str | None = None
        self.heading: str | None = None
        self.saw_item_heading = False

    def _set_part(self, part: str) -> None:
        self.part, self.item, self.item_title, self.note, self.heading = part, None, None, None, None
        if self.form_type == "10-Q" and part == "I":
            # Part I of a 10-Q opens with the financial statements (Item 1), and
            # many filers never print an "Item 1" heading.
            self._set_item("1", None)

    def _set_item(self, code: str, title: str | None, combined: bool = False) -> None:
        if self.form_type == "10-Q" and self.part is None:
            self.part = "I"
        self.item = code
        canonical = canonical_item_title(self.form_type, self.part, code)
        # "Items 1 and 2. Business and Properties": the printed title is more accurate.
        self.item_title = title if (combined and title) else (canonical or title)
        self.note, self.heading = None, None

    def observe(self, text: str, bold: bool) -> str | None:
        """Update state from a text block; return its heading kind or None."""
        if len(text) > MAX_HEADING_CHARS:
            return None

        m = PART_RE.match(text)
        if m and (bold or len(text) < 60):
            part = m.group(1).upper()
            if part != self.part:
                self._set_part(part)
            # "PART I. FINANCIAL INFORMATION ITEM 1. FINANCIAL STATEMENTS" on one line
            embedded = EMBEDDED_ITEM_RE.search(text, m.end())
            if embedded:
                self._set_item(embedded.group(1).upper(), None)
            return "part"

        m = ITEM_RE.match(text)
        # Needs a title, bold, or a trailing period: bare non-bold "Item 7" is a running header.
        if m and (m.group(3) or bold or m.group(4)):
            code = m.group(1).upper()
            self.saw_item_heading = True
            if self.item is None or item_sort_key(code) > item_sort_key(self.item):
                self._set_item(code, m.group(3), combined=bool(m.group(2)))
                return "item"
            return None

        if not self.saw_item_heading and (bold or len(text) < 120):
            key = match_item_title(self.form_type, text)
            if key:
                part, code = key
                if part and part != self.part:
                    self._set_part(part)
                self._set_item(code, None)
                return "item"

        if (self.form_type == "10-Q" and self.item is None and self.part in (None, "I")
                and STATEMENT_RE.search(text) and len(text) < 150):
            self._set_part("I")
            self._set_item("1", None)
            self.heading = text
            return "heading"

        # Non-standard 10-Ks (no "Item N" headings): financial statements are Item 8.
        if (self.form_type == "10-K" and not self.saw_item_heading and bold and len(text) < 150
                and (self.item is None or item_sort_key(self.item) < (8, ""))
                and (STATEMENT_RE.match(text) or AUDITOR_RE.match(text))):
            self._set_item("8", None)
            self.heading = text
            return "heading"

        m = NOTE_RE.match(text)
        if m and len(text) < 150:
            self.note = f"Note {m.group(1)} - {m.group(2)}"
            self.heading = None
            return "note"

        if bold and not text.endswith((".", ",", ";", ":")) or (bold and len(text) < 80):
            self.heading = text
            return "heading"
        return None

    def labels(self) -> dict:
        if self.item is None:
            section = "Cover Page"
        elif self.form_type == "10-Q":
            section = f"Part {self.part}, Item {self.item}. {self.item_title}"
        else:
            section = f"Item {self.item}. {self.item_title}"
        return {"part": self.part, "item": self.item, "section": section,
                "note": self.note, "subsection": self.heading}
