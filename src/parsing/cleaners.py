"""Text normalisation and page-artifact removal."""
from __future__ import annotations

import re
from collections import Counter

_WS = re.compile(r"\s+")
_INVISIBLE = dict.fromkeys(map(ord, "​‌‍﻿­"), None)
_PAGE_NUMBER = re.compile(r"^(page\s+)?[-–—]?\s*\(?[ivxlc\d]{1,4}\)?\s*[-–—]?$", re.I)


def normalize(text: str) -> str:
    """Collapse whitespace (incl. &nbsp;) and drop zero-width characters."""
    return _WS.sub(" ", text.translate(_INVISIBLE).replace("\xa0", " ")).strip()


def is_page_number(text: str) -> bool:
    return bool(_PAGE_NUMBER.match(text))


def page_artifact_indices(blocks: list[dict], window: int = 2, min_repeats: int = 3) -> set[int]:
    """Indices of running headers/footers and page numbers.

    A block is an artifact if it sits within `window` blocks of a page break,
    is short, and (after replacing digits) repeats on at least `min_repeats`
    pages, e.g. "Apple Inc. | Q3 2025 Form 10-Q | 17" or Microsoft's "PART II /
    Item 7" page headers. Bare page numbers and "Table of Contents" links near
    page breaks are always artifacts.
    """
    breaks = [i for i, b in enumerate(blocks) if b["kind"] == "pagebreak"]
    near: set[int] = set()
    for br in breaks:
        # step outward from the break over text blocks only
        for direction in (-1, 1):
            i, seen = br + direction, 0
            while 0 <= i < len(blocks) and seen < window and blocks[i]["kind"] != "pagebreak":
                if blocks[i]["kind"] == "text" and len(blocks[i]["text"]) <= 120:
                    near.add(i)
                seen += 1
                i += direction

    def key(i: int) -> str:
        return re.sub(r"\d+", "#", blocks[i]["text"].lower())

    counts = Counter(key(i) for i in near)
    artifacts = set()
    for i in near:
        text = blocks[i]["text"]
        repeated = counts[key(i)] >= min_repeats and len(text) <= 80  # footers are short
        if is_page_number(text) or text.lower() in ("table of contents", "index") or repeated:
            artifacts.add(i)
    return artifacts
