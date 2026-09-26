"""Verify downloaded filings against the manifest using their own iXBRL cover tags.

Every modern 10-K/10-Q embeds machine-readable cover data (dei:* tags). We check
that the document on disk is the filing the manifest says it is:

    dei:DocumentType          == form_type
    dei:DocumentPeriodEndDate == period_of_report
    dei:EntityCentralIndexKey == cik

and record the fiscal year/period labels, which later phases need because a
company's fiscal quarters rarely line up with calendar quarters.

Usage:
    python -m src.ingestion.verify
"""
from __future__ import annotations

import html
import logging
import re

import pandas as pd

from src.config import DATA_DIR, FILINGS_CSV, METADATA_DIR
from src.ingestion.checkpoint import Checkpoint
from src.ingestion.downloader import CHECKPOINT_PATH

log = logging.getLogger(__name__)

VERIFICATION_CSV = METADATA_DIR / "verification.csv"
DEI_TAGS = ("DocumentType", "DocumentPeriodEndDate", "EntityCentralIndexKey",
            "DocumentFiscalYearFocus", "DocumentFiscalPeriodFocus", "EntityRegistrantName")


def read_dei(text: str) -> dict[str, str]:
    values = {}
    for tag in DEI_TAGS:
        m = re.search(rf'name="dei:{tag}"[^>]*>', text, re.I)
        if m:
            values[tag] = _plain(_element_body(text, m.end()))
    return values


def _element_body(text: str, start: int) -> str:
    """Body of the ix:nonNumeric opened just before `start`, honouring nesting
    (e.g. DocumentPeriodEndDate wraps a CurrentFiscalYearEndDate element)."""
    depth = 1
    for m in re.finditer(r"<(/?)ix:nonNumeric\b", text[start:], re.I):
        depth += -1 if m.group(1) else 1
        if depth == 0:
            return text[start:start + m.start()]
    return ""


def _plain(fragment: str) -> str:
    return " ".join(html.unescape(re.sub(r"<[^>]+>", "", fragment)).split())


def verify_filing(row: pd.Series) -> dict:
    path = DATA_DIR / row["local_path"]
    text = path.read_bytes().decode("utf-8", errors="replace")
    dei = read_dei(text)
    issues = []

    doc_type = dei.get("DocumentType")
    if doc_type != row["form_type"]:
        issues.append(f"DocumentType={doc_type!r}")

    period = pd.to_datetime(dei.get("DocumentPeriodEndDate"), errors="coerce")
    if pd.isna(period) or period.date().isoformat() != row["period_of_report"]:
        issues.append(f"DocumentPeriodEndDate={dei.get('DocumentPeriodEndDate')!r}")

    cik = dei.get("EntityCentralIndexKey", "")
    if not cik.isdigit() or int(cik) != int(row["cik"]):
        issues.append(f"EntityCentralIndexKey={cik!r}")

    return {
        "filing_id": row["filing_id"],
        "status": "ok" if not issues else "mismatch",
        "issues": "; ".join(issues),
        "registrant_name": dei.get("EntityRegistrantName"),
        "fiscal_year": dei.get("DocumentFiscalYearFocus"),
        "fiscal_period": dei.get("DocumentFiscalPeriodFocus"),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    manifest = pd.read_csv(FILINGS_CSV, dtype={"cik": str})
    ckpt = Checkpoint(CHECKPOINT_PATH)
    downloaded = manifest[manifest["filing_id"].map(ckpt.is_done)]

    report = pd.DataFrame([verify_filing(row) for _, row in downloaded.iterrows()])
    report.to_csv(VERIFICATION_CSV, index=False)

    bad = report[report["status"] != "ok"]
    for _, r in bad.iterrows():
        log.warning("%s: %s", r["filing_id"], r["issues"])
    print(f"verified {len(report)}/{len(manifest)} manifest filings: "
          f"{(report['status'] == 'ok').sum()} ok, {len(bad)} mismatched -> {VERIFICATION_CSV.name}")


if __name__ == "__main__":
    main()
