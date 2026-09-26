"""Build the source-of-truth manifests before any document is downloaded.

    Wikipedia Nasdaq-100 list -> companies.csv (ticker, company, CIK)
    SEC submissions API       -> filings.csv   (one row per 10-K / 10-Q to fetch)

Usage:
    python -m src.ingestion.metadata            # uses cached submissions JSON
    python -m src.ingestion.metadata --refresh  # re-fetch from SEC
"""
from __future__ import annotations

import argparse
import io
import json
import logging
from datetime import date
from pathlib import PurePosixPath

import pandas as pd
import requests

from src.config import (
    COMPANIES_CSV, DATA_DIR, FILINGS_CSV, METADATA_DIR, SUBMISSIONS_CACHE_DIR, load_config,
)
from src.ingestion.sec_client import SECClient

log = logging.getLogger(__name__)

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/{name}"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"
# Wikipedia is not SEC: no personal contact info is sent there.
WIKI_HEADERS = {"User-Agent": "SEC-EDGAR-RAG/0.1 (research project; python-requests)"}


# --------------------------------------------------------------------------- companies

def fetch_nasdaq100(url: str) -> pd.DataFrame:
    html = requests.get(url, headers=WIKI_HEADERS, timeout=30).text
    for table in pd.read_html(io.StringIO(html)):
        cols = {str(c).strip().lower(): c for c in table.columns}
        ticker_col = cols.get("ticker") or cols.get("symbol")
        company_col = cols.get("company") or cols.get("security")
        if ticker_col is not None and company_col is not None and len(table) >= 90:
            df = pd.DataFrame({
                "company": table[company_col].astype(str).str.strip(),
                "ticker": table[ticker_col].astype(str).str.strip().str.upper(),
            })
            for name in ("gics sector", "gics sub-industry"):
                if name in cols:
                    df[name.replace(" ", "_").replace("-", "_")] = table[cols[name]]
            return df
    raise RuntimeError(f"No Nasdaq-100 constituents table found at {url}")


def fetch_ticker_to_cik(client: SECClient) -> dict[str, str]:
    data = client.get_json(TICKER_MAP_URL)
    return {row["ticker"].upper(): f"{int(row['cik_str']):010d}" for row in data.values()}


def build_companies(client: SECClient, cfg: dict) -> pd.DataFrame:
    df = fetch_nasdaq100(cfg["universe"]["wikipedia_url"])
    log.info("Wikipedia: %d constituents", len(df))
    ticker_map = fetch_ticker_to_cik(client)
    # SEC writes share classes with '-' (BRK-B), Wikipedia sometimes with '.'.
    df["cik"] = df["ticker"].map(lambda t: ticker_map.get(t) or ticker_map.get(t.replace(".", "-")))

    missing = df[df["cik"].isna()]
    for _, row in missing.iterrows():
        log.warning("No CIK for %s (%s): skipped", row["ticker"], row["company"])
    df = df.dropna(subset=["cik"])

    # Multiple share classes (GOOGL/GOOG) are one filer: keep the first ticker.
    dupes = df[df.duplicated("cik", keep="first")]
    for _, row in dupes.iterrows():
        log.info("%s shares CIK %s with another ticker: dropped duplicate", row["ticker"], row["cik"])
    df = df.drop_duplicates("cik", keep="first").reset_index(drop=True)
    df["universe_snapshot_date"] = date.today().isoformat()
    return df


# --------------------------------------------------------------------------- filings

def load_submissions(client: SECClient, cik: str, refresh: bool) -> dict:
    """Company submissions JSON, cached on disk so re-runs don't hit SEC."""
    path = SUBMISSIONS_CACHE_DIR / f"CIK{cik}.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text(encoding="utf-8"))
    data = client.get_json(SUBMISSIONS_URL.format(name=f"CIK{cik}.json"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return data


def _rows(block: dict) -> list[dict]:
    """Submissions JSON stores filings column-wise; turn them into row dicts."""
    keys = list(block.keys())
    return [dict(zip(keys, values)) for values in zip(*(block[k] for k in keys))]


def select_filings(client: SECClient, submissions: dict, per_company: dict[str, int],
                   refresh: bool) -> list[dict]:
    """Latest N filings per form, newest first.

    `filings.recent` covers roughly the last 1000 filings; older ones live in
    extra pages that we only fetch when the recent page doesn't have enough.
    """
    wanted = dict(per_company)
    picked: list[dict] = []

    pages = [submissions["filings"]["recent"]]
    older = [f["name"] for f in submissions["filings"].get("files", [])]
    while pages:
        for row in sorted(_rows(pages.pop(0)), key=lambda r: r["filingDate"], reverse=True):
            if wanted.get(row["form"], 0) > 0:
                picked.append(row)
                wanted[row["form"]] -= 1
        if not any(wanted.values()) or not older:
            break
        name = older.pop(0)
        path = SUBMISSIONS_CACHE_DIR / name
        if path.exists() and not refresh:
            pages.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            page = client.get_json(SUBMISSIONS_URL.format(name=name))
            path.write_text(json.dumps(page), encoding="utf-8")
            pages.append(page)
    return picked


def build_manifest(client: SECClient, companies: pd.DataFrame, cfg: dict,
                   refresh: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_company = cfg["filings"]["per_company"]
    records, fiscal_year_ends = [], {}

    for _, co in companies.iterrows():
        subs = load_submissions(client, co["cik"], refresh)
        fiscal_year_ends[co["cik"]] = subs.get("fiscalYearEnd")
        filings = select_filings(client, subs, per_company, refresh)
        if not filings:
            forms = sorted({f for f in subs["filings"]["recent"]["form"]} & {"20-F", "40-F", "6-K"})
            log.warning("%s: no 10-K/10-Q (foreign filer? files %s)", co["ticker"], forms or "other forms")
        for f in filings:
            acc = f["accessionNumber"]
            period = f.get("reportDate") or f["filingDate"]
            filing_id = f"{co['ticker']}_{f['form'].replace('-', '')}_{period}"
            ext = PurePosixPath(f["primaryDocument"]).suffix or ".htm"
            records.append({
                "filing_id": filing_id,
                "company": co["company"],
                "ticker": co["ticker"],
                "cik": co["cik"],
                "form_type": f["form"],
                "filing_date": f["filingDate"],
                "period_of_report": period,
                "accession_number": acc,
                "primary_document": f["primaryDocument"],
                "is_inline_xbrl": bool(f.get("isInlineXBRL")),
                "document_url": ARCHIVE_URL.format(
                    cik=int(co["cik"]), acc=acc.replace("-", ""), doc=f["primaryDocument"]),
                "local_path": f"raw/{co['ticker']}/{filing_id}{ext}",
            })

    manifest = pd.DataFrame(records)
    # Disambiguate the rare case of two filings of one form for the same period.
    dup = manifest.duplicated("filing_id", keep=False)
    if dup.any():
        suffix = "_" + manifest.loc[dup, "accession_number"].str[-6:]
        manifest.loc[dup, "filing_id"] += suffix
        manifest.loc[dup, "local_path"] = [
            f"raw/{t}/{fid}{PurePosixPath(p).suffix}"
            for t, fid, p in manifest.loc[dup, ["ticker", "filing_id", "local_path"]].itertuples(index=False)
        ]
    assert manifest["filing_id"].is_unique and manifest["accession_number"].is_unique

    companies = companies.copy()
    companies["fiscal_year_end"] = companies["cik"].map(fiscal_year_ends)
    counts = manifest.groupby(["cik", "form_type"]).size().unstack(fill_value=0)
    for form in per_company:
        companies[f"n_{form.replace('-', '').lower()}"] = (
            companies["cik"].map(counts[form] if form in counts else {}).fillna(0).astype(int))
    return companies, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--refresh", action="store_true", help="re-fetch SEC submissions instead of using cache")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = load_config()
    client = SECClient(cfg)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    companies = build_companies(client, cfg)
    companies, manifest = build_manifest(client, companies, cfg, args.refresh)
    companies.to_csv(COMPANIES_CSV, index=False)
    manifest.to_csv(FILINGS_CSV, index=False)

    log.info("companies.csv: %d filers (%d with no 10-K/10-Q)",
             len(companies), int((companies.filter(like="n_").sum(axis=1) == 0).sum()))
    log.info("filings.csv: %d filings %s", len(manifest), manifest["form_type"].value_counts().to_dict())
    log.info("Wrote %s and %s", COMPANIES_CSV.relative_to(DATA_DIR.parent), FILINGS_CSV.relative_to(DATA_DIR.parent))


if __name__ == "__main__":
    main()
