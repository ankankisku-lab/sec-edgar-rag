"""Download raw 10-K / 10-Q documents listed in filings.csv.

For each filing: skip if already done -> download -> validate -> write file +
metadata sidecar -> checkpoint. Failures are logged and retried on the next run;
one bad filing never stops the batch.

Usage:
    python -m src.ingestion.downloader --tickers AAPL MSFT NVDA
    python -m src.ingestion.downloader --limit 20
    python -m src.ingestion.downloader                  # everything in the manifest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from datetime import datetime, timezone

import pandas as pd
from tqdm import tqdm

from src.config import CHECKPOINT_DIR, DATA_DIR, FILINGS_CSV, LOG_DIR, load_config
from src.ingestion.checkpoint import Checkpoint
from src.ingestion.sec_client import SECClient, SECClientError

log = logging.getLogger(__name__)

CHECKPOINT_PATH = CHECKPOINT_DIR / "download_state.json"


class ValidationError(ValueError):
    pass


def validate(content: bytes, form_type: str, min_bytes: int) -> None:
    """Reject anything that isn't plausibly the requested filing document."""
    if len(content) < min_bytes:
        raise ValidationError(f"too small ({len(content)} bytes)")
    head = content[:10_000].lower()
    if b"<html" not in head and b"<?xml" not in head:
        raise ValidationError("not an HTML/iXBRL document")
    if form_type.lower().encode() not in content.lower():
        raise ValidationError(f"document never mentions {form_type}")


def download_one(client: SECClient, row: pd.Series, min_bytes: int) -> dict:
    content = client.get(row["document_url"]).content
    validate(content, row["form_type"], min_bytes)

    path = DATA_DIR / row["local_path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(content)
    os.replace(tmp, path)  # never leave a half-written file under the final name

    info = {
        **json.loads(row.to_json()),  # to_json handles numpy scalars / NaN
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path.with_suffix(".json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return {"bytes": info["bytes"], "sha256": info["sha256"], "path": row["local_path"]}


def already_downloaded(ckpt: Checkpoint, row: pd.Series) -> bool:
    """Done in the checkpoint AND the file still exists with the recorded size."""
    if not ckpt.is_done(row["filing_id"]):
        return False
    path = DATA_DIR / row["local_path"]
    return path.exists() and path.stat().st_size == ckpt.get(row["filing_id"])["bytes"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", nargs="+", help="only these tickers")
    parser.add_argument("--forms", nargs="+", help="only these form types, e.g. 10-K")
    parser.add_argument("--limit", type=int, help="stop after this many filings from the manifest")
    args = parser.parse_args()

    LOG_DIR.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(LOG_DIR / "download.log", encoding="utf-8")],
    )

    cfg = load_config()
    manifest = pd.read_csv(FILINGS_CSV, dtype={"cik": str})
    if args.tickers:
        manifest = manifest[manifest["ticker"].isin([t.upper() for t in args.tickers])]
    if args.forms:
        manifest = manifest[manifest["form_type"].isin(args.forms)]
    if args.limit:
        manifest = manifest.head(args.limit)

    client = SECClient(cfg)
    ckpt = Checkpoint(CHECKPOINT_PATH)
    stats = {"downloaded": 0, "skipped": 0, "failed": 0}

    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="filings"):
        fid = row["filing_id"]
        if already_downloaded(ckpt, row):
            stats["skipped"] += 1
            continue
        try:
            result = download_one(client, row, cfg["download"]["min_bytes"])
        except (SECClientError, ValidationError, OSError) as e:
            log.error("%s failed: %s", fid, e)
            ckpt.mark(fid, "failed", error=str(e))
            stats["failed"] += 1
        else:
            log.info("%s ok (%d KB)", fid, result["bytes"] // 1024)
            ckpt.mark(fid, "done", error=None, **result)
            stats["downloaded"] += 1

    print(f"\n{stats} | checkpoint totals: {ckpt.summary()} | log: {LOG_DIR / 'download.log'}")
    if stats["failed"]:
        print("Re-run the same command to retry failures.")


if __name__ == "__main__":
    main()
