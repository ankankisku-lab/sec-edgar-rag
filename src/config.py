"""Project paths and config loading."""
from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"
METADATA_DIR = DATA_DIR / "metadata"
SUBMISSIONS_CACHE_DIR = METADATA_DIR / "submissions"
LOG_DIR = PROJECT_ROOT / "logs"

COMPANIES_CSV = METADATA_DIR / "companies.csv"
FILINGS_CSV = METADATA_DIR / "filings.csv"

load_dotenv(PROJECT_ROOT / ".env")


def load_config(name: str = "ingestion") -> dict:
    with open(CONFIG_DIR / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_env(name: str) -> str | None:
    return os.environ.get(name)
