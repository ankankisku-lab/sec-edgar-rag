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

# Keep downloaded models on D: with the project (the default is the user profile on C:).
MODELS_DIR = PROJECT_ROOT / "models"
os.environ.setdefault("HF_HOME", str(MODELS_DIR / "huggingface"))
# LlamaIndex's HuggingFaceEmbedding ignores HF_HOME and uses its own cache dir.
os.environ.setdefault("LLAMA_INDEX_CACHE_DIR", str(MODELS_DIR / "llama_index"))
# fastembed (BM25 sparse model) defaults to the system temp dir on C:.
os.environ.setdefault("FASTEMBED_CACHE_PATH", str(MODELS_DIR / "fastembed"))


# Environment overrides for service addresses, so the same YAML works natively (127.0.0.1)
# and inside docker compose (service names such as qdrant:6333).
ENV_OVERRIDES = {
    ("qdrant", ("url",)): "QDRANT_URL",
    ("llm", ("ollama", "host")): "OLLAMA_HOST",
    ("observability", ("tracing", "endpoint")): "PHOENIX_COLLECTOR_ENDPOINT",
    # per-request keep-alive overrides Ollama's own setting; the serving stack keeps the LLM loaded
    ("generation", ("llm", "keep_alive")): "LLM_KEEP_ALIVE",
}


def load_config(name: str = "ingestion") -> dict:
    with open(CONFIG_DIR / f"{name}.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for (cfg_name, keys), env in ENV_OVERRIDES.items():
        if cfg_name == name and os.environ.get(env):
            node = cfg
            for k in keys[:-1]:
                node = node[k]
            node[keys[-1]] = os.environ[env]
    return cfg


def get_env(name: str) -> str | None:
    return os.environ.get(name)
