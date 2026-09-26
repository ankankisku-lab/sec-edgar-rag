"""Start/stop the portable Ollama server with project settings (everything on D:)."""
from __future__ import annotations

import os
import subprocess
import time

import requests

from src.config import LOG_DIR, PROJECT_ROOT, load_config


def base_url() -> str:
    return f"http://{load_config('llm')['ollama']['host']}"


def is_up() -> bool:
    try:
        return requests.get(base_url() + "/api/version", timeout=2).ok
    except requests.RequestException:
        return False


def start(kv_cache_type: str = "q8_0") -> subprocess.Popen:
    """Launch `ollama serve`; the KV cache type is a server-wide setting."""
    cfg = load_config("llm")["ollama"]
    env = dict(os.environ,
               OLLAMA_HOST=cfg["host"],
               OLLAMA_MODELS=str(PROJECT_ROOT / cfg["models_dir"]),
               OLLAMA_FLASH_ATTENTION="1" if cfg["flash_attention"] else "0",
               OLLAMA_KV_CACHE_TYPE=kv_cache_type,
               OLLAMA_NUM_PARALLEL="1",        # parallel slots multiply KV-cache memory
               OLLAMA_MAX_LOADED_MODELS="1",
               # Default context for requests that don't set num_ctx (the OpenAI-compatible
               # endpoint can't): too small a default silently truncates long prompts.
               OLLAMA_CONTEXT_LENGTH=str(cfg.get("default_context_length", 8192)))
    LOG_DIR.mkdir(exist_ok=True)
    log = open(LOG_DIR / f"ollama_{kv_cache_type}.log", "a", encoding="utf-8")
    proc = subprocess.Popen([str(PROJECT_ROOT / cfg["exe"]), "serve"], env=env, stdout=log, stderr=log,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    for _ in range(60):
        if is_up():
            return proc
        time.sleep(0.5)
    proc.kill()
    raise RuntimeError("ollama serve did not start; see logs/ollama_*.log")


def stop(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
    # The server spawns model runners (llama-server.exe) that can outlive it and keep
    # holding VRAM; an orphaned runner starves the next run into CPU spill.
    for image in ("ollama.exe", "llama-server.exe"):
        subprocess.run(["taskkill", "/F", "/IM", image, "/T"], capture_output=True)
