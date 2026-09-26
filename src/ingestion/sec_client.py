"""Rate-limited HTTP client for SEC EDGAR with retries.

SEC fair-access rules: <=10 requests/second and a User-Agent that identifies
the requester with a contact email. Violations get the IP blocked for ~10 min.
"""
from __future__ import annotations

import logging
import time

import requests

from src.config import get_env, load_config

log = logging.getLogger(__name__)

# Statuses worth retrying; everything else (e.g. 404) fails immediately.
RETRYABLE_STATUS = {403, 429, 500, 502, 503, 504}
THROTTLE_MARKERS = (b"Request Rate Threshold Exceeded", b"Undeclared Automated Tool")


class SECClientError(RuntimeError):
    pass


class SECClient:
    def __init__(self, config: dict | None = None):
        cfg = (config or load_config())["sec"]
        user_agent = get_env(cfg["user_agent_env"])
        if not user_agent or "example.com" in user_agent or "@" not in user_agent:
            raise SECClientError(
                f"Set {cfg['user_agent_env']} in .env to 'Your Name your.email@domain' "
                "(SEC rejects requests without a contact User-Agent)."
            )
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
        self.min_interval = 1.0 / cfg["requests_per_second"]
        self.timeout = cfg["timeout_seconds"]
        self.max_retries = cfg["max_retries"]
        self.backoff_base = cfg["backoff_base_seconds"]
        self._last_request = 0.0

    def _throttle(self) -> None:
        wait = self.min_interval - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    def get(self, url: str) -> requests.Response:
        """GET with rate limiting and exponential backoff on transient failures."""
        last_error: str = ""
        for attempt in range(1, self.max_retries + 1):
            self._throttle()
            try:
                resp = self.session.get(url, timeout=self.timeout)
            except requests.RequestException as e:
                last_error = f"{type(e).__name__}: {e}"
            else:
                throttled = any(m in resp.content[:2000] for m in THROTTLE_MARKERS)
                if resp.status_code == 200 and not throttled:
                    return resp
                if resp.status_code not in RETRYABLE_STATUS and not throttled:
                    raise SECClientError(f"HTTP {resp.status_code} for {url}")
                last_error = "throttled by SEC" if throttled else f"HTTP {resp.status_code}"

            delay = self.backoff_base ** attempt
            log.warning("GET %s failed (%s), attempt %d/%d, retrying in %ss",
                        url, last_error, attempt, self.max_retries, delay)
            time.sleep(delay)
        raise SECClientError(f"Giving up on {url} after {self.max_retries} attempts: {last_error}")

    def get_json(self, url: str) -> dict:
        return self.get(url).json()
