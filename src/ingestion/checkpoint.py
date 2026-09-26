"""Persistent per-item progress tracking so interrupted runs resume where they stopped."""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path


class Checkpoint:
    """A JSON map of item_id -> state, rewritten atomically after every update.

    A crash mid-write leaves the previous file intact (write to .tmp, then
    os.replace), so the checkpoint itself can never be corrupted.
    """

    def __init__(self, path: Path):
        self.path = path
        self.state: dict[str, dict] = {}
        if path.exists():
            self.state = json.loads(path.read_text(encoding="utf-8"))

    def get(self, item_id: str) -> dict | None:
        return self.state.get(item_id)

    def is_done(self, item_id: str) -> bool:
        return self.state.get(item_id, {}).get("status") == "done"

    def mark(self, item_id: str, status: str, **fields) -> None:
        entry = self.state.setdefault(item_id, {"attempts": 0})
        entry.update(fields)
        entry["status"] = status
        entry["attempts"] += 1
        entry["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2, sort_keys=True), encoding="utf-8")
        # On Windows, antivirus / editor file watchers briefly lock the target and
        # os.replace raises PermissionError; the lock clears within milliseconds.
        for attempt in range(10):
            try:
                os.replace(tmp, self.path)
                return
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(0.1 * (attempt + 1))

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entry in self.state.values():
            counts[entry["status"]] = counts.get(entry["status"], 0) + 1
        return counts
