"""Parent nodes (the LLM's context units) in a local SQLite key-value store.

Parents are only ever fetched by ID after a child is retrieved, never searched,
so they don't belong in the vector database. SQLite is a single file on D:,
needs no server, and reads a handful of parents in well under a millisecond.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from llama_index.core.schema import TextNode

from src.config import DATA_DIR

DEFAULT_PATH = DATA_DIR / "index" / "parents.sqlite"


class ParentStore:
    def __init__(self, path: Path = DEFAULT_PATH, readonly: bool | None = None):
        # The API only reads parents. In Docker the file sits on a Windows bind mount, where SQLite's
        # WAL shared memory is unreliable, so it is opened read-only + immutable (no locks, no -shm).
        # Immutable mode ignores an un-checkpointed -wal file: run checkpoint() after indexing.
        readonly = os.environ.get("PARENT_STORE_READONLY") == "1" if readonly is None else readonly
        if readonly:
            self.conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro&immutable=1", uri=True)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS parents (node_id TEXT PRIMARY KEY, filing_id TEXT NOT NULL, node TEXT NOT NULL)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS parents_filing ON parents(filing_id)")

    def checkpoint(self) -> None:
        """Fold the WAL into the main file so read-only/immutable readers see every write."""
        self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def replace_filing(self, filing_id: str, nodes: list[TextNode]) -> None:
        """Idempotent: re-indexing a filing replaces its parents instead of duplicating them."""
        with self.conn:
            self.conn.execute("DELETE FROM parents WHERE filing_id = ?", (filing_id,))
            self.conn.executemany("INSERT INTO parents VALUES (?, ?, ?)",
                                  [(n.node_id, filing_id, n.to_json()) for n in nodes])

    def get(self, node_ids: list[str]) -> dict[str, TextNode]:
        if not node_ids:
            return {}
        marks = ",".join("?" * len(node_ids))
        rows = self.conn.execute(f"SELECT node_id, node FROM parents WHERE node_id IN ({marks})", node_ids)
        return {node_id: TextNode.from_json(node) for node_id, node in rows}

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM parents").fetchone()[0]
