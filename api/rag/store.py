"""SQLite vector store: chunk text + metadata + float32 embedding BLOBs.
Brute-force cosine in numpy -- exact, dependency-free, plenty for a few thousand chunks."""
from __future__ import annotations
import os
import sqlite3

import numpy as np

from . import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS documents (
    source TEXT PRIMARY KEY, sha256 TEXT NOT NULL, chunk_count INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY, source TEXT NOT NULL, heading TEXT NOT NULL, page INTEGER,
    chunk_index INTEGER NOT NULL, text TEXT NOT NULL, embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source);
"""


class Store:
    def __init__(self, path: str | None = None):
        path = path or config.DB_PATH
        if path != ":memory:":
            os.makedirs(os.path.dirname(path), exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.executescript(_SCHEMA)

    def close(self) -> None:
        self.conn.close()

    # -- meta ---------------------------------------------------------------
    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self.conn:
            self.conn.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, value))

    # -- documents ------------------------------------------------------------
    def doc_hashes(self) -> dict[str, str]:
        return dict(self.conn.execute("SELECT source, sha256 FROM documents"))

    def replace_document(self, source: str, sha: str, chunks, vectors) -> None:
        """Atomically swap one document's chunks (delete old, insert new)."""
        with self.conn:
            self.conn.execute("DELETE FROM chunks WHERE source=?", (source,))
            self.conn.executemany(
                "INSERT INTO chunks VALUES (?,?,?,?,?,?,?)",
                [(c.id, c.source, c.heading, c.page, c.index, c.text,
                  np.asarray(v, dtype=np.float32).tobytes())
                 for c, v in zip(chunks, vectors)],
            )
            self.conn.execute("INSERT OR REPLACE INTO documents VALUES (?,?,?)", (source, sha, len(chunks)))

    def delete_document(self, source: str) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM chunks WHERE source=?", (source,))
            self.conn.execute("DELETE FROM documents WHERE source=?", (source,))

    def clear(self) -> None:
        with self.conn:
            for t in ("chunks", "documents", "meta"):
                self.conn.execute(f"DELETE FROM {t}")

    # -- reads ----------------------------------------------------------------
    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def sources(self) -> list[tuple[str, int]]:
        return list(self.conn.execute("SELECT source, chunk_count FROM documents ORDER BY source"))

    def all_chunks(self):
        """(rows, matrix) -- rows are (id, source, heading, page, chunk_index, text)."""
        rows = self.conn.execute(
            "SELECT id, source, heading, page, chunk_index, text, embedding FROM chunks ORDER BY source, chunk_index"
        ).fetchall()
        if not rows:
            return [], np.zeros((0, 0), dtype=np.float32)
        mat = np.vstack([np.frombuffer(r[6], dtype=np.float32) for r in rows])
        return [r[:6] for r in rows], mat
