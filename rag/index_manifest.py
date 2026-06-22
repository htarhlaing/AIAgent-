from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import RLock


@dataclass(frozen=True)
class ManifestEntry:
    path: str
    content_hash: str
    chunk_ids: list[str]
    indexed_at: str


class IndexManifest:
    """SQLite-backed source-of-truth for files currently stored in Chroma."""

    def __init__(self, database_path: str) -> None:
        self.database_path = database_path
        os.makedirs(os.path.dirname(database_path), exist_ok=True)
        self._lock = RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS indexed_documents (
                    path TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL,
                    chunk_ids TEXT NOT NULL,
                    indexed_at TEXT NOT NULL
                )
                """
            )

    @staticmethod
    def _to_entry(row: sqlite3.Row) -> ManifestEntry:
        return ManifestEntry(
            path=row["path"],
            content_hash=row["content_hash"],
            chunk_ids=list(json.loads(row["chunk_ids"])),
            indexed_at=row["indexed_at"],
        )

    def all(self) -> dict[str, ManifestEntry]:
        with self._lock, self._connect() as connection:
            rows = connection.execute("SELECT * FROM indexed_documents").fetchall()
        return {row["path"]: self._to_entry(row) for row in rows}

    def upsert(self, path: str, content_hash: str, chunk_ids: list[str]) -> None:
        indexed_at = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO indexed_documents(path, content_hash, chunk_ids, indexed_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    content_hash=excluded.content_hash,
                    chunk_ids=excluded.chunk_ids,
                    indexed_at=excluded.indexed_at
                """,
                (path, content_hash, json.dumps(chunk_ids), indexed_at),
            )

    def remove(self, path: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("DELETE FROM indexed_documents WHERE path = ?", (path,))
