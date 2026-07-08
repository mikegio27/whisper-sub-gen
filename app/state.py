"""SQLite-backed record of what has been processed.

Lets the scanner cheaply skip media it already handled (same path + size +
mtime) without re-probing every file on every scan, and gives the API a
history to report.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path


class StateStore:
    def __init__(self, state_dir: str):
        try:
            Path(state_dir).mkdir(parents=True, exist_ok=True)
            self._db_path = str(Path(state_dir) / "whisper-sub-gen.db")
            self._lock = threading.Lock()
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        except (OSError, sqlite3.OperationalError) as exc:
            raise RuntimeError(
                f"cannot open state db in {state_dir!r}: {exc}. The container "
                f"runs as uid 1000 — make sure the volume mounted at "
                f"{state_dir!r} is writable by that uid (bind mounts often "
                f"default to root-owned)."
            ) from exc
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS files (
                    path        TEXT PRIMARY KEY,
                    size        INTEGER NOT NULL,
                    mtime       REAL NOT NULL,
                    status      TEXT NOT NULL,           -- done | failed | skipped
                    reason      TEXT,                    -- skip reason / error msg
                    language    TEXT,
                    subtitle    TEXT,                    -- output path when done
                    model       TEXT,
                    duration_s  REAL,                    -- media duration
                    elapsed_s   REAL,                    -- processing wall time
                    attempts    INTEGER NOT NULL DEFAULT 0,
                    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
                )
                """
            )
            self._conn.commit()

    def lookup(self, path: str) -> sqlite3.Row | None:
        with self._lock:
            cur = self._conn.execute("SELECT * FROM files WHERE path = ?", (path,))
            return cur.fetchone()

    def should_skip(self, path: str, size: int, mtime: float, max_retries: int) -> bool:
        """True when this exact file version was already handled."""
        row = self.lookup(path)
        if row is None:
            return False
        if row["size"] != size or abs(row["mtime"] - mtime) > 1:
            return False  # file changed since we saw it — reconsider
        if row["status"] in ("done", "skipped"):
            return True
        return row["attempts"] >= max_retries  # failed too many times

    def record(
        self,
        path: str,
        size: int,
        mtime: float,
        status: str,
        *,
        reason: str = "",
        language: str = "",
        subtitle: str = "",
        model: str = "",
        duration_s: float = 0.0,
        elapsed_s: float = 0.0,
        bump_attempts: bool = False,
    ) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO files
                    (path, size, mtime, status, reason, language, subtitle,
                     model, duration_s, elapsed_s, attempts, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                ON CONFLICT(path) DO UPDATE SET
                    size = excluded.size,
                    mtime = excluded.mtime,
                    status = excluded.status,
                    reason = excluded.reason,
                    language = excluded.language,
                    subtitle = excluded.subtitle,
                    model = excluded.model,
                    duration_s = excluded.duration_s,
                    elapsed_s = excluded.elapsed_s,
                    attempts = CASE WHEN ? THEN files.attempts + 1 ELSE files.attempts END,
                    updated_at = datetime('now')
                """,
                (
                    path, size, mtime, status, reason, language, subtitle,
                    model, duration_s, elapsed_s, 1 if bump_attempts else 0,
                    bump_attempts,
                ),
            )
            self._conn.commit()

    def forget(self, path: str) -> bool:
        """Drop a record so the file is re-processed on the next scan."""
        with self._lock:
            cur = self._conn.execute("DELETE FROM files WHERE path = ?", (path,))
            self._conn.commit()
            return cur.rowcount > 0

    def history(self, limit: int = 100, status: str | None = None) -> list[dict]:
        query = "SELECT * FROM files"
        params: tuple = ()
        if status:
            query += " WHERE status = ?"
            params = (status,)
        query += " ORDER BY updated_at DESC LIMIT ?"
        with self._lock:
            cur = self._conn.execute(query, params + (limit,))
            return [dict(r) for r in cur.fetchall()]

    def counts(self) -> dict[str, int]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT status, COUNT(*) AS n FROM files GROUP BY status"
            )
            return {r["status"]: r["n"] for r in cur.fetchall()}
