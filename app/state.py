"""SQLite-backed record of what has been processed.

Lets the scanner cheaply skip media it already handled (same path + size +
mtime) without re-probing every file on every scan, and gives the API a
history to report.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path

_ADDED_COLUMNS = (
    ("pipeline", "INTEGER NOT NULL DEFAULT 0"),  # transcriber.PIPELINE_VERSION
    ("sub_size", "INTEGER"),  # the .srt we wrote, to recognise it later
    ("sub_mtime", "REAL"),
    ("qa_violations", "REAL"),  # qa.score()["violations_per_100"]
    ("qa_json", "TEXT"),  # full qa.score() dict
    ("regen_attempts", "INTEGER NOT NULL DEFAULT 0"),  # failed regenerations since last success
)


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
            # Added with pipeline v1. ALTER in place so an existing prod db
            # keeps its history; old rows read as pipeline 0 (raw whisper).
            have = {r["name"] for r in self._conn.execute("PRAGMA table_info(files)")}
            for name, decl in _ADDED_COLUMNS:
                if name not in have:
                    self._conn.execute(f"ALTER TABLE files ADD COLUMN {name} {decl}")
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
        pipeline: int = 0,
        sub_size: int | None = None,
        sub_mtime: float | None = None,
        qa: dict | None = None,
    ) -> None:
        qa_violations = qa.get("violations_per_100") if qa else None
        qa_json = json.dumps(qa, sort_keys=True) if qa else None
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO files
                    (path, size, mtime, status, reason, language, subtitle,
                     model, duration_s, elapsed_s, attempts, pipeline,
                     sub_size, sub_mtime, qa_violations, qa_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
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
                    pipeline = excluded.pipeline,
                    sub_size = excluded.sub_size,
                    sub_mtime = excluded.sub_mtime,
                    qa_violations = excluded.qa_violations,
                    qa_json = excluded.qa_json,
                    regen_attempts = 0,
                    updated_at = datetime('now')
                """,
                (
                    path,
                    size,
                    mtime,
                    status,
                    reason,
                    language,
                    subtitle,
                    model,
                    duration_s,
                    elapsed_s,
                    1 if bump_attempts else 0,
                    pipeline,
                    sub_size,
                    sub_mtime,
                    qa_violations,
                    qa_json,
                    bump_attempts,
                ),
            )
            self._conn.commit()

    def outdated_output(self, path: str, pipeline: int, max_attempts: int = 0) -> str | None:
        """Our own subtitle for `path` if it was made by an older pipeline and
        is still byte-for-byte what we wrote (same size, mtime within 1 s).
        None when there's nothing we may safely regenerate, or when
        regeneration already failed `max_attempts` times (0 = no limit)."""
        row = self.lookup(path)
        if row is None or row["status"] != "done" or row["pipeline"] >= pipeline:
            return None
        if max_attempts and row["regen_attempts"] >= max_attempts:
            return None
        sub = row["subtitle"]
        if not sub:
            return None
        try:
            st = Path(sub).stat()
        except OSError:
            return None  # deleted or renamed by the owner: leave it alone
        if row["sub_size"] is None:
            # Rows from before v1 have no fingerprint. Fall back to "inode
            # unchanged since we finished": updated_at (UTC) is written right
            # after our rename. ctime, not mtime, because a replacement sub
            # unzipped by hand can carry an old mtime but always gets a fresh
            # ctime. 60 s of slack for NFS server clock skew.
            written = datetime.fromisoformat(row["updated_at"]).replace(tzinfo=UTC)
            if st.st_ctime > written.timestamp() + 60:
                return None
            return sub
        if st.st_size != row["sub_size"] or abs(st.st_mtime - (row["sub_mtime"] or 0)) > 1:
            return None
        return sub

    def record_regen_failure(self, path: str, reason: str) -> None:
        """A regeneration failed. Keep the row as it is (our old sub is still on
        disk and still ours), just count the attempt so it isn't retried forever."""
        with self._lock:
            self._conn.execute(
                "UPDATE files SET regen_attempts = regen_attempts + 1, reason = ?, "
                "updated_at = updated_at WHERE path = ?",
                (reason, path),
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
            cur = self._conn.execute("SELECT status, COUNT(*) AS n FROM files GROUP BY status")
            return {r["status"]: r["n"] for r in cur.fetchall()}
