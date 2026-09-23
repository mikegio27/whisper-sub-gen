import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

from app.state import StateStore


class StateStoreTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = StateStore(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.addCleanup(self.store._conn.close)

    def test_db_created_in_state_dir(self):
        self.assertTrue((Path(self._tmp.name) / "whisper-sub-gen.db").exists())

    def test_unknown_file_not_skipped(self):
        self.assertFalse(self.store.should_skip("/m/a.mkv", 100, 1000.0, max_retries=2))

    def test_done_and_skipped_are_skipped(self):
        self.store.record("/m/a.mkv", 100, 1000.0, "done")
        self.store.record("/m/b.mkv", 100, 1000.0, "skipped", reason="subs present")
        self.assertTrue(self.store.should_skip("/m/a.mkv", 100, 1000.0, max_retries=2))
        self.assertTrue(self.store.should_skip("/m/b.mkv", 100, 1000.0, max_retries=2))

    def test_changed_file_reconsidered(self):
        self.store.record("/m/a.mkv", 100, 1000.0, "done")
        self.assertFalse(self.store.should_skip("/m/a.mkv", 101, 1000.0, max_retries=2))
        self.assertFalse(self.store.should_skip("/m/a.mkv", 100, 1002.0, max_retries=2))

    def test_mtime_tolerance_of_one_second(self):
        # NFS/filesystems can round mtimes; within 1s counts as the same file.
        self.store.record("/m/a.mkv", 100, 1000.0, "done")
        self.assertTrue(self.store.should_skip("/m/a.mkv", 100, 1000.9, max_retries=2))

    def test_failed_retried_until_max_retries(self):
        path = "/m/a.mkv"
        self.store.record(path, 100, 1000.0, "failed", bump_attempts=True)
        self.assertFalse(self.store.should_skip(path, 100, 1000.0, max_retries=2))
        self.store.record(path, 100, 1000.0, "failed", bump_attempts=True)
        self.assertEqual(self.store.lookup(path)["attempts"], 2)
        self.assertTrue(self.store.should_skip(path, 100, 1000.0, max_retries=2))

    def test_record_without_bump_keeps_attempts(self):
        path = "/m/a.mkv"
        self.store.record(path, 100, 1000.0, "failed", bump_attempts=True)
        self.store.record(path, 100, 1000.0, "done")
        self.assertEqual(self.store.lookup(path)["attempts"], 1)

    def test_forget(self):
        self.store.record("/m/a.mkv", 100, 1000.0, "done")
        self.assertTrue(self.store.forget("/m/a.mkv"))
        self.assertFalse(self.store.forget("/m/a.mkv"))
        self.assertFalse(self.store.should_skip("/m/a.mkv", 100, 1000.0, max_retries=2))

    def test_history_and_counts(self):
        self.store.record("/m/a.mkv", 1, 1.0, "done")
        self.store.record("/m/b.mkv", 1, 1.0, "failed", bump_attempts=True)
        self.store.record("/m/c.mkv", 1, 1.0, "done")
        self.assertEqual(self.store.counts(), {"done": 2, "failed": 1})
        self.assertEqual(len(self.store.history()), 3)
        failed = self.store.history(status="failed")
        self.assertEqual([r["path"] for r in failed], ["/m/b.mkv"])
        self.assertEqual(len(self.store.history(limit=1)), 1)


class OutdatedOutputTest(unittest.TestCase):
    """outdated_output must only ever hand back a sub we wrote and nobody touched."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.store = StateStore(str(self.dir))
        self.addCleanup(self.store._conn.close)
        self.sub = self.dir / "Movie.en.srt"
        self.sub.write_text("1\n00:00:01,000 --> 00:00:02,000\nHi\n")

    def record_v1(self, pipeline=1):
        st = self.sub.stat()
        self.store.record(
            "/m/Movie.mkv",
            100,
            1000.0,
            "done",
            subtitle=str(self.sub),
            pipeline=pipeline,
            sub_size=st.st_size,
            sub_mtime=st.st_mtime,
            qa={"violations_per_100": 3.5},
        )

    def test_current_pipeline_is_not_outdated(self):
        self.record_v1(pipeline=2)
        self.assertIsNone(self.store.outdated_output("/m/Movie.mkv", 2))

    def test_untouched_old_output_is_returned(self):
        self.record_v1(pipeline=1)
        self.assertEqual(self.store.outdated_output("/m/Movie.mkv", 2), str(self.sub))
        self.assertEqual(self.store.lookup("/m/Movie.mkv")["qa_violations"], 3.5)

    def test_edited_output_is_left_alone(self):
        self.record_v1(pipeline=1)
        self.sub.write_text("hand-fixed and longer than before\n")
        self.assertIsNone(self.store.outdated_output("/m/Movie.mkv", 2))

    def test_same_size_replacement_with_old_mtime_is_left_alone(self):
        self.record_v1(pipeline=1)
        st = self.sub.stat()
        os.utime(self.sub, (st.st_atime, st.st_mtime - 3600))
        self.assertIsNone(self.store.outdated_output("/m/Movie.mkv", 2))

    def test_deleted_output_is_not_returned(self):
        self.record_v1(pipeline=1)
        self.sub.unlink()
        self.assertIsNone(self.store.outdated_output("/m/Movie.mkv", 2))

    def test_failed_and_skipped_rows_are_ignored(self):
        self.store.record("/m/Movie.mkv", 100, 1000.0, "skipped", subtitle=str(self.sub))
        self.assertIsNone(self.store.outdated_output("/m/Movie.mkv", 2))

    def test_legacy_row_uses_inode_change_time(self):
        # Pre-v1 rows have no fingerprint. Recorded "now" (after the sub was
        # written) => ours; a sub replaced later has a newer ctime => not ours.
        self.store.record("/m/Movie.mkv", 100, 1000.0, "done", subtitle=str(self.sub))
        self.assertEqual(self.store.outdated_output("/m/Movie.mkv", 1), str(self.sub))
        with self.store._lock:
            self.store._conn.execute(
                "UPDATE files SET updated_at = datetime('now', '-1 day') WHERE path = ?",
                ("/m/Movie.mkv",),
            )
        self.assertIsNone(self.store.outdated_output("/m/Movie.mkv", 1))


class MigrationTest(unittest.TestCase):
    def test_pre_v1_db_gains_columns_and_keeps_rows(self):
        with tempfile.TemporaryDirectory() as d:
            conn = sqlite3.connect(Path(d) / "whisper-sub-gen.db")
            conn.execute(
                """CREATE TABLE files (
                    path TEXT PRIMARY KEY, size INTEGER NOT NULL, mtime REAL NOT NULL,
                    status TEXT NOT NULL, reason TEXT, language TEXT, subtitle TEXT,
                    model TEXT, duration_s REAL, elapsed_s REAL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL DEFAULT (datetime('now')))"""
            )
            conn.execute(
                "INSERT INTO files (path, size, mtime, status) VALUES ('/a', 1, 1, 'done')"
            )
            conn.commit()
            conn.close()
            store = StateStore(d)
            self.addCleanup(store._conn.close)
            row = store.lookup("/a")
            self.assertEqual(row["pipeline"], 0)
            self.assertIsNone(row["qa_json"])
            # Re-opening an already migrated db is a no-op.
            StateStore(d)._conn.close()


if __name__ == "__main__":
    unittest.main()
