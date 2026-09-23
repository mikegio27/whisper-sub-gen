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


if __name__ == "__main__":
    unittest.main()
