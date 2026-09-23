"""Never clobber a subtitle we didn't write: output placement and regeneration.

Importing app.worker constructs the Worker singleton (sqlite in
settings.state_dir), so point that at a temp dir first. Each test builds its
own Worker against its own temp state dir. The transcriber is faked: nothing
loads a model or runs ffmpeg.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app.config import settings

_STATE = tempfile.TemporaryDirectory()
settings.state_dir = _STATE.name

from app.transcriber import ModelLoadError, OutputConflict, place_output  # noqa: E402
from app.worker import Job, Worker  # noqa: E402


def tearDownModule():
    from app.worker import worker

    worker.store._conn.close()
    _STATE.cleanup()


class PlaceOutputTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.part = self.dir / "Movie.en.srt.part"
        self.target = self.dir / "Movie.en.srt"
        self.part.write_text("ours\n")

    def test_creates_absent_target(self):
        place_output(self.part, self.target, None)
        self.assertEqual(self.target.read_text(), "ours\n")
        self.assertFalse(self.part.exists())

    def test_refuses_existing_target_without_permission(self):
        self.target.write_text("human\n")
        with self.assertRaises(OutputConflict):
            place_output(self.part, self.target, None)
        with self.assertRaises(OutputConflict):
            place_output(self.part, self.target, lambda _t: False)
        self.assertEqual(self.target.read_text(), "human\n")

    def test_replaces_when_allowed(self):
        self.target.write_text("old ours\n")
        place_output(self.part, self.target, lambda t: t == self.target)
        self.assertEqual(self.target.read_text(), "ours\n")
        self.assertFalse(self.part.exists())

    def test_filesystem_without_hard_links(self):
        self.target.write_text("human\n")
        with (
            mock.patch("app.transcriber.os.link", side_effect=PermissionError),
            self.assertRaises(OutputConflict),
        ):
            place_output(self.part, self.target, None)
        self.target.unlink()
        with mock.patch("app.transcriber.os.link", side_effect=PermissionError):
            place_output(self.part, self.target, None)
        self.assertEqual(self.target.read_text(), "ours\n")


class RegenerationTest(unittest.TestCase):
    """The owner may download a human sub while a (minutes-long) job runs."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        state = self.dir / "state"
        with mock.patch.object(settings, "state_dir", str(state)):
            self.worker = Worker()
        self.addCleanup(self.worker.store._conn.close)
        self.video = self.dir / "Movie.mkv"
        self.video.write_bytes(b"x" * 100)
        self.ours = self.dir / "Movie.en.srt"
        self.ours.write_text("1\n00:00:01,000 --> 00:00:02,000\nold ours\n")
        self.record_ours(pipeline=0)

    def record_ours(self, pipeline):
        st, so = self.video.stat(), self.ours.stat()
        self.worker.store.record(
            str(self.video),
            st.st_size,
            st.st_mtime,
            "done",
            subtitle=str(self.ours),
            pipeline=pipeline,
            sub_size=so.st_size,
            sub_mtime=so.st_mtime,
        )

    def fake_transcribe(self, lang, *, meanwhile=None, fail=False):
        """Mimic Transcriber.transcribe: run `meanwhile` (the owner acting mid-job),
        then write via the real place_output with the given may_replace."""

        def transcribe(video, *, may_replace=None):
            if meanwhile:
                meanwhile()
            if fail:
                raise RuntimeError("no speech detected")
            target = self.dir / f"Movie.{lang}.srt"
            part = self.dir / f"Movie.{lang}.srt.part"
            part.write_text("new ours\n")
            try:
                place_output(part, target, may_replace)
            finally:
                part.unlink(missing_ok=True)
            st = target.stat()
            return {
                "subtitle": str(target),
                "subtitle_size": st.st_size,
                "subtitle_mtime": st.st_mtime,
                "language": lang,
                "segments": 1,
                "words": 2,
                "duration_s": 1.0,
                "elapsed_s": 1.0,
                "pipeline": 1,
                "qa": {"violations_per_100": 0.0},
            }

        return transcribe

    def run_regen(self, transcribe):
        own = self.worker._outdated_own_sub(self.video)
        self.assertEqual(own, str(self.ours))
        job = Job(id=1, path=self.video, source="regen", replaces=own)
        with mock.patch.object(self.worker.transcriber, "transcribe", transcribe):
            self.worker._process(job)

    def human_download(self):
        self.ours.write_text("HUMAN SUBTITLE, longer than ours\n")

    def test_regen_replaces_untouched_own_sub(self):
        self.run_regen(self.fake_transcribe("en"))
        self.assertEqual(self.ours.read_text(), "new ours\n")
        row = self.worker.store.lookup(str(self.video))
        self.assertEqual((row["status"], row["pipeline"]), ("done", 1))

    def test_human_sub_downloaded_mid_job_same_language_survives(self):
        with self.assertLogs("app.worker", level="WARNING"):
            self.run_regen(self.fake_transcribe("en", meanwhile=self.human_download))
        self.assertTrue(self.ours.read_text().startswith("HUMAN"))
        self.assertEqual(self.worker.store.lookup(str(self.video))["status"], "skipped")

    def test_human_sub_downloaded_mid_job_language_change_survives(self):
        self.run_regen(self.fake_transcribe("es", meanwhile=self.human_download))
        self.assertTrue(self.ours.exists())
        self.assertTrue(self.ours.read_text().startswith("HUMAN"))

    def test_language_change_removes_our_untouched_old_sub(self):
        self.run_regen(self.fake_transcribe("es"))
        self.assertFalse(self.ours.exists())
        self.assertTrue((self.dir / "Movie.es.srt").exists())

    def test_plain_job_never_overwrites_a_sub_that_appeared_mid_job(self):
        self.ours.unlink()
        self.worker.store.forget(str(self.video))
        job = Job(id=1, path=self.video, source="scan")
        with (
            mock.patch("app.worker.check_needs_subtitles", return_value=(True, "")),
            mock.patch.object(
                self.worker.transcriber,
                "transcribe",
                self.fake_transcribe("en", meanwhile=self.human_download),
            ),
            self.assertLogs("app.worker", level="WARNING"),
        ):
            self.worker._process(job)
        self.assertTrue(self.ours.read_text().startswith("HUMAN"))

    def test_failed_regen_keeps_row_and_gives_up_after_max_retries(self):
        with (
            mock.patch.object(settings, "max_retries", 2),
            self.assertLogs("app.worker", level="ERROR"),
        ):
            self.run_regen(self.fake_transcribe("en", fail=True))
            row = self.worker.store.lookup(str(self.video))
            self.assertEqual((row["status"], row["pipeline"]), ("done", 0))
            self.assertEqual(row["subtitle"], str(self.ours))
            self.run_regen(self.fake_transcribe("en", fail=True))
            self.assertIsNone(self.worker._outdated_own_sub(self.video))
        self.assertTrue(self.ours.exists())

    def test_other_sub_next_to_ours_blocks_regen(self):
        (self.dir / "Movie.en.hi.srt").write_text("human SDH\n")
        self.assertIsNone(self.worker._outdated_own_sub(self.video))

    def test_force_request_upgrades_a_queued_regen_job(self):
        own = self.worker._outdated_own_sub(self.video)
        regen = self.worker.enqueue(self.video, source="regen", replaces=own)
        forced = self.worker.enqueue(self.video, source="api", bypass_window=True, force=True)
        self.assertIs(forced, regen)
        self.assertTrue(regen.force)
        self.assertIsNone(regen.replaces)
        self.assertIsNone(self.worker.enqueue(self.video, source="scan"))


class ModelGuardTest(unittest.TestCase):
    """A model that can't load/run (CUDA vs host driver) must not be blamed on files."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        with mock.patch.object(settings, "state_dir", str(self.dir / "state")):
            self.worker = Worker()
        self.addCleanup(self.worker.store._conn.close)
        self.video = self.dir / "Movie.mkv"
        self.video.write_bytes(b"x")

    def test_load_failure_sets_load_error_and_records_nothing(self):
        t = self.worker.transcriber
        boom = RuntimeError("CUDA driver version is insufficient for CUDA runtime version")
        with (
            mock.patch.object(t, "_build_model", side_effect=boom),
            mock.patch("app.worker.check_needs_subtitles", return_value=(True, "")),
            self.assertLogs("app", level="ERROR"),
        ):
            t.warm_up()
            self.assertIn("insufficient", t.load_error)
            self.worker._process(Job(id=1, path=self.video, source="scan"))
        self.assertIsNone(self.worker.store.lookup(str(self.video)))

    def test_successful_load_clears_load_error(self):
        t = self.worker.transcriber
        t.load_error = "earlier failure"
        with mock.patch.object(t, "_build_model", return_value=object()):
            t.warm_up()
        self.assertIsNone(t.load_error)

    def test_load_error_is_a_distinct_type(self):
        t = self.worker.transcriber
        with (
            mock.patch.object(t, "_build_model", side_effect=OSError("libcublas.so.12")),
            self.assertLogs("app.transcriber", level="ERROR"),
            self.assertRaises(ModelLoadError),
        ):
            t._load_model()


if __name__ == "__main__":
    unittest.main()
