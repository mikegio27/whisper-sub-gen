"""Path validation for /scan and /process.

Importing app.api constructs the Worker singleton, which opens sqlite in
settings.state_dir, so point that at a temp dir first. The whisper model is
loaded lazily on the first job, so nothing here downloads it.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app.config import settings

_STATE = tempfile.TemporaryDirectory()
settings.state_dir = _STATE.name

from fastapi import HTTPException  # noqa: E402

from app.api import _validate_media_path, healthz  # noqa: E402


def tearDownModule():
    from app.worker import worker

    worker.store._conn.close()
    _STATE.cleanup()


class ValidateMediaPathTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name).resolve()
        self.media = base / "media"
        self.movies = self.media / "movies"
        self.movies.mkdir(parents=True)
        self.video = self.movies / "A.mkv"
        self.video.touch()
        self.outside = base / "media-other"  # shares a string prefix with media
        self.outside.mkdir()
        (self.outside / "B.mkv").touch()
        patcher = mock.patch.object(settings, "media_dirs", f"{self.media}, /nonexistent-root")
        patcher.start()
        self.addCleanup(patcher.stop)

    def assertRejected(self, raw: str, code: int = 400):
        with self.assertRaises(HTTPException) as cm:
            _validate_media_path(raw)
        self.assertEqual(cm.exception.status_code, code)

    def test_file_inside_accepted(self):
        self.assertEqual(_validate_media_path(str(self.video)), self.video)

    def test_root_and_subdir_accepted(self):
        self.assertEqual(_validate_media_path(str(self.media)), self.media)
        self.assertEqual(_validate_media_path(str(self.movies)), self.movies)

    def test_outside_rejected(self):
        self.assertRejected("/etc/passwd")

    def test_prefix_sibling_rejected(self):
        self.assertRejected(str(self.outside / "B.mkv"))

    def test_dotdot_traversal_rejected(self):
        self.assertRejected(f"{self.movies}/../../media-other/B.mkv")

    def test_dotdot_that_stays_inside_accepted(self):
        self.assertEqual(_validate_media_path(f"{self.movies}/../movies/A.mkv"), self.video)

    def test_symlink_escape_rejected(self):
        link = self.movies / "escape"
        os.symlink(self.outside, link)
        self.assertRejected(str(link / "B.mkv"))

    def test_missing_inside_is_404(self):
        self.assertRejected(str(self.movies / "nope.mkv"), code=404)


class HealthzTest(unittest.TestCase):
    def test_ok_until_the_model_is_unusable(self):
        from app.worker import worker

        self.assertEqual(healthz(), {"status": "ok"})
        with mock.patch.object(worker.transcriber, "load_error", "libcublas.so.12 not found"):
            with self.assertRaises(HTTPException) as ctx:
                healthz()
        self.assertEqual(ctx.exception.status_code, 503)


if __name__ == "__main__":
    unittest.main()
