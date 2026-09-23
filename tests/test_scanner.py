import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app import scanner
from app.config import settings
from app.scanner import check_needs_subtitles, external_subtitles, output_path
from app.transcriber import _extract_audio, _format_ts


class FormatTsTest(unittest.TestCase):
    def test_zero(self):
        self.assertEqual(_format_ts(0), "00:00:00,000")

    def test_components(self):
        self.assertEqual(_format_ts(3661.234), "01:01:01,234")

    def test_rounds_to_nearest_ms(self):
        self.assertEqual(_format_ts(1.0006), "00:00:01,001")
        self.assertEqual(_format_ts(59.9999), "00:01:00,000")

    def test_negative_clamped(self):
        self.assertEqual(_format_ts(-0.5), "00:00:00,000")

    def test_hours_beyond_99(self):
        self.assertEqual(_format_ts(100 * 3600), "100:00:00,000")


class ExtractAudioTest(unittest.TestCase):
    def test_timeout_cleans_up_and_raises(self):
        with tempfile.TemporaryDirectory() as d:
            timeout = subprocess.TimeoutExpired(cmd="ffmpeg", timeout=1)
            with mock.patch("app.transcriber.subprocess.run", side_effect=timeout):
                with self.assertRaisesRegex(RuntimeError, "timed out"):
                    _extract_audio(Path(d) / "movie.mkv", Path(d) / "tmp")
            self.assertEqual(list((Path(d) / "tmp").iterdir()), [])


class SidecarTest(unittest.TestCase):
    """check_needs_subtitles against real sidecar files in a temp dir."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.video = self.dir / "Movie (2020).mkv"
        self.video.touch()
        # Defaults, with embedded probing off so no ffprobe is needed.
        for name, value in {
            "language": "",
            "subtitle_tag": "",
            "overwrite_existing_output": False,
            "skip_if_external_subs": True,
            "skip_if_embedded_subs": False,
            "embedded_text_subs_only": False,
        }.items():
            patcher = mock.patch.object(settings, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def touch(self, name: str) -> Path:
        p = self.dir / name
        p.touch()
        return p

    def test_no_subs_needs_generation(self):
        self.assertEqual(check_needs_subtitles(self.video), (True, ""))

    def test_existing_output_blocks(self):
        self.touch("Movie (2020).en.srt")
        needs, reason = check_needs_subtitles(self.video)
        self.assertFalse(needs)
        self.assertIn("output already exists", reason)

    def test_other_sidecar_formats_block(self):
        self.touch("Movie (2020).eng.forced.ass")
        needs, reason = check_needs_subtitles(self.video)
        self.assertFalse(needs)
        self.assertIn("external subtitles present", reason)

    def test_other_video_sidecars_ignored(self):
        self.touch("Other Movie.en.srt")
        self.touch("Movie (2020).nfo")
        self.assertEqual(check_needs_subtitles(self.video), (True, ""))

    def test_external_subs_allowed_when_disabled(self):
        self.touch("Movie (2020).fr.vtt")
        with mock.patch.object(settings, "skip_if_external_subs", False):
            self.assertEqual(check_needs_subtitles(self.video), (True, ""))

    def test_forced_language_only_matches_that_language(self):
        self.touch("Movie (2020).fr.srt")
        with (
            mock.patch.object(settings, "language", "en"),
            mock.patch.object(settings, "skip_if_external_subs", False),
        ):
            self.assertEqual(check_needs_subtitles(self.video), (True, ""))
            self.touch("Movie (2020).en.srt")
            self.assertFalse(check_needs_subtitles(self.video)[0])

    def test_overwrite_ignores_own_output(self):
        self.touch("Movie (2020).en.srt")
        with (
            mock.patch.object(settings, "overwrite_existing_output", True),
            mock.patch.object(settings, "skip_if_external_subs", False),
        ):
            self.assertEqual(check_needs_subtitles(self.video), (True, ""))

    def test_subtitle_tag_in_output_name(self):
        with mock.patch.object(settings, "subtitle_tag", ".whisper"):
            self.assertEqual(output_path(self.video, "en").name, "Movie (2020).whisper.en.srt")

    def test_external_subtitles_lists_matching(self):
        a = self.touch("Movie (2020).en.srt")
        b = self.touch("Movie (2020).SDH.SUB")
        self.touch("Movie (2020)-trailer.mkv")
        self.assertEqual(sorted(external_subtitles(self.video)), sorted([a, b]))

    def test_embedded_subs_block(self):
        probe = {
            "streams": [
                {"codec_type": "audio"},
                {"codec_type": "subtitle", "codec_name": "subrip", "tags": {"language": "eng"}},
            ]
        }
        with (
            mock.patch.object(settings, "skip_if_embedded_subs", True),
            mock.patch.object(scanner, "ffprobe", return_value=probe),
        ):
            needs, reason = check_needs_subtitles(self.video)
        self.assertFalse(needs)
        self.assertIn("embedded subtitles present (eng)", reason)

    def test_bitmap_subs_ignored_in_text_only_mode(self):
        probe = {"streams": [{"codec_type": "subtitle", "codec_name": "hdmv_pgs_subtitle"}]}
        with (
            mock.patch.object(settings, "skip_if_embedded_subs", True),
            mock.patch.object(settings, "embedded_text_subs_only", True),
            mock.patch.object(scanner, "ffprobe", return_value=probe),
        ):
            self.assertEqual(check_needs_subtitles(self.video), (True, ""))

    def test_ffprobe_failure_falls_through_to_generation(self):
        with (
            mock.patch.object(settings, "skip_if_embedded_subs", True),
            mock.patch.object(scanner, "ffprobe", side_effect=RuntimeError("boom")),
            self.assertLogs("app.scanner", level="WARNING"),
        ):
            self.assertEqual(check_needs_subtitles(self.video), (True, ""))


if __name__ == "__main__":
    unittest.main()
