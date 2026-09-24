import subprocess
import unittest
from pathlib import Path
from unittest import mock

from app import audio
from app.audio import AudioTrack, ffmpeg_cmd, pick_track


def stream(channels=6, layout="5.1(side)", lang="eng", title="", default=0, comment=0):
    return {
        "codec_type": "audio",
        "channels": channels,
        "channel_layout": layout,
        "tags": {"language": lang, "title": title},
        "disposition": {"default": default, "comment": comment},
    }


class PickTrackTest(unittest.TestCase):
    def test_no_audio(self):
        self.assertIsNone(pick_track({"streams": [{"codec_type": "video"}]}))

    def test_commentary_is_avoided_even_when_default(self):
        probe = {
            "streams": [
                {"codec_type": "video"},
                stream(title="Director's Commentary", default=1),
                stream(),
            ]
        }
        self.assertEqual(pick_track(probe).index, 1)

    def test_comment_disposition_is_avoided(self):
        probe = {"streams": [stream(comment=1), stream(channels=2, layout="stereo")]}
        self.assertEqual(pick_track(probe).index, 1)

    def test_requested_language_wins_over_default(self):
        probe = {"streams": [stream(lang="fre", default=1), stream(lang="eng")]}
        self.assertEqual(pick_track(probe, "en").index, 1)

    def test_default_disposition_then_first(self):
        probe = {"streams": [stream(), stream(default=1)]}
        self.assertEqual(pick_track(probe).index, 1)
        probe = {"streams": [stream(), stream()]}
        self.assertEqual(pick_track(probe).index, 0)


class CenterChannelTest(unittest.TestCase):
    def test_surround_layouts_have_center(self):
        for layout, ch in (("5.1", 6), ("5.1(side)", 6), ("7.1", 8), ("6.1", 7)):
            self.assertTrue(AudioTrack(0, ch, layout, "", "").has_center, layout)

    def test_stereo_mono_unknown_do_not(self):
        for layout, ch in (("stereo", 2), ("mono", 1), ("", 6), ("unknown", 6)):
            self.assertFalse(AudioTrack(0, ch, layout, "", "").has_center, layout)

    def test_cmd_keeps_only_fc_for_surround(self):
        track = AudioTrack(1, 6, "5.1", "eng", "")
        cmd = ffmpeg_cmd(Path("m.mkv"), track, center_only=True)
        self.assertIn("pan=mono|c0=FC", cmd)
        self.assertEqual(cmd[cmd.index("-map") + 1], "0:a:1")

    def test_cmd_downmixes_stereo_or_when_disabled(self):
        stereo = AudioTrack(0, 2, "stereo", "", "")
        self.assertNotIn("-af", ffmpeg_cmd(Path("m.mkv"), stereo, center_only=True))
        surround = AudioTrack(0, 6, "5.1", "", "")
        self.assertNotIn("-af", ffmpeg_cmd(Path("m.mkv"), surround, center_only=False))


class DecodeTest(unittest.TestCase):
    def test_timeout_raises_runtime_error(self):
        timeout = subprocess.TimeoutExpired(cmd="ffmpeg", timeout=1)
        with (
            mock.patch("app.audio.subprocess.run", side_effect=timeout),
            self.assertRaisesRegex(RuntimeError, "timed out"),
        ):
            audio._run_ffmpeg(["ffmpeg"])

    def test_silent_center_falls_back_to_downmix(self):
        import numpy as np

        silent = np.zeros(16_000, dtype=np.float32)
        loud = np.full(16_000, 0.1, dtype=np.float32)
        probe = {"streams": [stream()]}
        with (
            mock.patch.object(audio, "_run_ffmpeg", side_effect=[silent, loud]) as run,
            self.assertLogs("app.audio", level="WARNING"),
        ):
            out = audio.load_audio(Path("m.mkv"), probe)
        self.assertIs(out, loud)
        self.assertIn("pan=mono|c0=FC", run.call_args_list[0].args[0])
        self.assertNotIn("-af", run.call_args_list[1].args[0])


class ChunkBoundsTest(unittest.TestCase):
    def test_short_input_is_one_chunk(self):
        import numpy as np

        a = np.ones(16_000 * 100, dtype=np.float32)
        self.assertEqual(audio.chunk_bounds(a, chunk_s=100), [(0, len(a))])
        self.assertEqual(audio.chunk_bounds(a, chunk_s=0), [(0, len(a))])

    def test_cuts_at_the_quiet_spot_and_covers_everything(self):
        import numpy as np

        sr = 16_000
        a = np.full(sr * 300, 0.5, dtype=np.float32)  # 5 min of "speech"
        a[sr * 110 : sr * 111] = 0.0  # 1 s of silence 10 s after the 100 s target
        a[sr * 215 : sr * 216] = 0.0
        b = audio.chunk_bounds(a, chunk_s=100, search_s=30)
        self.assertEqual(b[0][0], 0)
        self.assertEqual(b[-1][1], len(a))
        for (_, e), (s, _) in zip(b, b[1:], strict=False):
            self.assertEqual(e, s)  # contiguous, no gaps or overlap
        self.assertTrue(sr * 110 <= b[0][1] <= sr * 111, b)
        self.assertTrue(sr * 215 <= b[1][1] <= sr * 216, b)


class DecodeCheckTest(unittest.TestCase):
    def test_stream_duration_from_field_or_mkv_tag(self):
        self.assertEqual(audio._stream_duration({"duration": "7035.754667"}), 7035.754667)
        self.assertAlmostEqual(
            audio._stream_duration({"duration": "N/A", "tags": {"DURATION": "02:07:25.696000000"}}),
            7645.696,
        )
        self.assertIsNone(audio._stream_duration({}))
        self.assertEqual(pick_track({"streams": [stream()]}).duration, None)

    def test_short_decode_is_retried(self):
        import numpy as np

        sr = audio.SAMPLE_RATE
        short = np.zeros(sr * 9, dtype=np.float32)  # 1 s missing
        full = np.zeros(sr * 10, dtype=np.float32)
        with (
            mock.patch.object(audio, "_run_ffmpeg", side_effect=[short, full]) as run,
            self.assertLogs("app.audio", level="WARNING"),
        ):
            out = audio._decode_checked(Path("m.mkv"), ["ffmpeg"], expected=10.0)
        self.assertIs(out, full)
        self.assertEqual(run.call_count, 2)

    def test_healthy_or_unknown_duration_decodes_once(self):
        import numpy as np

        full = np.zeros(audio.SAMPLE_RATE * 10, dtype=np.float32)
        for expected in (10.0, 10.2, None):
            with mock.patch.object(audio, "_run_ffmpeg", return_value=full) as run:
                audio._decode_checked(Path("m.mkv"), ["ffmpeg"], expected=expected)
            self.assertEqual(run.call_count, 1, expected)


if __name__ == "__main__":
    unittest.main()
