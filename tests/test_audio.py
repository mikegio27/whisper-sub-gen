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


if __name__ == "__main__":
    unittest.main()
