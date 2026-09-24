import random
import threading
import unittest
from pathlib import Path
from unittest import mock

from app import shots
from app.shots import (
    ShotJob,
    build_command,
    filter_cuts,
    parse_scdet,
    pick_decoder,
    probe_fps,
    snap,
)
from app.srt import Cue
from app.standards import DEFAULT_RULES

R = DEFAULT_RULES
FPS = 24000 / 1001
FRAME = 1 / FPS
TEXT = "Short line."  # 11 chars: cps never matters unless a test wants it to


def probe(codec="h264", fps="24000/1001", start="0.000000", extra_streams=()):
    return {
        "streams": [
            *extra_streams,
            {
                "index": len(extra_streams),
                "codec_type": "video",
                "codec_name": codec,
                "width": 1920,
                "height": 800,
                "avg_frame_rate": fps,
            },
            {"index": len(extra_streams) + 1, "codec_type": "audio", "codec_name": "aac"},
        ],
        "format": {"start_time": start},
    }


class ProbeTests(unittest.TestCase):
    def test_fps(self):
        self.assertAlmostEqual(probe_fps(probe()), 23.976, places=3)
        self.assertEqual(probe_fps(probe(fps="25/1")), 25.0)
        self.assertEqual(probe_fps(probe(fps="0/0")), 24.0)
        self.assertEqual(probe_fps({}), 24.0)

    def test_cover_art_is_skipped(self):
        art = {"index": 0, "codec_type": "video", "codec_name": "mjpeg", "avg_frame_rate": "0/0",
               "disposition": {"attached_pic": 1}}  # fmt: skip
        p = probe(fps="25/1", extra_streams=[art])
        self.assertEqual(probe_fps(p), 25.0)
        cmd = build_command(Path("/m/x.mkv"), p, decoder="cpu", threshold=5)
        self.assertEqual(cmd[cmd.index("-map") + 1], "0:1")

    def test_pick_decoder(self):
        self.assertEqual(pick_decoder(probe(), "auto", "cuda"), "cuda")
        self.assertEqual(pick_decoder(probe(), "auto", "cpu"), "cpu")
        self.assertEqual(pick_decoder(probe(), "cpu", "cuda"), "cpu")
        self.assertEqual(pick_decoder(probe(), "cuda", "cpu"), "cuda")
        self.assertEqual(pick_decoder(probe(codec="prores"), "cuda", "cuda"), "cpu")

    def test_commands(self):
        cuda = build_command(Path("/m/x.mkv"), probe("hevc"), decoder="cuda", threshold=5)
        self.assertEqual(cuda[cuda.index("-hwaccel") + 1], "cuda")
        self.assertIn("scale_cuda=320:-2", cuda[cuda.index("-vf") + 1])
        cpu = build_command(Path("/m/x.mkv"), probe(), decoder="cpu", threshold=5, threads=6)
        self.assertEqual(cpu[cpu.index("-threads") + 1], "6")
        self.assertIn("scdet=threshold=5", cpu[cpu.index("-vf") + 1])


class DetectParseTests(unittest.TestCase):
    LOG = (
        "[Parsed_scdet_1 @ 0x7f] lavfi.scd.score: 15.449, lavfi.scd.time: 91.306\n"
        "frame= 100 fps=0.0\n"
        "[scdet @ 0x7f] lavfi.scd.score: 5.100, lavfi.scd.time: 12.5\n"
    )

    def test_parse(self):
        self.assertEqual(parse_scdet(self.LOG), [(12.5, 5.1), (91.306, 15.449)])

    def test_isolated_candidates_are_cuts(self):
        self.assertEqual(filter_cuts([(1.0, 5.5), (3.0, 30.0)]), [1.0, 3.0])

    def test_motion_cluster_dropped(self):
        # Spinning camera: many similar scores a few frames apart.
        cands = [(10.0, 9.8), (10.04, 6.7), (10.2, 6.2), (10.5, 7.0), (20.0, 12.0)]
        self.assertEqual(filter_cuts(cands), [20.0])

    def test_cut_with_echo_frame_kept(self):
        self.assertEqual(filter_cuts([(2.84, 15.35), (2.88, 5.38)]), [2.84])

    def test_detect_cuts_removes_start_time_and_falls_back(self):
        calls = []

        def fake_run(cmd, timeout, stop):
            calls.append(cmd)
            if "-hwaccel" in cmd:
                raise shots.ShotDetectionError("ffmpeg exit 1: Cannot load libnvcuvid.so.1")
            return self.LOG

        stats = {}
        with mock.patch.object(shots, "_run", fake_run):
            cuts = shots.detect_cuts(
                Path("/m/x.mp4"), probe(start="0.5"), whisper_device="cuda", stats=stats
            )
        self.assertEqual(cuts, [12.0, 90.806])
        self.assertEqual(len(calls), 2)
        self.assertEqual(stats["decoder"], "cpu")
        self.assertIn("libnvcuvid", stats["cuda_error"])


def cue(s, e, text=TEXT):
    return Cue(s, e, text)


class SnapTests(unittest.TestCase):
    def test_start_after_cut_moves_back_to_cut(self):
        out, st = snap([cue(10.3, 12.0)], [10.0], R, FPS)
        self.assertEqual(out[0].start, 10.0)
        self.assertEqual(st["starts_snapped"], 1)

    def test_start_just_before_cut_moves_forward(self):
        out, _ = snap([cue(9.9, 12.0)], [10.0], R, FPS)
        self.assertEqual(out[0].start, 10.0)

    def test_start_far_from_cut_untouched(self):
        # 0.6 s after (past 12 frames) and 0.3 s before (more than a few frames).
        for s in (10.6, 9.7):
            out, st = snap([cue(s, 12.0)], [10.0], R, FPS)
            self.assertEqual(out[0].start, s)
            self.assertEqual(st["starts_snapped"], 0)

    def test_start_rounds_up_to_the_cut_frame(self):
        out, _ = snap([cue(10.2, 12.0)], [10.0104], R, FPS)
        self.assertEqual(out[0].start, 10.011)

    def test_end_extended_to_two_frames_before_cut(self):
        out, st = snap([cue(8.0, 9.7)], [10.0], R, FPS)
        self.assertEqual(out[0].end, 10.0 - 0.083)
        self.assertEqual(st["ends_snapped"], 1)

    def test_end_trimmed_back_before_cut(self):
        out, _ = snap([cue(8.0, 10.3)], [10.0], R, FPS)
        self.assertEqual(out[0].end, 9.917)

    def test_chained_pair_across_a_cut(self):
        out, st = snap([cue(8.0, 9.8), cue(10.2, 12.0)], [10.0], R, FPS)
        self.assertEqual([(c.start, c.end) for c in out], [(8.0, 9.917), (10.0, 12.0)])
        self.assertEqual(st, {"starts_snapped": 1, "ends_snapped": 1, "skipped": 0})

    def test_no_overlap_or_short_gap(self):
        # The next cue starts 50 ms before the cut and can't snap to it (cps would pass 20),
        # so extending this end to cut - 2 frames would leave a 33 ms gap: skipped.
        out, st = snap([cue(8.0, 9.7), cue(9.95, 16.9, "x" * 139)], [10.0], R, FPS)
        self.assertEqual((out[0].end, out[1].start), (9.7, 9.95))
        self.assertEqual(st["skipped"], 2)

    def test_skips_below_min_duration(self):
        out, st = snap([cue(9.6, 10.45)], [10.5], R, FPS)  # trim end -> 0.817 s < 5/6 s
        # The start snap isn't in range; the end target (10.417) would give 0.817 s.
        self.assertEqual(out[0].end, 10.45)
        self.assertEqual(st["skipped"], 1)

    def test_skips_above_max_duration(self):
        out, st = snap([cue(3.0, 9.9)], [10.3], R, FPS)  # extend to 10.217 -> 7.2 s
        self.assertEqual(out[0].end, 9.9)
        self.assertEqual(st["skipped"], 1)

    def test_skips_when_cps_would_exceed_max(self):
        text = "x" * 30  # 30 chars over 1.6 s = 18.75 cps; trimmed to 1.4 s = 21.4 cps
        out, _ = snap([cue(8.5, 10.1, text)], [10.0], R, FPS)
        self.assertEqual(out[0].end, 10.1)

    def test_no_cuts_is_identity(self):
        cues = [cue(1.0, 2.5), cue(3.0, 4.2)]
        out, st = snap(cues, [], R, FPS)
        self.assertEqual(out, cues)
        self.assertEqual(st["starts_snapped"] + st["ends_snapped"] + st["skipped"], 0)

    def test_fps_default(self):
        out, _ = snap([cue(8.0, 9.7)], [10.0], R, 0)
        self.assertEqual(out[0].end, 10.0 - 0.083)

    def test_random_never_breaks_rules(self):
        rnd = random.Random(7)
        for _ in range(200):
            t, cues = 0.0, []
            for _ in range(30):
                t += rnd.choice([0.083, 0.1, 0.3, 0.7, 2.0])
                d = rnd.uniform(R.min_duration, R.max_duration)
                s, e = round(t, 3), round(t + d, 3)
                cues.append(cue(s, e, "y" * rnd.randint(2, 40)))
                t = e
            cuts = sorted(rnd.uniform(0, t) for _ in range(40))
            out, _ = snap(cues, cuts, R, FPS)
            self.assertEqual(len(out), len(cues))
            for a, b in zip(cues, out, strict=True):
                self.assertEqual(a.text, b.text)
                self.assertLessEqual(a.start - b.start, 0.5 + 1e-6)  # never > 0.5 s early
                self.assertLessEqual(b.start - a.start, 4 * FRAME + 1e-3)
                self.assertGreaterEqual(b.duration, R.min_duration - 1e-9)
                self.assertLessEqual(b.duration, R.max_duration + 1e-9)
                if b.cps > R.max_cps:
                    self.assertLessEqual(b.cps, a.cps + 1e-9)
            for x, y in zip(out, out[1:], strict=False):
                self.assertGreaterEqual(round((y.start - x.end) * 1000), round(R.min_gap * 1000))


class ShotJobTests(unittest.TestCase):
    def test_fails_open(self):
        with mock.patch.object(shots, "detect_cuts", side_effect=RuntimeError("boom")):
            job = ShotJob(Path("/m/x.mp4"), {})
            cues = [cue(1.0, 2.5)]
            out, st = job.snap(cues)
        self.assertIs(out, cues)
        self.assertIn("boom", st["error"])

    def test_snaps_with_detected_cuts(self):
        with mock.patch.object(shots, "detect_cuts", return_value=[10.0]):
            job = ShotJob(Path("/m/x.mp4"), {})
            out, st = job.snap([cue(10.3, 12.0)], R, FPS)
        self.assertEqual(out[0].start, 10.0)
        self.assertEqual(st["cuts"], 1)
        self.assertEqual(st["starts_snapped"], 1)

    def test_close_stops_detection(self):
        started = threading.Event()

        def slow(video, probe, *, stop, stats, **kw):
            started.set()
            stop.wait(5)
            raise shots.ShotDetectionError("cancelled")

        with mock.patch.object(shots, "detect_cuts", slow):
            job = ShotJob(Path("/m/x.mp4"), {})
            started.wait(1)
            job.close()
            out, st = job.snap([])
        self.assertIn("cancelled", st["error"])


if __name__ == "__main__":
    unittest.main()
