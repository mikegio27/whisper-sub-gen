"""Shot-change detection (ffmpeg `scdet`) and snapping cue times to the cuts.

Professional subtitles are timed to shot changes, because a cue that appears or vanishes a
few frames away from a cut makes the picture and the text flicker separately. The rules
follow the Netflix Timed Text Style Guide, "Subtitle Timing Guidelines"
(https://partnerhelp.netflixstudios.com/hc/en-us/articles/360051554394):

- In-times: "Where dialogue starts on the shot change or within half a second past the shot
  change, please set the in-time to the first frame of the shot change." Dialogue starting
  just before a cut is moved "up to the first frame of the new shot" (we only do this for a
  few frames; pulling it back to half a second before the cut isn't done).
- Out-times: "If an out-time is within half a second of the last frame before the shot
  change, extend the out-time to the shot change, respecting the two-frame gap from the shot
  change." We also trim an out-time that hangs up to half a second past a cut back to two
  frames before it (the same convention from the other side; the guide only spells out
  extending).
- Half a second is 12 frames at 24 fps; the 2-frame gap stays 2 frames at every frame rate.

A snap is skipped whenever it would break a `CueRules` limit (overlap, gap < min_gap,
duration outside [min_duration, max_duration], or pushing cps over max_cps). Everything
here is pure except `detect_cuts` / `ShotJob`, which run ffmpeg.

Detection speed (2026-09-23, 5-min slices and full 1080p h264/x265 films): decode
dominates, so downscaling before `scdet` barely helps on the CPU, and `-skip_frame nokey`
finds 1 cut of 72. NVDEC (`-hwaccel cuda` + `scale_cuda`) is 1.6-3.3x faster in wall time
than all-core software decode and ~50-80x cheaper in CPU time, so `SHOT_DECODE=auto` uses it
on CUDA hosts and falls back to software decode. Detection runs in a background thread
next to the ASR (`ShotJob`).
"""

from __future__ import annotations

import bisect
import logging
import math
import re
import shutil
import subprocess
import threading
import time
from collections.abc import Sequence
from pathlib import Path

from app.srt import Cue
from app.standards import DEFAULT_RULES, CueRules

log = logging.getLogger(__name__)

DEFAULT_FPS = 24.0
# scdet score floor. Real cuts between dark shots score 5-7 (Interstellar, Lebowski car
# scenes); scdet's own default of 10 misses ~1 in 5 cuts there. Fast motion also scores
# 4-10, but in clusters, which `filter_cuts` removes.
DEFAULT_THRESHOLD = 5.0
# Candidates closer together than this form one cluster (flash, whip pan, the camera
# inside the bowling ball). Half a second: nothing snaps to a shot that short anyway.
CLUSTER_S = 0.5
# A cluster keeps its strongest candidate only if it beats the runner-up by this factor
# (a cut followed by a smaller echo on the next frame); otherwise it's all motion.
CLUSTER_DOMINANCE = 2.0
# Width of the frames scdet sees. The score is a mean abs frame difference, so it hardly
# changes with size (identical cut lists at 320 px and 1080p on Lebowski).
SCAN_WIDTH = 320

# Codecs NVDEC decodes (ffprobe codec_name). Others go straight to the CPU path.
_NVDEC = frozenset({"h264", "hevc", "av1", "vp9", "vp8", "mpeg2video", "mpeg4", "vc1"})
_SCD_LINE = re.compile(r"lavfi\.scd\.score:\s*([\d.]+),\s*lavfi\.scd\.time:\s*(-?[\d.]+)")


# --- probe helpers --------------------------------------------------------------------------


def video_stream(probe: dict) -> dict | None:
    """The first real video stream (cover art in mkv/mp4 is a 'video' stream too)."""
    for s in probe.get("streams", []):
        if s.get("codec_type") == "video" and not (s.get("disposition") or {}).get("attached_pic"):
            return s
    return None


def probe_fps(probe: dict, default: float = DEFAULT_FPS) -> float:
    """avg_frame_rate of the video stream ("24000/1001"), else r_frame_rate, else default."""
    s = video_stream(probe) or {}
    for key in ("avg_frame_rate", "r_frame_rate"):
        num, _, den = str(s.get(key, "")).partition("/")
        try:
            fps = float(num) / float(den or 1)
        except (ValueError, ZeroDivisionError):
            continue
        if 1.0 <= fps <= 240.0:
            return fps
    return default


def _start_time(probe: dict) -> float:
    try:
        return float(probe.get("format", {}).get("start_time", 0.0))
    except (TypeError, ValueError):
        return 0.0


# --- detection ------------------------------------------------------------------------------


def parse_scdet(stderr: str) -> list[tuple[float, float]]:
    """(time, score) for every `scdet` log line, sorted by time."""
    out = [(float(m.group(2)), float(m.group(1))) for m in _SCD_LINE.finditer(stderr)]
    return sorted(out)


def filter_cuts(
    candidates: Sequence[tuple[float, float]],
    *,
    cluster_s: float = CLUSTER_S,
    dominance: float = CLUSTER_DOMINANCE,
) -> list[float]:
    """Candidates (time, score) -> cut times. Isolated candidates are cuts. A chain of
    candidates each < cluster_s apart is motion or a flash, unless one of them scores at
    least `dominance` x the next best (a real cut with an echo frame)."""
    cands = sorted(candidates)
    cuts: list[float] = []
    i = 0
    while i < len(cands):
        j = i + 1
        while j < len(cands) and cands[j][0] - cands[j - 1][0] < cluster_s:
            j += 1
        group = cands[i:j]
        if len(group) == 1:
            cuts.append(group[0][0])
        else:
            ranked = sorted(group, key=lambda c: c[1], reverse=True)
            if ranked[0][1] >= dominance * ranked[1][1]:
                cuts.append(ranked[0][0])
        i = j
    return cuts


def build_command(
    video: Path, probe: dict, *, decoder: str, threshold: float, threads: int = 0
) -> list[str]:
    """ffmpeg argv that logs one `lavfi.scd.*` line per candidate. decoder: cuda | cpu."""
    s = video_stream(probe) or {}
    index = s.get("index")
    vmap = f"0:{index}" if index is not None else "0:v:0"
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-nostats", "-v", "info"]
    if decoder == "cuda":
        # Decode and downscale on the GPU; only 320 px frames cross PCIe. yuv420p also
        # flattens 10-bit (p010) sources. Full films: ~60 s on a 5090, 15 CPU-s.
        cmd += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
        vf = (
            f"scale_cuda={SCAN_WIDTH}:-2:format=yuv420p,hwdownload,format=yuv420p,"
            f"scdet=threshold={threshold}"
        )
    else:
        if threads:
            cmd += ["-threads", str(threads)]
        # Deblocking is invisible at 320 px and costs ~25% of h264 decode.
        cmd += ["-skip_loop_filter", "all"]
        vf = f"scale={SCAN_WIDTH}:-2:flags=fast_bilinear,scdet=threshold={threshold}"
    # fmt: off
    cmd += ["-i", str(video), "-map", vmap, "-an", "-sn", "-dn", "-vf", vf, "-f", "null", "-"]
    # fmt: on
    return cmd


def pick_decoder(probe: dict, mode: str, whisper_device: str) -> str:
    """auto -> cuda when whisper runs on CUDA and NVDEC knows the codec, else cpu."""
    codec = (video_stream(probe) or {}).get("codec_name", "")
    mode = (mode or "auto").strip().lower()
    if mode == "cpu" or codec not in _NVDEC:
        return "cpu"
    if mode == "cuda" or whisper_device == "cuda":
        return "cuda"
    return "cpu"


class ShotDetectionError(RuntimeError):
    pass


def _run(cmd: list[str], timeout: float, stop: threading.Event) -> str:
    # Lowest CPU priority: it runs next to the ASR (and prod's Jellyfin transcodes).
    # Interstellar on the dev box: +22 s niced vs +27 s not, on an 89 s job.
    if shutil.which("nice"):
        cmd = ["nice", "-n", "19", *cmd]
    proc = subprocess.Popen(
        cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
    )
    deadline = time.monotonic() + timeout
    chunks: list[str] = []
    try:
        while True:
            try:
                # Retrying communicate() after a timeout loses no output.
                _, err = proc.communicate(timeout=1.0)
                chunks.append(err or "")
                break
            except subprocess.TimeoutExpired:
                if stop.is_set():
                    raise ShotDetectionError("cancelled") from None
                if time.monotonic() > deadline:
                    raise ShotDetectionError(f"timed out after {timeout:.0f}s") from None
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    err = "".join(chunks)
    if proc.returncode != 0:
        tail = err.strip().splitlines()[-3:]
        raise ShotDetectionError(f"ffmpeg exit {proc.returncode}: {' | '.join(tail)[:300]}")
    return err


def detect_cuts(
    video: Path,
    probe: dict,
    *,
    decode: str = "auto",
    whisper_device: str = "cpu",
    threshold: float = DEFAULT_THRESHOLD,
    threads: int = 0,
    timeout: float = 1800.0,
    stop: threading.Event | None = None,
    stats: dict | None = None,
) -> list[float]:
    """Shot-change times in seconds on the player's timeline (container start_time
    removed). NVDEC failures (no `video` driver capability, unsupported profile) fall
    back to software decode. Raises ShotDetectionError when ffmpeg fails outright."""
    stop = stop or threading.Event()
    stats = stats if stats is not None else {}
    decoder = pick_decoder(probe, decode, whisper_device)
    try:
        err = _run(
            build_command(video, probe, decoder=decoder, threshold=threshold, threads=threads),
            timeout,
            stop,
        )
    except ShotDetectionError as exc:
        if decoder != "cuda" or stop.is_set():
            raise
        log.warning("NVDEC shot detection failed for %s (%s); using CPU", video.name, exc)
        stats["cuda_error"] = str(exc)[:200]
        decoder = "cpu"
        err = _run(
            build_command(video, probe, decoder=decoder, threshold=threshold, threads=threads),
            timeout,
            stop,
        )
    stats["decoder"] = decoder
    candidates = parse_scdet(err)
    stats["candidates"] = len(candidates)
    offset = _start_time(probe)
    return [round(t - offset, 6) for t in filter_cuts(candidates)]


class ShotJob:
    """Runs `detect_cuts` in a background thread for the length of one transcription, so
    it overlaps the ASR. `snap()` joins it and never raises (fails open)."""

    def __init__(self, video: Path, probe: dict, **kwargs) -> None:
        self.video = video
        self._stop = threading.Event()
        self._kwargs = kwargs
        self._cuts: list[float] | None = None
        self._error: str | None = None
        self._stats: dict = {}
        self._elapsed = 0.0
        self._thread = threading.Thread(target=self._work, args=(probe,), name="shots", daemon=True)
        self._thread.start()

    def _work(self, probe: dict) -> None:
        t0 = time.monotonic()
        try:
            self._cuts = detect_cuts(
                self.video, probe, stop=self._stop, stats=self._stats, **self._kwargs
            )
        except Exception as exc:
            self._error = f"{type(exc).__name__}: {exc}"[:300]
        finally:
            self._elapsed = time.monotonic() - t0

    def close(self) -> None:
        """Kill ffmpeg (job failed or was cancelled). Idempotent."""
        self._stop.set()

    def snap(
        self, cues: list[Cue], rules: CueRules = DEFAULT_RULES, fps: float = DEFAULT_FPS
    ) -> tuple[list[Cue], dict]:
        t0 = time.monotonic()
        self._thread.join()
        stats: dict = {
            "enabled": True,
            "detect_s": round(self._elapsed, 1),
            "wait_s": round(time.monotonic() - t0, 1),
            **self._stats,
        }
        if self._cuts is None:
            log.warning("shot detection failed for %s: %s", self.video.name, self._error)
            stats["error"] = self._error or "no result"
            return cues, stats
        try:
            out, snapped = snap(cues, self._cuts, rules, fps)
        except Exception as exc:  # fail open: keep the composed timings
            log.exception("shot snapping failed for %s", self.video.name)
            stats["error"] = f"{type(exc).__name__}: {exc}"[:300]
            return cues, stats
        stats.update(cuts=len(self._cuts), fps=round(fps, 3), **snapped)
        log.info(
            "shots %s: %d cuts (%s, %.0fs, waited %.1fs), snapped %d starts / %d ends",
            self.video.name,
            len(self._cuts),
            stats.get("decoder", "?"),
            self._elapsed,
            stats["wait_s"],
            snapped["starts_snapped"],
            snapped["ends_snapped"],
        )
        return out, stats


# --- snapping (pure) ------------------------------------------------------------------------


def snap(
    cues: Sequence[Cue],
    cuts: Sequence[float],
    rules: CueRules = DEFAULT_RULES,
    fps: float = DEFAULT_FPS,
    *,
    window_frames: int = 12,
    start_before_frames: int = 4,
    gap_frames: int = 2,
) -> tuple[list[Cue], dict]:
    """Snap cue in/out times to shot changes (see the module docstring for the rules).

    - start in [cut - start_before_frames, cut + window_frames] -> start = cut (the nearest)
    - end in [cut - gap - window_frames, cut - gap + window_frames] -> end = cut - gap
    A snap that would break a rule in `rules` is skipped. Times are in whole ms; a start
    goes to the first ms at or after the cut's frame, so it can't show on the old shot.
    Returns (cues, {"starts_snapped", "ends_snapped", "skipped"}).
    """
    fps = fps if fps and fps > 0 else DEFAULT_FPS
    frame = 1000.0 / fps
    win = round(window_frames * frame)
    before = round(start_before_frames * frame)
    gap_ms = max(round(gap_frames * frame), round(rules.min_gap * 1000))
    min_gap = round(rules.min_gap * 1000)
    cut_ms = sorted({math.ceil(c * 1000 - 1e-6) for c in cuts})

    starts = [round(c.start * 1000) for c in cues]
    ends = [round(c.end * 1000) for c in cues]
    texts = [c.text for c in cues]
    n = len(cues)
    stats = {"starts_snapped": 0, "ends_snapped": 0, "skipped": 0}

    def ok(k: int, s: int, e: int) -> bool:
        d = e / 1000 - s / 1000
        if not rules.min_duration <= d <= rules.max_duration:
            return False
        chars = len(texts[k].replace("\n", ""))
        old = ends[k] / 1000 - starts[k] / 1000
        cps, old_cps = chars / d, (chars / old if old > 0 else math.inf)
        if cps > rules.max_cps and cps > old_cps:
            return False
        if k > 0 and s - ends[k - 1] < min_gap:
            return False
        return not (k + 1 < n and starts[k + 1] - e < min_gap)

    def nearest(t: int, lo: int, hi: int, target: int) -> int | None:
        """The cut in [t - lo, t + hi] closest to `target`."""
        i = bisect.bisect_left(cut_ms, t - lo)
        best = None
        while i < len(cut_ms) and cut_ms[i] <= t + hi:
            if best is None or abs(cut_ms[i] - target) < abs(best - target):
                best = cut_ms[i]
            i += 1
        return best

    for k in range(n):
        # A start `win` after the cut, or `before` ahead of it: cut in [s - win, s + before].
        c = nearest(starts[k], win, before, starts[k])
        if c is None or c == starts[k]:
            continue
        if ok(k, c, ends[k]):
            starts[k] = c
            stats["starts_snapped"] += 1
        else:
            stats["skipped"] += 1

    for k in range(n):
        # Target end = cut - gap, within `win` either way.
        c = nearest(ends[k] + gap_ms, win, win, ends[k] + gap_ms)
        if c is None:
            continue
        e = c - gap_ms
        if e == ends[k]:
            continue
        if ok(k, starts[k], e):
            ends[k] = e
            stats["ends_snapped"] += 1
        else:
            stats["skipped"] += 1

    return [Cue(starts[k] / 1000, ends[k] / 1000, texts[k]) for k in range(n)], stats
