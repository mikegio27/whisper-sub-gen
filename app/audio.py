"""Audio preparation: pick the dialogue track and decode it to 16 kHz mono float32.

Film mixes put dialogue in the center channel and music/effects mostly in the
others, so for a 5.1/7.1 track we keep only FC instead of downmixing. That
helps both the ASR and (later) the forced aligner, which is far less robust to
music than Whisper. The whole file is decoded once into memory and shared by
every stage (~0.7 GB float32 for a 3 h film).
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000

# Decoding even a long film takes well under a minute; this only exists so a
# wedged ffmpeg (e.g. a stalled NFS read) can't hang the worker.
EXTRACT_TIMEOUT_S = 1800

# An FC channel this quiet (dBFS RMS over the whole film) is an empty upmix
# slot, not dialogue. Fall back to a plain downmix.
SILENT_CENTER_DBFS = -55.0

# Named ffmpeg layouts with 3+ channels but no front-center speaker. pan=FC
# on these yields silence (the fallback would catch it, after a wasted decode).
_NO_FC = frozenset({"quad", "quad(side)", "2.2", "3.0(back)", "6.0(front)", "6.1(front)"})

_COMMENTARY_HINTS = ("commentary", "director", "descriptive", "audio description")


@dataclass(frozen=True)
class AudioTrack:
    index: int  # position among the file's audio streams (ffmpeg "0:a:<index>")
    channels: int
    layout: str
    language: str
    title: str

    @property
    def has_center(self) -> bool:
        # ffmpeg names: "5.1", "5.1(side)", "7.1", "6.1", "3.0", "4.0", ...
        # Anything with >= 3 channels in a named surround layout carries FC,
        # except the few in _NO_FC.
        # Stereo/mono and "unknown" layouts don't (or we can't tell).
        layout = self.layout.lower()
        if layout in ("", "unknown", "mono", "stereo", "2.1", "downmix") or layout in _NO_FC:
            return False
        return self.channels >= 3


def pick_track(probe: dict, language: str = "") -> AudioTrack | None:
    """The track most likely to be the main dialogue: skip commentary and
    audio-description tracks, prefer the requested language, then the default
    disposition, then the first."""
    audio = [s for s in probe.get("streams", []) if s.get("codec_type") == "audio"]
    tracks = []
    for i, s in enumerate(audio):
        tags = s.get("tags") or {}
        title = str(tags.get("title", ""))
        disp = s.get("disposition") or {}
        tracks.append(
            (
                AudioTrack(
                    index=i,
                    channels=int(s.get("channels") or 0),
                    layout=str(s.get("channel_layout", "")),
                    language=str(tags.get("language", "")),
                    title=title,
                ),
                bool(disp.get("comment") or disp.get("visual_impaired"))
                or any(h in title.lower() for h in _COMMENTARY_HINTS),
                bool(disp.get("default")),
            )
        )
    if not tracks:
        return None

    lang3 = _iso639_2(language) if language else ""

    def rank(item: tuple[AudioTrack, bool, bool]) -> tuple:
        track, commentary, default = item
        lang_match = bool(lang3) and track.language.lower() in (language.lower(), lang3)
        return (commentary, not lang_match, not default, track.index)

    return min(tracks, key=rank)[0]


def ffmpeg_cmd(video: Path, track: AudioTrack | None, center_only: bool) -> list[str]:
    stream = f"0:a:{track.index}" if track else "0:a:0"
    # fmt: off
    cmd = [
        "ffmpeg", "-nostdin", "-v", "error",
        "-i", str(video),
        "-vn", "-sn", "-dn",
        "-map", stream,
    ]
    # fmt: on
    if center_only and track and track.has_center:
        cmd += ["-af", "pan=mono|c0=FC"]
    # fmt: off
    cmd += [
        "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-f", "s16le", "-c:a", "pcm_s16le",
        "pipe:1",
    ]
    # fmt: on
    return cmd


def _run_ffmpeg(cmd: list[str]):
    import numpy as np

    try:
        result = subprocess.run(cmd, capture_output=True, check=False, timeout=EXTRACT_TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"ffmpeg audio decode timed out after {EXTRACT_TIMEOUT_S}s") from exc
    if result.returncode != 0 or not result.stdout:
        err = result.stderr.decode(errors="replace").strip()[:400]
        raise RuntimeError(f"ffmpeg audio decode failed: {err or 'no audio output'}")
    pcm = np.frombuffer(result.stdout, dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


def rms_dbfs(samples) -> float:
    import numpy as np

    if len(samples) == 0:
        return float("-inf")
    # Subsample: a 3 h film is ~170M samples and a rough level is enough.
    sub = samples[:: max(1, len(samples) // 2_000_000)]
    rms = float(np.sqrt(np.mean(np.square(sub, dtype=np.float64))))
    return 20 * np.log10(rms) if rms > 0 else float("-inf")


def load_audio(video: Path, probe: dict, *, language: str = "", center_only: bool = True):
    """Decode the dialogue track of `video` to a float32 numpy array at 16 kHz."""
    track = pick_track(probe, language)
    use_center = center_only and track is not None and track.has_center
    samples = _run_ffmpeg(ffmpeg_cmd(video, track, use_center))
    if use_center:
        level = rms_dbfs(samples)
        if level < SILENT_CENTER_DBFS:
            log.warning(
                "center channel of %s is near-silent (%.1f dBFS); using a full downmix",
                video.name,
                level,
            )
            samples = _run_ffmpeg(ffmpeg_cmd(video, track, center_only=False))
            use_center = False
    log.info(
        "decoded %s: track %s (%s, %s ch, %s) %s, %.0fs",
        video.name,
        track.index if track else 0,
        track.language or "?" if track else "?",
        track.channels if track else "?",
        track.layout or "?" if track else "?",
        "center channel" if use_center else "downmix",
        len(samples) / SAMPLE_RATE,
    )
    return samples


_ISO639_1_TO_2 = {
    "en": "eng",
    "es": "spa",
    "fr": "fra",
    "de": "deu",
    "it": "ita",
    "pt": "por",
    "ja": "jpn",
    "ko": "kor",
    "zh": "zho",
    "ru": "rus",
    "nl": "nld",
    "sv": "swe",
    "pl": "pol",
}


def _iso639_2(code: str) -> str:
    return _ISO639_1_TO_2.get(code.lower(), code.lower())
