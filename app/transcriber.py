"""faster-whisper wrapper: media in, sidecar .srt out."""

from __future__ import annotations

import logging
import math
import os
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import settings
from .scanner import media_duration, output_path

log = logging.getLogger(__name__)


@dataclass
class Progress:
    """Shared with the API so /status can report live progress."""

    path: str = ""
    duration_s: float = 0.0
    transcribed_s: float = 0.0
    segments: int = 0
    started_at: float = 0.0
    language: str = ""
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def snapshot(self) -> dict:
        with self._lock:
            pct = (
                round(100 * self.transcribed_s / self.duration_s, 1)
                if self.duration_s
                else 0.0
            )
            return {
                "path": self.path,
                "language": self.language,
                "duration_s": round(self.duration_s, 1),
                "transcribed_s": round(self.transcribed_s, 1),
                "percent": min(pct, 100.0),
                "segments": self.segments,
                "elapsed_s": round(time.time() - self.started_at, 1)
                if self.started_at
                else 0.0,
            }


def _format_ts(seconds: float) -> str:
    ms = max(0, int(round(seconds * 1000)))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _extract_audio(video: Path, dest_dir: Path) -> Path:
    """ffmpeg fallback for containers PyAV chokes on: 16 kHz mono wav."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(suffix=".wav", dir=dest_dir)
    os.close(fd)
    wav = Path(name)
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", str(video),
        "-vn", "-sn", "-dn",
        "-map", "0:a:0",
        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        str(wav),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        wav.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg audio extraction failed: {result.stderr[:400]}")
    return wav


class Transcriber:
    """Owns the (lazily loaded, long-lived) whisper model."""

    def __init__(self) -> None:
        self._model = None
        self._model_lock = threading.Lock()
        self.progress = Progress()

    @property
    def model_name(self) -> str:
        return settings.whisper_model

    def _load_model(self):
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    from faster_whisper import WhisperModel

                    log.info(
                        "loading whisper model %s (device=%s compute=%s threads=%s)",
                        settings.whisper_model,
                        settings.whisper_device,
                        settings.compute_type,
                        settings.cpu_threads or "auto",
                    )
                    t0 = time.time()
                    self._model = WhisperModel(
                        settings.whisper_model,
                        device=settings.whisper_device,
                        compute_type=settings.compute_type,
                        cpu_threads=settings.cpu_threads,
                        download_root=settings.model_dir,
                    )
                    log.info("model loaded in %.1fs", time.time() - t0)
        return self._model

    def transcribe(self, video: Path) -> dict:
        """Generate a sidecar SRT for `video`. Returns a result summary."""
        model = self._load_model()
        duration = media_duration(video)
        started = time.time()

        with self.progress._lock:
            self.progress.path = str(video)
            self.progress.duration_s = duration
            self.progress.transcribed_s = 0.0
            self.progress.segments = 0
            self.progress.started_at = started
            self.progress.language = ""

        kwargs = dict(
            task=settings.task,
            language=settings.language or None,
            beam_size=settings.beam_size,
            vad_filter=settings.vad_filter,
        )

        extracted: Path | None = None
        try:
            try:
                segments, info = model.transcribe(str(video), **kwargs)
            except Exception as exc:  # PyAV can't demux everything
                log.warning(
                    "direct decode failed for %s (%s); extracting audio with ffmpeg",
                    video,
                    exc,
                )
                extracted = _extract_audio(video, Path(settings.state_dir) / "tmp")
                segments, info = model.transcribe(str(extracted), **kwargs)

            lang = settings.language or info.language or "und"
            with self.progress._lock:
                self.progress.language = lang

            target = output_path(video, lang)
            tmp_target = target.with_suffix(target.suffix + ".part")
            count = 0
            with open(tmp_target, "w", encoding="utf-8") as fh:
                for seg in segments:  # generator — transcription happens here
                    text = seg.text.strip()
                    if not text:
                        continue
                    count += 1
                    fh.write(
                        f"{count}\n"
                        f"{_format_ts(seg.start)} --> {_format_ts(seg.end)}\n"
                        f"{text}\n\n"
                    )
                    with self.progress._lock:
                        self.progress.transcribed_s = seg.end
                        self.progress.segments = count

            if count == 0:
                tmp_target.unlink(missing_ok=True)
                raise RuntimeError("no speech detected — nothing to write")

            tmp_target.rename(target)
            elapsed = time.time() - started
            speed = duration / elapsed if elapsed and duration else math.nan
            log.info(
                "wrote %s: %d segments, %.0fs media in %.0fs (%.1fx realtime)",
                target,
                count,
                duration,
                elapsed,
                speed,
            )
            return {
                "subtitle": str(target),
                "language": lang,
                "segments": count,
                "duration_s": duration,
                "elapsed_s": elapsed,
            }
        finally:
            if extracted is not None:
                extracted.unlink(missing_ok=True)
            with self.progress._lock:
                self.progress.path = ""
                self.progress.started_at = 0.0
