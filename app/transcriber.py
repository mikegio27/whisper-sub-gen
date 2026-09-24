"""Pipeline driver: media in, sidecar .srt out.

audio.load_audio -> faster-whisper (word timestamps) -> align (CTC forced alignment)
-> cues.compose -> srt. See docs/ROADMAP.md for the stages still to come (LLM correction).
"""

from __future__ import annotations

import logging
import math
import os
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

from .align import Aligner, AlignSegment
from .audio import SAMPLE_RATE, chunk_bounds, load_audio
from .config import settings
from .correct import correct_video
from .cues import Word, compose
from .hallucination import filter_segments
from .punctuation import PROMPT as PUNCT_PROMPT
from .punctuation import repair as repair_punctuation
from .qa import score as qa_score
from .scanner import ffprobe, output_path
from .shots import ShotJob, probe_fps
from .srt import render_srt

# Bump whenever output quality changes materially. Stored per file, so
# REGENERATE_OUTDATED can redo subs made by an older pipeline.
#   0: raw whisper segments (<= sha-2fae320)
#   1: center-channel audio, word timestamps, composed cues (docs/ROADMAP.md P1)
#   2: word times from CTC forced alignment (MMS aligner, app/align.py, ROADMAP P2)
#   3: + re-decode of whisper's no-punctuation stretches (app/punctuation.py)
#   4: + shot-change snapping (app/shots.py), min_linger 1.0, lead_in 0.1
PIPELINE_VERSION = 4

log = logging.getLogger(__name__)


class TranscriptionCancelled(Exception):
    """Raised inside the segment loop when a shutdown abort is requested."""


class ModelLoadError(RuntimeError):
    """The model can't be loaded or run (e.g. a CUDA/driver mismatch). A
    service-wide problem: never counted against individual files."""


class OutputConflict(Exception):
    """A subtitle we may not replace sits at the output path (it can appear
    while a long job runs, e.g. the owner downloading a human sub)."""


def place_output(part: Path, target: Path, may_replace: Callable[[Path], bool] | None) -> None:
    """Move the finished `.part` to `target` without ever clobbering a file we
    aren't allowed to replace. os.link is an atomic "create only if absent";
    only when something is already there do we ask `may_replace` and rename
    over it."""
    try:
        os.link(part, target)
    except FileExistsError:
        if may_replace is None or not may_replace(target):
            raise OutputConflict(f"{target.name} already exists; not overwriting it") from None
        part.replace(target)
        return
    except OSError:
        # Filesystem without hard links: check-then-rename, a small window.
        if target.exists() and (may_replace is None or not may_replace(target)):
            raise OutputConflict(f"{target.name} already exists; not overwriting it") from None
        part.replace(target)
        return
    part.unlink()


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
            pct = round(100 * self.transcribed_s / self.duration_s, 1) if self.duration_s else 0.0
            return {
                "path": self.path,
                "language": self.language,
                "duration_s": round(self.duration_s, 1),
                "transcribed_s": round(self.transcribed_s, 1),
                "percent": min(pct, 100.0),
                "segments": self.segments,
                "elapsed_s": round(time.time() - self.started_at, 1) if self.started_at else 0.0,
            }


class Transcriber:
    """Owns the (lazily loaded, long-lived) whisper model."""

    def __init__(self) -> None:
        self._model = None
        self._model_lock = threading.Lock()
        self.progress = Progress()
        # Set during shutdown (abort mode) to stop the active job cleanly.
        self.cancel = threading.Event()
        # Last model load/smoke-test failure; /healthz reports unhealthy while set.
        self.load_error: str | None = None
        self._aligner: Aligner | None = None

    @property
    def model_name(self) -> str:
        return settings.whisper_model

    def _load_model(self):
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    try:
                        self._model = self._build_model()
                        self.load_error = None
                    except Exception as exc:
                        self.load_error = f"{type(exc).__name__}: {exc}"[:500]
                        log.error("whisper model unusable: %s", self.load_error)
                        raise ModelLoadError(self.load_error) from exc
        return self._model

    def warm_up(self) -> None:
        """Load the model now instead of on the first job, so a broken GPU
        stack shows up at startup (/healthz goes 503) instead of as a
        library full of failed files."""
        try:
            self._load_model()
        except ModelLoadError:
            pass  # already logged and exposed via load_error

    def _build_model(self):
        from faster_whisper import WhisperModel

        log.info(
            "loading whisper model %s (device=%s compute=%s threads=%s)",
            settings.whisper_model,
            settings.whisper_device,
            settings.compute_type,
            settings.cpu_threads or "auto",
        )
        t0 = time.time()
        model = WhisperModel(
            settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.compute_type,
            cpu_threads=settings.cpu_threads,
            download_root=settings.model_dir,
        )
        # CUDA libs (cuBLAS) load lazily, so constructing the model proves
        # little: run one second of silence through it end to end.
        import numpy as np

        segments, _ = model.transcribe(
            np.zeros(SAMPLE_RATE, dtype=np.float32), language="en", beam_size=1
        )
        list(segments)
        log.info("model loaded and smoke-tested in %.1fs", time.time() - t0)
        return model

    def transcribe(self, video: Path, *, may_replace: Callable[[Path], bool] | None = None) -> dict:
        """Generate a sidecar SRT for `video`. Returns a result summary.

        An existing file at the output path is only replaced when
        `may_replace(path)` says so, checked at the last moment; otherwise
        OutputConflict is raised and nothing is written."""
        model = self._load_model()
        try:
            probe = ffprobe(video)
        except (RuntimeError, OSError, subprocess.SubprocessError, ValueError) as exc:
            log.warning("ffprobe failed for %s (%s); decoding the first audio track", video, exc)
            probe = {}
        duration = _probe_duration(probe)
        started = time.time()

        with self.progress._lock:
            self.progress.path = str(video)
            self.progress.duration_s = duration
            self.progress.transcribed_s = 0.0
            self.progress.segments = 0
            self.progress.started_at = started
            self.progress.language = ""

        # Shot-change detection (ffmpeg) runs next to the ASR; snap() joins it
        # after compose and fails open.
        shots = (
            ShotJob(
                video,
                probe,
                decode=settings.shot_decode,
                whisper_device=settings.whisper_device,
                threshold=settings.shot_threshold,
                threads=settings.shot_threads or settings.cpu_threads,
                timeout=settings.shot_timeout_s,
            )
            if settings.shot_snap
            else None
        )
        try:
            audio = load_audio(
                video,
                probe,
                language=settings.language,
                center_only=settings.audio_center_channel,
            )
            if self.cancel.is_set():
                raise TranscriptionCancelled(str(video))
            duration = duration or len(audio) / SAMPLE_RATE
            with self.progress._lock:
                self.progress.duration_s = duration

            bounds = chunk_bounds(audio, settings.asr_chunk_s)
            segments, info = self._asr(model, audio, bounds)
            lang = settings.language or info.language or "und"
            with self.progress._lock:
                self.progress.language = lang

            asr_segments = self._collect_words(video, segments, duration, started)
            # Before alignment: no point aligning a fake line over the score.
            asr_segments, hallucinated = filter_segments(asr_segments)
            if hallucinated:
                log.info(
                    "dropped %d non-speech hallucinations in %s: %s",
                    len(hallucinated),
                    video.name,
                    "; ".join(hallucinated[:8]),
                )
            punct: dict = {"enabled": False}
            if settings.punct_repair:
                asr_segments, punct = self._repair_punctuation(
                    video, model, audio, asr_segments, lang
                )
            words = [w for seg in asr_segments for w in seg.words]
            aligned: dict = {"enabled": False}
            if settings.align_words:
                words, aligned = self._align(video, audio, asr_segments, words, lang)
            del audio  # ~0.7 GB for a long film; not needed past alignment
            corrected: dict = {"enabled": False}
            if settings.llm_correct and settings.ollama_url:
                # Fails open (keeps the ASR text); stops early on cancel/budget.
                try:
                    words, corrected = correct_video(words, video, cancel=self.cancel)
                except Exception as exc:  # fail open: keep the ASR words
                    log.exception("LLM correction failed for %s; keeping ASR text", video.name)
                    corrected = {"error": f"{type(exc).__name__}: {exc}"[:300]}
                corrected["enabled"] = True
                if self.cancel.is_set():
                    raise TranscriptionCancelled(str(video))
            cues = compose(words)
            if not cues:
                raise RuntimeError("no speech detected — nothing to write")
            shot_stats: dict = {"enabled": False}
            if shots is not None:
                cues, shot_stats = shots.snap(cues, fps=probe_fps(probe))
            quality = qa_score(cues)

            target = output_path(video, lang)
            tmp_target = target.with_suffix(target.suffix + ".part")
            try:
                tmp_target.write_text(render_srt(cues), encoding="utf-8")
                place_output(tmp_target, target, may_replace)
            except BaseException:
                # Never leave a stale .part next to the media.
                tmp_target.unlink(missing_ok=True)
                raise
            st = target.stat()
            elapsed = time.time() - started
            speed = duration / elapsed if elapsed and duration else math.nan
            log.info(
                "wrote %s: %d cues from %d words, %.0fs media in %.0fs (%.1fx realtime), "
                "qa %.1f violations/100 cues",
                target,
                len(cues),
                len(words),
                duration,
                elapsed,
                speed,
                quality["violations_per_100"],
            )
            return {
                "subtitle": str(target),
                "subtitle_size": st.st_size,
                "subtitle_mtime": st.st_mtime,
                "language": lang,
                "segments": len(cues),
                "words": len(words),
                "duration_s": duration,
                "elapsed_s": elapsed,
                "pipeline": _effective_pipeline(aligned),
                "qa": quality,
                "hallucinations_dropped": len(hallucinated),
                "punct_repair": punct,
                "aligned": aligned,
                "corrected": corrected,
                "shots": shot_stats,
            }
        finally:
            if shots is not None:
                shots.close()  # kills ffmpeg if the job failed or was cancelled
            with self.progress._lock:
                self.progress.path = ""
                self.progress.started_at = 0.0
                self.progress.duration_s = 0.0
                self.progress.transcribed_s = 0.0
                self.progress.segments = 0
                self.progress.language = ""

    def _align(
        self, video: Path, audio, asr_segments: list[AlignSegment], words: list[Word], lang: str
    ) -> tuple[list[Word], dict]:
        """Forced-align the words. Never fails the job: on any aligner error the
        whisper word times are kept, and the stats say so."""
        if self._aligner is None:
            self._aligner = Aligner(
                settings.align_model,
                device=settings.whisper_device,
                model_dir=settings.model_dir,
                batch_seconds=settings.align_batch_seconds,
                min_score=settings.align_min_score,
                cpu_threads=settings.cpu_threads,
            )

        def check_cancel() -> None:
            if self.cancel.is_set():
                raise TranscriptionCancelled(str(video))

        try:
            res = self._aligner.align(audio, asr_segments, lang, check_cancel=check_cancel)
        except TranscriptionCancelled:
            raise
        except Exception as exc:
            log.exception("alignment failed for %s; keeping whisper word times", video.name)
            return words, {"enabled": True, "error": f"{type(exc).__name__}: {exc}"[:300]}
        finally:
            if settings.align_free_after_job:
                self._aligner.free()
        stats = res.stats.as_dict()
        log.info(
            "aligned %s: %d/%d segments (%d fell back: %s) in %.1fs, mean score %s",
            video.name,
            stats["aligned"],
            stats["segments"],
            stats["fallback"],
            stats["reasons"] or "-",
            stats["elapsed_s"],
            stats["mean_score"],
        )
        return res.words, {"enabled": True, **stats}

    @staticmethod
    def _asr_kwargs() -> dict:
        """faster-whisper decode options shared by the main pass and repairs."""
        return {
            "task": settings.task,
            "beam_size": settings.beam_size,
            "condition_on_previous_text": settings.condition_on_previous_text,
            "word_timestamps": True,
            "hallucination_silence_threshold": settings.hallucination_silence_threshold or None,
            "vad_filter": settings.vad_filter,
            "vad_parameters": {
                "threshold": settings.vad_threshold,
                "min_silence_duration_ms": settings.vad_min_silence_ms,
                "speech_pad_ms": settings.vad_speech_pad_ms,
            },
        }

    def _repair_punctuation(self, video: Path, model, audio, segments, lang: str):
        """Re-decode whisper's no-punctuation stretches (app/punctuation.py). Never
        fails the job: on any error the original segments are kept."""
        kwargs = {**self._asr_kwargs(), "initial_prompt": PUNCT_PROMPT}

        def decode(a: float, b: float) -> list[Word]:
            if self.cancel.is_set():
                raise TranscriptionCancelled(str(video))
            lo, hi = int(a * SAMPLE_RATE), int(b * SAMPLE_RATE)
            segs, _ = model.transcribe(audio[lo:hi], language=lang, **kwargs)
            return [
                Word(w.start + a, w.end + a, w.word, w.probability)
                for seg in segs
                for w in seg.words or ()
            ]

        try:
            segments, stats = repair_punctuation(
                segments, decode, lambda s, e, w: AlignSegment(s, e, w)
            )
        except TranscriptionCancelled:
            raise
        except Exception:
            log.exception("punctuation repair failed for %s; keeping whisper text", video.name)
            return segments, {"enabled": True, "error": True}
        if stats.candidates:
            log.info(
                "punctuation repair %s: %d/%d stretches re-decoded (%s)",
                video.name,
                stats.repaired,
                stats.candidates,
                stats.reasons or "-",
            )
        return segments, {"enabled": True, **stats.as_dict()}

    def _asr(self, model, audio, bounds: list[tuple[int, int]]):
        """Run faster-whisper chunk by chunk (see audio.chunk_bounds), yielding
        segments with film-relative times. The first chunk detects the language;
        later chunks reuse it so a quiet or foreign-dialogue stretch can't flip
        it mid-film. Returns (segment generator, first chunk's info)."""
        kwargs = self._asr_kwargs()
        s0, e0 = bounds[0]
        first, info = model.transcribe(audio[s0:e0], language=settings.language or None, **kwargs)
        language = settings.language or info.language

        def shifted():
            for i, (s, e) in enumerate(bounds):
                if i == 0:
                    segs = first
                else:
                    segs, _ = model.transcribe(audio[s:e], language=language, **kwargs)
                off = s / SAMPLE_RATE
                for seg in segs:
                    yield SimpleNamespace(
                        start=seg.start + off,
                        end=seg.end + off,
                        words=[
                            SimpleNamespace(
                                start=w.start + off,
                                end=w.end + off,
                                word=w.word,
                                probability=w.probability,
                            )
                            for w in seg.words or ()
                        ],
                    )

        if len(bounds) > 1:
            log.info(
                "transcribing in %d chunks (ASR_CHUNK_S=%s)", len(bounds), settings.asr_chunk_s
            )
        return shifted(), info

    def _collect_words(
        self, video: Path, segments, duration: float, started: float
    ) -> list[AlignSegment]:
        """Drain the segment generator (this is where the ASR actually runs),
        keeping live progress and honouring the cancel flag. Words stay grouped
        by segment, which is the unit the aligner works in."""
        out: list[AlignSegment] = []
        count = 0
        heartbeat = settings.progress_log_seconds
        next_heartbeat = time.time() + heartbeat if heartbeat else math.inf
        for seg in segments:
            if self.cancel.is_set():
                raise TranscriptionCancelled(str(video))
            seg_words = [Word(w.start, w.end, w.word, w.probability) for w in seg.words or ()]
            if seg_words:
                out.append(AlignSegment(seg.start, seg.end, seg_words))
            count += 1
            with self.progress._lock:
                self.progress.transcribed_s = seg.end
                self.progress.segments = count
            now = time.time()
            if now >= next_heartbeat:
                next_heartbeat = now + heartbeat
                elapsed = now - started
                speed = seg.end / elapsed if elapsed else 0.0
                if duration and speed:
                    pct = min(100.0, 100 * seg.end / duration)
                    eta_min = (duration - seg.end) / speed / 60
                    log.info(
                        "progress %s: %.0f%% (%.0f/%.0fs), %.1fx realtime, ~%.1f min left",
                        video.name,
                        pct,
                        seg.end,
                        duration,
                        speed,
                        eta_min,
                    )
                else:
                    log.info(
                        "progress %s: %.0fs transcribed, %.1fx realtime",
                        video.name,
                        seg.end,
                        speed,
                    )
        return out


def _effective_pipeline(aligned: dict) -> int:
    """The version this output really is. When alignment was off, errored, or
    mostly fell back, the timings are v1's, so record 1: REGENERATE_OUTDATED
    then redoes the file once the aligner works, instead of trusting it."""
    if not aligned.get("enabled") or aligned.get("error"):
        return 1
    segments = aligned.get("segments") or 0
    if segments and aligned.get("aligned", 0) < 0.5 * segments:
        return 1
    return PIPELINE_VERSION


def _probe_duration(probe: dict) -> float:
    try:
        return float(probe.get("format", {}).get("duration", 0.0))
    except (TypeError, ValueError):
        return 0.0
