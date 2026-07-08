"""Environment-driven configuration.

Every setting can be provided as an environment variable so the container is
fully configurable from a k8s Deployment or a plain `docker run`.
"""

from __future__ import annotations

import re
from datetime import time

from pydantic import field_validator
from pydantic_settings import BaseSettings

_WINDOW_RE = re.compile(r"^(\d{1,2}):(\d{2})-(\d{1,2}):(\d{2})$")


class Settings(BaseSettings):
    # --- Media discovery ---
    media_dirs: str = "/media"  # comma-separated list of mount points to scan
    video_extensions: str = "mkv,mp4,avi,mov,m4v,ts,webm,wmv,flv,mpg,mpeg"
    ignore_patterns: str = (
        "*/extras/*,*/trailers/*,*/backdrops/*,*/trickplay/*,"
        "*-trailer.*,*sample*,*/theme.*"
    )
    # Skip files modified more recently than this (still being copied/imported).
    file_min_age_minutes: int = 10

    # --- Skip logic: what counts as "already has subtitles" ---
    skip_if_external_subs: bool = True  # any sidecar .srt/.ass/.vtt/... next to file
    skip_if_embedded_subs: bool = True  # any subtitle stream inside the container
    # Only count embedded text subs (srt/ass/mov_text); bitmap subs (PGS/VobSub)
    # won't block generation when this is true.
    embedded_text_subs_only: bool = False
    max_retries: int = 2  # per-file transcription attempts before giving up

    # --- Whisper / transcription ---
    whisper_model: str = "large-v3-turbo"
    whisper_device: str = "cpu"  # cpu | cuda
    # auto => int8 on cpu, float16 on cuda
    whisper_compute_type: str = "auto"
    cpu_threads: int = 0  # 0 = let ctranslate2 pick (all cores)
    beam_size: int = 5
    # Empty = auto-detect per file; otherwise ISO 639-1 like "en"
    language: str = ""
    task: str = "transcribe"  # transcribe | translate (translate => English)
    vad_filter: bool = True  # skip long silences, big speedup on movies
    model_dir: str = "/models"  # HuggingFace download cache (mount a volume)

    # --- Output ---
    # Filename becomes "<video stem><subtitle_tag>.<lang>.srt"
    subtitle_tag: str = ""
    overwrite_existing_output: bool = False

    # --- Run mode & scheduling ---
    # continuous: scan every scan_interval_minutes and process
    # manual:     only work when triggered through the API
    run_mode: str = "continuous"
    scan_interval_minutes: int = 60
    scan_on_startup: bool = True
    # "HH:MM-HH:MM" (may cross midnight, e.g. "22:00-06:00"). Empty = no gating.
    # Outside the window the worker pauses between files.
    work_window: str = ""
    # Jobs queued through the API ignore the work window when true.
    manual_bypass_window: bool = True

    # --- API ---
    host: str = "0.0.0.0"
    port: int = 8000
    api_key: str = ""  # empty = no auth; otherwise required as X-Api-Key/Bearer

    # --- Misc ---
    state_dir: str = "/data"  # sqlite db + temp audio extractions
    webhook_url: str = ""  # POSTed a JSON result after each processed file
    log_level: str = "INFO"
    # Log transcription progress (%, speed, ETA) every N seconds; 0 disables.
    progress_log_seconds: int = 60
    # On SIGTERM: "abort" cancels the active job and deletes its partial
    # output (it is retried after restart); "finish" completes the active
    # file first — make sure the pod's terminationGracePeriodSeconds is
    # large enough to transcribe one full movie.
    shutdown_mode: str = "abort"

    @field_validator("shutdown_mode")
    @classmethod
    def _check_shutdown_mode(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in ("abort", "finish"):
            raise ValueError("SHUTDOWN_MODE must be 'abort' or 'finish'")
        return v

    @field_validator("run_mode")
    @classmethod
    def _check_run_mode(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in ("continuous", "manual"):
            raise ValueError("RUN_MODE must be 'continuous' or 'manual'")
        return v

    @field_validator("work_window")
    @classmethod
    def _check_window(cls, v: str) -> str:
        v = v.strip()
        if v and not _WINDOW_RE.match(v):
            raise ValueError("WORK_WINDOW must look like '22:00-06:00'")
        return v

    # --- Parsed helpers ---

    @property
    def media_dir_list(self) -> list[str]:
        return [d.strip() for d in self.media_dirs.split(",") if d.strip()]

    @property
    def extension_set(self) -> set[str]:
        return {
            e.strip().lstrip(".").lower()
            for e in self.video_extensions.split(",")
            if e.strip()
        }

    @property
    def ignore_pattern_list(self) -> list[str]:
        return [p.strip() for p in self.ignore_patterns.split(",") if p.strip()]

    @property
    def window(self) -> tuple[time, time] | None:
        m = _WINDOW_RE.match(self.work_window)
        if not m:
            return None
        h1, m1, h2, m2 = (int(g) for g in m.groups())
        return time(h1, m1), time(h2, m2)

    @property
    def compute_type(self) -> str:
        if self.whisper_compute_type != "auto":
            return self.whisper_compute_type
        return "float16" if self.whisper_device == "cuda" else "int8"


settings = Settings()


def in_work_window(now_time: time | None = None) -> bool:
    """True when processing is currently allowed."""
    win = settings.window
    if win is None:
        return True
    from datetime import datetime

    now = now_time or datetime.now().time()
    start, end = win
    if start <= end:
        return start <= now < end
    return now >= start or now < end  # window crosses midnight
