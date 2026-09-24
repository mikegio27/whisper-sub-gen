"""Environment-driven configuration.

Every setting can be provided as an environment variable so the container is
fully configurable from a k8s Deployment or a plain `docker run`.
"""

from __future__ import annotations

import re
from datetime import datetime, time

from pydantic import field_validator
from pydantic_settings import BaseSettings

_WINDOW_RE = re.compile(r"^(\d{1,2}):(\d{2})-(\d{1,2}):(\d{2})$")


class Settings(BaseSettings):
    # --- Media discovery ---
    media_dirs: str = "/media"  # comma-separated list of mount points to scan
    video_extensions: str = "mkv,mp4,avi,mov,m4v,ts,webm,wmv,flv,mpg,mpeg"
    ignore_patterns: str = (
        "*/extras/*,*/trailers/*,*/backdrops/*,*/trickplay/*,*-trailer.*,*sample*,*/theme.*"
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
    # Silero VAD pre-filter. Off by default: on film audio it drops quiet or
    # music-backed dialogue outright and garbles word timestamps at chunk
    # boundaries. On a 4 min OotP clip (2026-09-22) it cut 2:45 of the 4:03,
    # lost "Come on, Dudley, let's go. What's going on?" and smeared "What are
    # you doing?" across 9 s; with it off, text and word times were right. It
    # only buys speed, which the GPU makes moot. Silence hallucinations are
    # handled by hallucination_silence_threshold + the cue composer instead.
    vad_filter: bool = False
    # Only used when vad_filter is on.
    vad_threshold: float = 0.5
    vad_min_silence_ms: int = 500
    vad_speech_pad_ms: int = 200
    # Feeding each window the previous text makes one hallucination repeat for
    # minutes (the "Is she dead?" x3 loops). Off costs little on films.
    condition_on_previous_text: bool = False
    # Re-decode stretches whisper wrote as run-on lowercase with no punctuation,
    # with a punctuated prompt (app/punctuation.py). Kept only when the words
    # match and are now punctuated.
    punct_repair: bool = True
    # Transcribe in chunks of about this many seconds, cut at quiet points.
    # faster-whisper's feature extraction holds ~3.3 GB per hour of input in
    # RAM; 1200 s keeps that near 1.1 GB whatever the film's length. 0 = one
    # pass over the whole film.
    asr_chunk_s: int = 1200
    # Skip silent stretches longer than this (s) around a suspected
    # hallucination. Needs word timestamps, which the pipeline always uses.
    hallucination_silence_threshold: float = 2.0
    # Keep only the center channel of 5.1/7.1 tracks (dialogue lives there),
    # falling back to a downmix when it's silent. false = always downmix.
    audio_center_channel: bool = True
    model_dir: str = "/models"  # HuggingFace download cache (mount a volume)

    # --- Forced alignment (app/align.py, ROADMAP P2) ---
    # Re-time whisper's words with a CTC aligner. A segment that can't be
    # aligned, or an aligner failure, keeps whisper's word times.
    align_words: bool = True
    align_model: str = "MahmoudAshraf/mms-300m-1130-forced-aligner"
    # Mean per-token log-prob below which a segment keeps whisper's timings.
    # -5 drops ~0.5% of segments on The Big Lebowski (2026-09-22), incl. the
    # "Transcription by CastingWords" credits hallucination; see align.py.
    align_min_score: float = -5.0
    # Padded audio per forward pass (s). Bounds the aligner's VRAM.
    align_batch_seconds: float = 120.0
    # Unload the aligner after each job: the stages share a 12 GB card.
    align_free_after_job: bool = True

    # --- Shot-change snapping (app/shots.py) ---
    # Snap cue in/out times to nearby cuts (Netflix timing guide). ffmpeg scdet
    # runs in a background thread alongside the ASR; any failure keeps the
    # composed timings. Eval 2026-09-23 (5 films): median end error -2..-47 ms
    # on 4/5, QA/100 lower on 4/5; onsets within +-7 ms. ~+20% job time on a 5090.
    shot_snap: bool = True
    # auto = NVDEC (-hwaccel cuda + scale_cuda) when WHISPER_DEVICE=cuda, else CPU; NVDEC
    # errors fall back to CPU. On k8s NVDEC needs NVIDIA_DRIVER_CAPABILITIES to
    # include "video". cpu | cuda force one.
    shot_decode: str = "auto"
    # scdet score floor. Dark-scene cuts score 5-7; motion clusters are
    # filtered separately (shots.filter_cuts).
    shot_threshold: float = 5.0
    # ffmpeg decode threads for the CPU path; 0 = CPU_THREADS, else ffmpeg's default.
    shot_threads: int = 0
    shot_timeout_s: float = 1800.0

    # --- Output ---
    # Filename becomes "<video stem><subtitle_tag>.<lang>.srt"
    subtitle_tag: str = ""
    overwrite_existing_output: bool = False
    # Re-generate subs this service wrote with an older pipeline version, but
    # only while the file on disk is still exactly what we wrote (size+mtime),
    # so a sub the owner replaced or edited is never touched.
    regenerate_outdated: bool = False

    # --- LLM word correction (app/correct.py, app/context.py, ROADMAP P3) ---
    # A local LLM (Ollama) proposes replacements for low-confidence words;
    # code accepts only sound-alike edits or exact cast/character names. Fails
    # open: Ollama down, slow or talking nonsense = the ASR text is kept.
    # Off until the eval shows it helps.
    llm_correct: bool = False
    ollama_url: str = ""  # e.g. http://ollama.ollama.svc.cluster.local:11434
    # dozai client token (Bearer). With it, OLLAMA_URL points at dozai
    # (http://dozai.dozai.svc.cluster.local:8080/ollama/auto), which picks the GPU,
    # records usage and owns model loading. Empty = talk to Ollama directly. A
    # secret: from the whisper-dozai SealedSecret, never the ConfigMap.
    ollama_api_key: str = ""
    ollama_model: str = "qwen3.5:4b"
    # Words whisper is less sure of than this get a second look. 0.6 flagged
    # ~9-12% of words on the OotP clip (2026-09-22), incl. "Austin?" 0.22
    # (asked you), "Fig?" 0.47; capitalised names are flagged separately.
    llm_flag_prob: float = 0.6
    llm_timeout_s: float = 120.0  # per request (the first one loads the model)
    llm_budget_s: float = 900.0  # per film; stop sending windows past this
    # Character names come from Jellyfin's People for the item. The API key is
    # a secret: set it from a SealedSecret (secretRef), never the ConfigMap.
    jellyfin_url: str = ""  # http://jellyfin.jellyfin.svc.cluster.local:8096
    jellyfin_api_key: str = ""

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
    # After each job, restart cleanly (SIGTERM to ourselves; the pod restarts)
    # if RSS is above this fraction of the container memory limit. Between
    # jobs nothing is lost; mid-job OOMKills lose the film. 0 disables.
    recycle_memory_fraction: float = 0.7

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
            e.strip().lstrip(".").lower() for e in self.video_extensions.split(",") if e.strip()
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
    now = now_time or datetime.now().time()
    start, end = win
    if start <= end:
        return start <= now < end
    return now >= start or now < end  # window crosses midnight
