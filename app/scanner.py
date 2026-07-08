"""Media discovery and existing-subtitle detection."""

from __future__ import annotations

import fnmatch
import json
import logging
import subprocess
import time
from pathlib import Path

from .config import settings

log = logging.getLogger(__name__)

SUBTITLE_EXTENSIONS = {".srt", ".ass", ".ssa", ".vtt", ".sub", ".idx", ".smi"}
TEXT_SUB_CODECS = {"subrip", "srt", "ass", "ssa", "webvtt", "mov_text", "text"}


def output_path(video: Path, lang: str) -> Path:
    """Sidecar path Jellyfin will pick up: '<stem><tag>.<lang>.srt'."""
    return video.with_name(f"{video.stem}{settings.subtitle_tag}.{lang}.srt")


def _ignored(path: Path) -> bool:
    p = str(path).lower()
    return any(fnmatch.fnmatch(p, pat.lower()) for pat in settings.ignore_pattern_list)


def find_videos(roots: list[str] | None = None) -> list[Path]:
    """All candidate video files under the media roots, oldest first."""
    exts = settings.extension_set
    min_age = settings.file_min_age_minutes * 60
    now = time.time()
    videos: list[tuple[float, Path]] = []
    for root in roots or settings.media_dir_list:
        base = Path(root)
        if not base.is_dir():
            log.warning("media dir %s does not exist or is not mounted", root)
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            # Leftover partial from a hard kill (an active job touches its
            # .part constantly, so a stale mtime means it's orphaned).
            if path.name.endswith(".srt.part"):
                try:
                    if now - path.stat().st_mtime > 3600:
                        path.unlink()
                        log.info("removed stale partial %s", path)
                except OSError:
                    pass
                continue
            if path.suffix.lstrip(".").lower() not in exts:
                continue
            if _ignored(path):
                continue
            try:
                st = path.stat()
            except OSError:
                continue
            if now - st.st_mtime < min_age:
                log.debug("skipping %s: modified too recently", path)
                continue
            videos.append((st.st_mtime, path))
    videos.sort()
    return [p for _, p in videos]


def external_subtitles(video: Path) -> list[Path]:
    """Sidecar subtitle files whose name starts with the video's stem."""
    stem = video.stem
    found = []
    try:
        for sib in video.parent.iterdir():
            if (
                sib.is_file()
                and sib.suffix.lower() in SUBTITLE_EXTENSIONS
                and sib.stem.split(".")[0] == stem.split(".")[0]
                and sib.name.startswith(stem)
            ):
                found.append(sib)
    except OSError as exc:
        log.warning("could not list %s: %s", video.parent, exc)
    return found


def ffprobe(video: Path) -> dict:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-print_format", "json",
            "-show_streams", "-show_format", str(video),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if out.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {out.stderr.strip()[:400]}")
    return json.loads(out.stdout)


def embedded_subtitle_streams(probe: dict) -> list[dict]:
    streams = [s for s in probe.get("streams", []) if s.get("codec_type") == "subtitle"]
    if settings.embedded_text_subs_only:
        streams = [s for s in streams if s.get("codec_name", "") in TEXT_SUB_CODECS]
    return streams


def check_needs_subtitles(video: Path) -> tuple[bool, str]:
    """(needs_generation, reason_if_not)."""
    lang = settings.language or "*"
    if not settings.overwrite_existing_output:
        # Anything we (or a previous run) already wrote for any language.
        pattern = f"{video.stem}{settings.subtitle_tag}.{lang}.srt"
        existing = [
            s for s in external_subtitles(video)
            if fnmatch.fnmatch(s.name, pattern)
        ]
        if existing:
            return False, f"output already exists: {existing[0].name}"

    if settings.skip_if_external_subs:
        subs = external_subtitles(video)
        if subs:
            return False, f"external subtitles present: {subs[0].name}"

    if settings.skip_if_embedded_subs:
        try:
            probe = ffprobe(video)
        except (RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
            log.warning("ffprobe failed for %s: %s", video, exc)
            return True, ""  # can't tell — let transcription try
        streams = embedded_subtitle_streams(probe)
        if streams:
            langs = {s.get("tags", {}).get("language", "?") for s in streams}
            return False, f"embedded subtitles present ({', '.join(sorted(langs))})"

    return True, ""


def media_duration(video: Path) -> float:
    try:
        probe = ffprobe(video)
        return float(probe.get("format", {}).get("duration", 0.0))
    except Exception:
        return 0.0
