"""Job queue, worker thread, and schedule gating."""

from __future__ import annotations

import itertools
import json
import logging
import queue
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from . import metrics
from .config import in_work_window, settings
from .scanner import check_needs_subtitles, external_subtitles, find_videos
from .state import StateStore
from .transcriber import (
    PIPELINE_VERSION,
    ModelLoadError,
    OutputConflict,
    Transcriber,
    TranscriptionCancelled,
)

log = logging.getLogger(__name__)


@dataclass
class Job:
    id: int
    path: Path
    source: str  # scan | api
    bypass_window: bool = False
    force: bool = False  # regenerate even if subtitles already exist
    # Our own outdated subtitle this job replaces (REGENERATE_OUTDATED).
    replaces: str | None = None
    queued_at: float = field(default_factory=time.time)


class Worker:
    def __init__(self) -> None:
        self.store = StateStore(settings.state_dir)
        self.transcriber = Transcriber()
        self._queue: queue.Queue[Job] = queue.Queue()
        self._queued: dict[str, Job] = {}
        self._current: Job | None = None
        self._job_ids = itertools.count(1)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._scan_now = threading.Event()
        self._scanning = False
        self._last_scan: str | None = None
        self._paused_for_window = False
        self._threads: list[threading.Thread] = []

    # --- lifecycle ---

    def start(self) -> None:
        self._threads = [threading.Thread(target=self._work_loop, name="worker", daemon=True)]
        if settings.whisper_device != "cpu":
            # Surface a broken GPU stack at startup, not as failed files.
            threading.Thread(target=self.transcriber.warm_up, name="warm-up", daemon=True).start()
        if settings.run_mode == "continuous":
            self._threads.append(
                threading.Thread(target=self._scan_loop, name="scanner", daemon=True)
            )
        for t in self._threads:
            t.start()
        log.info(
            "worker started (run_mode=%s, window=%s, media=%s)",
            settings.run_mode,
            settings.work_window or "always",
            settings.media_dir_list,
        )

    def stop(self) -> None:
        """Begin shutdown. abort: cancel the active job and clean up its
        partial output; finish: let the active job complete. Queued jobs are
        dropped either way — the next scan re-finds them."""
        self._stop.set()
        self._scan_now.set()
        if settings.shutdown_mode == "abort":
            self.transcriber.cancel.set()
        with self._lock:
            active = self._current
        if active:
            log.info(
                "shutdown: %s active job %s (%d queued jobs dropped, re-queued on next scan)",
                "aborting" if settings.shutdown_mode == "abort" else "finishing",
                active.path.name,
                self._queue.qsize(),
            )

    def join(self, timeout: float | None = None) -> None:
        """Wait for the worker thread to finish the current job and exit."""
        for t in self._threads:
            if t.name == "worker":
                t.join(timeout)
                if t.is_alive():
                    log.warning("worker did not stop within %ss", timeout)

    # --- public API used by routes ---

    def enqueue(
        self,
        path: Path,
        source: str,
        bypass_window: bool = False,
        force: bool = False,
        replaces: str | None = None,
    ) -> Job | None:
        with self._lock:
            key = str(path)
            queued = self._queued.get(key)
            if queued is not None and force and not queued.force:
                # An explicit /process force beats a scan/regen job already
                # waiting for the same file (it has already forgotten the row).
                queued.force, queued.replaces = True, None
                queued.bypass_window = queued.bypass_window or bypass_window
                return queued
            if queued is not None or (self._current and str(self._current.path) == key):
                return None
            job = Job(
                id=next(self._job_ids),
                path=path,
                source=source,
                bypass_window=bypass_window,
                force=force,
                replaces=replaces,
            )
            self._queued[key] = job
        self._queue.put(job)
        metrics.QUEUED.inc()
        return job

    def trigger_scan(self, roots: list[str] | None = None) -> bool:
        """Kick a scan in the background. Returns False if one is running."""
        if self._scanning:
            return False
        threading.Thread(
            target=self._scan, args=(roots, "api"), name="api-scan", daemon=True
        ).start()
        return True

    def status(self) -> dict:
        with self._lock:
            current = (
                {
                    "id": self._current.id,
                    "source": self._current.source,
                    **self.transcriber.progress.snapshot(),
                }
                if self._current
                else None
            )
            pending = sorted(self._queued)
        return {
            "run_mode": settings.run_mode,
            "model": settings.whisper_model,
            "device": settings.whisper_device,
            "work_window": settings.work_window or None,
            "in_window": in_work_window(),
            "paused_for_window": self._paused_for_window,
            "scanning": self._scanning,
            "last_scan": self._last_scan,
            "current": current,
            "queue_length": len(pending),
            "queue": pending[:50],
            "totals": self.store.counts(),
        }

    # --- scanning ---

    def _scan_loop(self) -> None:
        if not settings.scan_on_startup:
            self._scan_now.wait(timeout=settings.scan_interval_minutes * 60)
            self._scan_now.clear()
        while not self._stop.is_set():
            self._scan(None, "scheduled")
            self._scan_now.wait(timeout=settings.scan_interval_minutes * 60)
            self._scan_now.clear()

    def _scan(self, roots: list[str] | None, source: str) -> None:
        if self._scanning:
            return
        self._scanning = True
        try:
            t0 = time.time()
            log.info("scan started (%s)", source)
            queued = skipped = regen = 0
            for video in find_videos(roots):
                if self._stop.is_set():
                    return
                try:
                    st = video.stat()
                except OSError:
                    continue
                if self.store.should_skip(
                    str(video), st.st_size, st.st_mtime, settings.max_retries
                ):
                    own = self._outdated_own_sub(video) if settings.regenerate_outdated else None
                    if own and self.enqueue(video, source="regen", replaces=own):
                        regen += 1
                    continue
                needs, reason = check_needs_subtitles(video)
                if not needs:
                    self.store.record(str(video), st.st_size, st.st_mtime, "skipped", reason=reason)
                    skipped += 1
                    continue
                if self.enqueue(video, source="scan"):
                    queued += 1
            self._last_scan = datetime.now().astimezone().isoformat(timespec="seconds")
            metrics.SCANS.inc()
            log.info(
                "scan finished in %.0fs: %d queued, %d newly skipped, %d queued to regenerate",
                time.time() - t0,
                queued,
                skipped,
                regen,
            )
        except Exception:
            log.exception("scan failed")
        finally:
            self._scanning = False

    # --- processing ---

    def _work_loop(self) -> None:
        while not self._stop.is_set():
            try:
                job = self._queue.get(timeout=5)
            except queue.Empty:
                continue
            self._wait_for_window(job)
            if self._stop.is_set():
                return
            with self._lock:
                self._current = job
                self._queued.pop(str(job.path), None)
            try:
                self._process(job)
            finally:
                with self._lock:
                    self._current = None

    def _wait_for_window(self, job: Job) -> None:
        if job.bypass_window and settings.manual_bypass_window:
            return
        while not in_work_window() and not self._stop.is_set():
            if not self._paused_for_window:
                log.info(
                    "outside work window %s — pausing (queue: %d)",
                    settings.work_window,
                    self._queue.qsize() + 1,
                )
                self._paused_for_window = True
            self._stop.wait(timeout=60)
        self._paused_for_window = False

    def _process(self, job: Job) -> None:
        video = job.path
        try:
            st = video.stat()
        except OSError as exc:
            log.warning("%s vanished before processing: %s", video, exc)
            return

        if job.replaces:
            # Re-check: the owner may have replaced our sub, or added a human
            # one next to it, since the scan.
            if self._outdated_own_sub(video) != job.replaces:
                log.info("not regenerating %s: its subtitles changed since the scan", video.name)
                return
        # Re-check right before working: subs may have appeared since queueing.
        elif not job.force:
            needs, reason = check_needs_subtitles(video)
            if not needs:
                self.store.record(str(video), st.st_size, st.st_mtime, "skipped", reason=reason)
                return

        log.info(
            "processing %s (job %d, via %s, %d more queued)",
            video,
            job.id,
            job.source,
            self._queue.qsize(),
        )
        try:
            result = self.transcriber.transcribe(video, may_replace=self._may_replace(job))
        except OutputConflict as exc:
            # A sub appeared at our output path while we worked (e.g. the owner
            # downloaded one). Theirs wins; nothing was written.
            log.warning("not writing subtitles for %s: %s", video.name, exc)
            self.store.record(str(video), st.st_size, st.st_mtime, "skipped", reason=str(exc))
            return
        except ModelLoadError:
            # Not this file's fault. Record nothing (so it isn't burned against
            # MAX_RETRIES); /healthz is failing, so the pod gets restarted.
            log.error("job %d dropped: whisper model unusable", job.id)
            return
        except TranscriptionCancelled:
            # Shutdown abort: partial output already cleaned up. Leave no
            # state record so the file is picked up again after restart.
            log.info("job %d aborted by shutdown: %s", job.id, video.name)
            return
        except Exception as exc:
            log.exception("transcription failed for %s", video)
            if job.replaces:
                # Our old sub is still on disk and still ours; keep its row and
                # fingerprint, just count the failed attempt.
                self.store.record_regen_failure(str(video), f"regeneration failed: {exc}"[:500])
                metrics.FAILED.inc()
                return
            self.store.record(
                str(video),
                st.st_size,
                st.st_mtime,
                "failed",
                reason=str(exc)[:500],
                model=settings.whisper_model,
                bump_attempts=True,
            )
            metrics.FAILED.inc()
            self._notify({"status": "failed", "path": str(video), "error": str(exc)[:500]})
            return

        if job.replaces and job.replaces != result["subtitle"]:
            # Language detection changed the file name; drop our old copy so
            # Jellyfin doesn't show two tracks, but only if it's still exactly
            # what we wrote (the job took minutes; the owner may have replaced it).
            if self.store.outdated_output(str(video), PIPELINE_VERSION) == job.replaces:
                Path(job.replaces).unlink(missing_ok=True)
                log.info("removed superseded %s", job.replaces)
            else:
                log.info("keeping %s: it changed during regeneration", job.replaces)

        self.store.record(
            str(video),
            st.st_size,
            st.st_mtime,
            "done",
            language=result["language"],
            subtitle=result["subtitle"],
            model=settings.whisper_model,
            duration_s=result["duration_s"],
            elapsed_s=result["elapsed_s"],
            pipeline=result["pipeline"],
            sub_size=result["subtitle_size"],
            sub_mtime=result["subtitle_mtime"],
            qa=result["qa"],
        )
        metrics.PROCESSED.inc()
        metrics.QA_VIOLATIONS.observe(result["qa"]["violations_per_100"])
        metrics.MEDIA_SECONDS.inc(result["duration_s"])
        log.info(
            "job %d done: %s [%s] — %d files left in queue",
            job.id,
            video.name,
            result["language"],
            self._queue.qsize(),
        )
        self._notify({"status": "done", "path": str(video), **result})

    def _may_replace(self, job: Job):
        """What the transcriber may overwrite at the output path, decided at the
        moment of writing. Plain jobs: nothing. Regeneration: only our own
        outdated sub, re-verified then. /process force: anything (the owner
        asked for it explicitly)."""
        if job.force:
            return lambda _target: True
        if job.replaces:
            video, replaces = str(job.path), job.replaces
            return lambda target: (
                str(target) == replaces
                and self.store.outdated_output(video, PIPELINE_VERSION) == replaces
            )
        return None

    def _outdated_own_sub(self, video: Path) -> str | None:
        """Our sub for `video` when it's safe to regenerate: made by an older
        pipeline, untouched since, the video itself unchanged, and no other
        subtitle file sitting next to it (a human sub makes ours redundant,
        and we'd rather leave both alone than guess)."""
        try:
            st = video.stat()
        except OSError:
            return None
        row = self.store.lookup(str(video))
        if row is None or row["size"] != st.st_size or abs(row["mtime"] - st.st_mtime) > 1:
            return None
        own = self.store.outdated_output(str(video), PIPELINE_VERSION, settings.max_retries)
        if own is None:
            return None
        others = [s for s in external_subtitles(video) if str(s) != own]
        return None if others else own

    def _notify(self, payload: dict) -> None:
        if not settings.webhook_url:
            return
        try:
            req = urllib.request.Request(
                settings.webhook_url,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=10).close()
        except Exception as exc:
            log.warning("webhook delivery failed: %s", exc)


worker = Worker()
