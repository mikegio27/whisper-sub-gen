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
from .scanner import check_needs_subtitles, find_videos
from .state import StateStore
from .transcriber import Transcriber, TranscriptionCancelled

log = logging.getLogger(__name__)


@dataclass
class Job:
    id: int
    path: Path
    source: str  # scan | api
    bypass_window: bool = False
    force: bool = False  # regenerate even if subtitles already exist
    queued_at: float = field(default_factory=time.time)


class Worker:
    def __init__(self) -> None:
        self.store = StateStore(settings.state_dir)
        self.transcriber = Transcriber()
        self._queue: queue.Queue[Job] = queue.Queue()
        self._queued_paths: set[str] = set()
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
        self._threads = [
            threading.Thread(target=self._work_loop, name="worker", daemon=True)
        ]
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
                "shutdown: %s active job %s (%d queued jobs dropped, "
                "re-queued on next scan)",
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
    ) -> Job | None:
        with self._lock:
            key = str(path)
            if key in self._queued_paths or (
                self._current and str(self._current.path) == key
            ):
                return None
            job = Job(
                id=next(self._job_ids),
                path=path,
                source=source,
                bypass_window=bypass_window,
                force=force,
            )
            self._queued_paths.add(key)
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
            pending = sorted(self._queued_paths)
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
            queued = skipped = 0
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
                    continue
                needs, reason = check_needs_subtitles(video)
                if not needs:
                    self.store.record(
                        str(video), st.st_size, st.st_mtime, "skipped", reason=reason
                    )
                    skipped += 1
                    continue
                if self.enqueue(video, source="scan"):
                    queued += 1
            self._last_scan = datetime.now().astimezone().isoformat(timespec="seconds")
            metrics.SCANS.inc()
            log.info(
                "scan finished in %.0fs: %d queued, %d newly skipped",
                time.time() - t0,
                queued,
                skipped,
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
                self._queued_paths.discard(str(job.path))
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

        # Re-check right before working: subs may have appeared since queueing.
        if not job.force:
            needs, reason = check_needs_subtitles(video)
            if not needs:
                self.store.record(str(video), st.st_size, st.st_mtime, "skipped", reason=reason)
                return

        log.info(
            "processing %s (job %d, via %s, %d more queued)",
            video, job.id, job.source, self._queue.qsize(),
        )
        try:
            result = self.transcriber.transcribe(video)
        except TranscriptionCancelled:
            # Shutdown abort: partial output already cleaned up. Leave no
            # state record so the file is picked up again after restart.
            log.info("job %d aborted by shutdown: %s", job.id, video.name)
            return
        except Exception as exc:
            log.exception("transcription failed for %s", video)
            self.store.record(
                str(video), st.st_size, st.st_mtime, "failed",
                reason=str(exc)[:500], model=settings.whisper_model,
                bump_attempts=True,
            )
            metrics.FAILED.inc()
            self._notify({"status": "failed", "path": str(video), "error": str(exc)[:500]})
            return

        self.store.record(
            str(video), st.st_size, st.st_mtime, "done",
            language=result["language"], subtitle=result["subtitle"],
            model=settings.whisper_model,
            duration_s=result["duration_s"], elapsed_s=result["elapsed_s"],
        )
        metrics.PROCESSED.inc()
        metrics.MEDIA_SECONDS.inc(result["duration_s"])
        log.info(
            "job %d done: %s [%s] — %d files left in queue",
            job.id, video.name, result["language"], self._queue.qsize(),
        )
        self._notify({"status": "done", "path": str(video), **result})

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
