"""HTTP API: health, status, history, and job triggers."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from .config import settings
from .worker import worker

router = APIRouter()


def require_api_key(request: Request) -> None:
    if not settings.api_key:
        return
    auth = request.headers.get("authorization", "")
    key = request.headers.get("x-api-key", "")
    if auth.startswith("Bearer "):
        key = key or auth.removeprefix("Bearer ").strip()
    if key != settings.api_key:
        raise HTTPException(status_code=401, detail="invalid or missing API key")


class ScanRequest(BaseModel):
    # Optional subset of paths to scan; defaults to all configured media dirs.
    paths: list[str] | None = None


class ProcessRequest(BaseModel):
    path: str
    # Force regeneration even if subtitles already exist.
    force: bool = False


def _validate_media_path(raw: str) -> Path:
    path = Path(raw).resolve()
    allowed = [Path(d).resolve() for d in settings.media_dir_list]
    if not any(path == root or path.is_relative_to(root) for root in allowed):
        raise HTTPException(
            status_code=400,
            detail=f"path must be inside a configured media dir: {settings.media_dir_list}",
        )
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"no such file or directory: {path}")
    return path


@router.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@router.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.get("/status", dependencies=[Depends(require_api_key)])
def status() -> dict:
    return worker.status()


@router.get("/history", dependencies=[Depends(require_api_key)])
def history(limit: int = 100, status: str | None = None) -> list[dict]:
    if status and status not in ("done", "failed", "skipped"):
        raise HTTPException(status_code=400, detail="status must be done|failed|skipped")
    return worker.store.history(limit=min(limit, 1000), status=status)


@router.post("/scan", dependencies=[Depends(require_api_key)], status_code=202)
def scan(body: ScanRequest | None = None) -> dict:
    """Trigger a library scan (e.g. from a Jellyfin post-import hook)."""
    roots = None
    if body and body.paths:
        roots = [str(_validate_media_path(p)) for p in body.paths]
    if not worker.trigger_scan(roots):
        return {"started": False, "detail": "a scan is already running"}
    return {"started": True, "paths": roots or settings.media_dir_list}


@router.post("/process", dependencies=[Depends(require_api_key)], status_code=202)
def process(body: ProcessRequest) -> dict:
    """Queue one file (or every video in a directory) for transcription."""
    target = _validate_media_path(body.path)

    if target.is_dir():
        if not worker.trigger_scan([str(target)]):
            raise HTTPException(status_code=409, detail="a scan is already running")
        return {"queued": True, "scan": str(target)}

    if target.suffix.lstrip(".").lower() not in settings.extension_set:
        raise HTTPException(status_code=400, detail=f"not a video file: {target.name}")

    if body.force:
        worker.store.forget(str(target))
    else:
        from .scanner import check_needs_subtitles

        needs, reason = check_needs_subtitles(target)
        if not needs:
            return {"queued": False, "detail": reason}

    job = worker.enqueue(target, source="api", bypass_window=True, force=body.force)
    if job is None:
        return {"queued": False, "detail": "already queued or processing"}
    return {"queued": True, "job_id": job.id, "path": str(target)}


@router.delete("/history", dependencies=[Depends(require_api_key)])
def forget(path: str) -> dict:
    """Forget a file so the next scan reconsiders it."""
    return {"forgotten": worker.store.forget(str(Path(path).resolve()))}
