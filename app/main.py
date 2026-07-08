"""Application entrypoint: FastAPI app + background worker."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from . import metrics
from .api import router
from .config import settings
from .worker import worker

logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s %(levelname)s [%(threadName)s] %(name)s: %(message)s",
)


class _QuietProbes(logging.Filter):
    """Keep k8s probe and scrape spam out of the access log."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return "/healthz" not in msg and "/metrics" not in msg


logging.getLogger("uvicorn.access").addFilter(_QuietProbes())


@asynccontextmanager
async def lifespan(app: FastAPI):
    metrics.QUEUE_LENGTH.set_function(lambda: worker.status()["queue_length"])
    metrics.PROCESSING.set_function(
        lambda: 1 if worker.transcriber.progress.snapshot()["path"] else 0
    )
    metrics.CURRENT_PROGRESS.set_function(
        lambda: worker.transcriber.progress.snapshot()["percent"]
    )
    worker.start()
    yield
    worker.stop()
    # abort: the active job notices the cancel flag within a few segments.
    # finish: block until the file completes — the pod's grace period
    # (SIGKILL) is the hard ceiling.
    worker.join(timeout=60 if settings.shutdown_mode == "abort" else None)
    logging.getLogger(__name__).info("shutdown complete")


app = FastAPI(
    title="whisper-sub-gen",
    description="Whisper-powered subtitle generation for Jellyfin media libraries",
    lifespan=lifespan,
)
app.include_router(router)


def run() -> None:
    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port, log_config=None)


if __name__ == "__main__":
    run()
