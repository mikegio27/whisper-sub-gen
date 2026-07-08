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


@asynccontextmanager
async def lifespan(app: FastAPI):
    metrics.QUEUE_LENGTH.set_function(lambda: worker.status()["queue_length"])
    worker.start()
    yield
    worker.stop()


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
