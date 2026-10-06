"""Detect API (FastAPI, port 8001): video in, ``detections.jsonl`` streamed out.

Used when detection runs on another (GPU) machine than the Blur API and its
worker (``DETECT_URL``). One video at a time per process: a second request
gets ``503 detector_busy`` with ``Retry-After`` and the worker retries. The
analysis runs in a child process (tracker ids are process-global) and its
output lines are streamed as they are written. The received video and the
analysis files are deleted as soon as the response ends, including when the
client disconnects.

Run it with ``sgblur-video serve-detect`` or
``uvicorn --factory sgblur_video.api.detect_api:create_app --port 8001``.
"""

import asyncio
import logging
import multiprocessing
import shutil
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

from sgblur_video import __version__
from sgblur_video.api.errors import PROBE_STATUS, ApiError, install_error_handler
from sgblur_video.api.upload import UploadError, receive_file, safe_suffix
from sgblur_video.config import Settings, get_settings
from sgblur_video.core.probe import UnsupportedVideoError, probe

logger = logging.getLogger(__name__)

_TAIL_POLL_S = 0.2


def _analyse(video: Path, output: Path, settings: Settings) -> None:
    """Child process: pass 1 only (imports the model here, not in the API process)."""
    from sgblur_video.core.pipeline import run_detect

    logging.basicConfig(level=settings.log_level)
    run_detect(video, output, settings)


async def _tail(
    path: Path, process: multiprocessing.process.BaseProcess, staging: Path, lock: asyncio.Lock
) -> AsyncIterator[bytes]:
    """Stream the lines of ``path`` while the child writes it; clean up at the end."""
    try:
        while not path.exists() and process.is_alive():
            await asyncio.sleep(_TAIL_POLL_S)
        if not path.exists():
            return
        with path.open("rb") as handle:
            pending = b""
            while True:
                chunk = await run_in_threadpool(handle.readline)
                if chunk:
                    pending += chunk
                    if pending.endswith(b"\n"):
                        yield pending
                        if b'"type": "footer"' in pending:
                            return
                        pending = b""
                    continue
                if not process.is_alive():
                    rest = await run_in_threadpool(handle.read)
                    if rest:
                        yield pending + rest
                    return
                await asyncio.sleep(_TAIL_POLL_S)
    finally:
        if process.is_alive():
            process.kill()
        process.join(timeout=5)
        shutil.rmtree(staging, ignore_errors=True)
        lock.release()


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the Detect API application.

    Args:
        settings: Settings (defaults to the environment).

    Returns:
        The FastAPI application.
    """
    settings = settings or get_settings()
    settings.effective_tmp_dir.mkdir(parents=True, exist_ok=True)
    app = FastAPI(title="SGBlur-Video Detect API", version=__version__, redoc_url=None)
    install_error_handler(app)
    lock = asyncio.Lock()
    context = multiprocessing.get_context("spawn")

    @app.get("/", tags=["ops"])
    async def health() -> dict[str, Any]:
        """Health and current load."""
        return {"name": f"{settings.api_name} Detect API", "version": __version__, "busy": lock.locked()}

    @app.post("/detect/", tags=["detect"])
    async def detect(request: Request) -> StreamingResponse:
        """Upload a video (multipart field ``video``); stream ``detections.jsonl`` back (NDJSON)."""
        if lock.locked():
            raise ApiError(503, "detector_busy", "Detector busy, retry later.", {"Retry-After": "30"})
        await lock.acquire()
        staging = settings.effective_tmp_dir / f"detect-{uuid.uuid4()}"
        try:
            staging.mkdir(parents=True)
            received = await receive_file(
                request.headers.get("content-type"),
                request.stream(),
                field_name="video",
                destination=staging / "upload",
                max_bytes=settings.max_upload_bytes,
            )
            suffix = safe_suffix(received.suffix)
            video = (staging / "upload").rename(staging / f"input{suffix}")
            await run_in_threadpool(probe, video, settings)
            output = staging / "detections.jsonl"
            process = context.Process(target=_analyse, args=(video, output, settings), daemon=True)
            process.start()
        except UploadError as exc:
            shutil.rmtree(staging, ignore_errors=True)
            lock.release()
            raise ApiError(exc.status, exc.code, str(exc)) from exc
        except UnsupportedVideoError as exc:
            shutil.rmtree(staging, ignore_errors=True)
            lock.release()
            raise ApiError(PROBE_STATUS.get(exc.code, 422), exc.code, str(exc)) from exc
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            lock.release()
            raise
        logger.info("detection started (%d bytes)", received.size)
        return StreamingResponse(_tail(output, process, staging, lock), media_type="application/x-ndjson")

    @app.get("/metrics", tags=["ops"], response_class=PlainTextResponse)
    async def metrics() -> str:
        """Prometheus metrics of the detector."""
        return (
            "# HELP sgblur_video_detector_busy Whether a video is being analysed.\n"
            "# TYPE sgblur_video_detector_busy gauge\n"
            f"sgblur_video_detector_busy {int(lock.locked())}\n"
        )

    return app
