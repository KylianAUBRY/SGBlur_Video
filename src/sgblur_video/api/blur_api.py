"""Blur API (FastAPI, port 8000): uploads, asynchronous jobs, results, metrics.

The contract is ``docs/design/api.md`` / ``docs/design/openapi.yaml``. This
process never runs inference: it streams uploads to disk, probes them, queues
jobs in the SQLite store and serves results; the worker does the processing.

Run it with ``sgblur-video serve`` (API + one worker) or
``uvicorn --factory sgblur_video.api.blur_api:create_app``.
"""

import asyncio
import json
import logging
import secrets
import shutil
import uuid
from collections.abc import AsyncIterator
from importlib import resources
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, Query, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    Response,
    StreamingResponse,
)
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

from sgblur_video import __version__
from sgblur_video.api.errors import PROBE_STATUS, ApiError, install_error_handler
from sgblur_video.api.metrics import render_metrics
from sgblur_video.api.upload import UploadError, receive_file, safe_suffix
from sgblur_video.config import Settings, get_settings
from sgblur_video.core.probe import UnsupportedVideoError, probe
from sgblur_video.jobs.runner import delete_job_files
from sgblur_video.jobs.store import FINISHED, Job, JobStore
from sgblur_video.jobs.worker import callback_allowed, job_status
from sgblur_video.models import RegistryError, load_registry, select_model

logger = logging.getLogger(__name__)

_SYNC_POLL_S = 0.5
#: The web page served at ``/ui`` (static; it calls the API from the browser).
_UI_PAGE = (resources.files("sgblur_video.api") / "ui.html").read_text(encoding="utf-8")
_STREAM_CHUNK = 1 << 20


def _model_summary(settings: Settings) -> dict[str, str]:
    try:
        entry = select_model(
            load_registry(settings.models_file),
            family=settings.model_family,
            available_memory_gib=None,
            name=settings.model_name,
        )
    except RegistryError:
        return {"name": "unknown", "version": "unknown"}
    return {"name": entry.name, "version": entry.version}


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the Blur API application.

    Args:
        settings: Settings (defaults to the environment).

    Returns:
        The FastAPI application.
    """
    settings = settings or get_settings()
    store = JobStore(settings.data_dir)
    settings.effective_tmp_dir.mkdir(parents=True, exist_ok=True)
    app = FastAPI(
        title="SGBlur-Video",
        version=__version__,
        summary="Privacy blurring of street-level videos for Panoramax.",
        docs_url="/docs",
        redoc_url=None,
    )
    install_error_handler(app)
    model = _model_summary(settings)
    tracker = settings.tracker_config.stem

    async def authenticate(authorization: Annotated[str | None, Header()] = None) -> None:
        if settings.api_token is None:
            return
        expected = f"Bearer {settings.api_token.get_secret_value()}"
        if authorization is None or not secrets.compare_digest(authorization, expected):
            raise ApiError(
                401, "unauthorized", "Missing or invalid bearer token.", {"WWW-Authenticate": "Bearer"}
            )

    auth = [Depends(authenticate)]

    def get_job(job_id: str) -> Job:
        try:
            uuid.UUID(job_id)
        except ValueError as exc:
            raise ApiError(404, "job_not_found", "Job not found.") from exc
        job = store.get(job_id)
        if job is None:
            raise ApiError(404, "job_not_found", "Job not found.")
        return job

    def require_results(job: Job) -> None:
        if job.status == "expired":
            raise ApiError(410, "job_expired", "Job results have expired.")
        if job.status in {"failed", "cancelled"}:
            raise ApiError(409, "job_failed", f"Job {job.status}.")
        if job.status != "succeeded":
            raise ApiError(409, "job_not_ready", "Job is still running.")

    @app.get("/", tags=["ops"])
    async def health() -> dict[str, Any]:
        """Health, version and configured model."""
        counts = await run_in_threadpool(store.counts)
        return {
            "name": settings.api_name,
            "version": __version__,
            "status": "ok",
            "model": model,
            "tracker": tracker,
            "device": settings.device,
            "queue": {"queued": counts.get("queued", 0), "running": counts.get("running", 0)},
        }

    @app.post("/blur/", tags=["blur"], dependencies=auth, status_code=202)
    async def blur(
        request: Request,
        keep: Annotated[int, Query(ge=0, le=1)] = 0,
        sync: Annotated[int, Query(ge=0, le=1)] = 0,
        frames: Annotated[int, Query(ge=0, le=1)] = 0,
        callback_url: Annotated[str | None, Query(max_length=2048)] = None,
        accept: Annotated[str | None, Header()] = None,
    ) -> Response:
        """Upload a video (multipart field ``video``) and queue it for blurring."""
        if keep and not settings.keep_enabled:
            raise ApiError(422, "keep_unavailable", "keep=1 is not available on this instance.")
        if callback_url and not callback_allowed(callback_url, settings):
            raise ApiError(422, "callback_not_allowed", "callback_url host is not allowed.")
        if sync and settings.sync_max_duration_s == 0:
            raise ApiError(422, "sync_too_long", "sync=1 is disabled on this instance.")
        counts = await run_in_threadpool(store.counts)
        if counts.get("queued", 0) >= settings.queue_max:
            raise ApiError(503, "queue_full", "Queue is full, retry later.", {"Retry-After": "60"})
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > settings.max_upload_bytes + 1_000_000:
            raise ApiError(413, "file_too_large", f"File larger than {settings.max_upload_bytes} bytes.")

        job_id = str(uuid.uuid4())
        staging = settings.effective_tmp_dir / job_id
        staging.mkdir(parents=True)
        try:
            received = await receive_file(
                request.headers.get("content-type"),
                request.stream(),
                field_name="video",
                destination=staging / "upload",
                max_bytes=settings.max_upload_bytes,
            )
            suffix = safe_suffix(received.suffix)
            input_path = (staging / "upload").rename(staging / f"input{suffix}")
            info = await run_in_threadpool(probe, input_path, settings)
            if sync and info.duration_s > settings.sync_max_duration_s:
                raise ApiError(
                    422,
                    "sync_too_long",
                    f"sync=1 is limited to videos up to {settings.sync_max_duration_s} s.",
                )
        except UploadError as exc:
            shutil.rmtree(staging, ignore_errors=True)
            raise ApiError(exc.status, exc.code, str(exc)) from exc
        except UnsupportedVideoError as exc:
            shutil.rmtree(staging, ignore_errors=True)
            raise ApiError(PROBE_STATUS.get(exc.code, 422), exc.code, str(exc)) from exc
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise

        params = {
            "keep": bool(keep),
            "frames": bool(frames),
            "sync": bool(sync),
            "callback_url": callback_url,
        }
        destination = store.jobs_dir / job_id
        staging.rename(destination)
        job = await run_in_threadpool(store.create, job_id, params, input_path.name, info.frame_count)
        logger.info("job %s queued (%d bytes, %.1f s of video)", job_id, received.size, info.duration_s)
        if not sync:
            return JSONResponse(job_status(job), status_code=202, headers={"Location": f"/jobs/{job_id}"})
        return await _sync_response(job_id, accept)

    async def _sync_response(job_id: str, accept: str | None) -> Response:
        deadline = asyncio.get_running_loop().time() + settings.job_timeout_s
        job = store.get(job_id)
        while job is not None and job.status not in FINISHED:
            if asyncio.get_running_loop().time() > deadline:
                break
            await asyncio.sleep(_SYNC_POLL_S)
            job = store.get(job_id)
        if job is None or job.status != "succeeded":
            code = job.error_code if job is not None and job.error_code else "processing_error"
            raise ApiError(
                500 if code in {"processing_error", "worker_crash"} else 422, code, "Job did not succeed."
            )
        paths = store.paths(job)

        def cleanup() -> None:
            delete_job_files(paths)
            store.mark_expired(job_id)

        if accept and "multipart/form-data" in accept:
            boundary = secrets.token_hex(16)
            metadata = paths.metadata.read_bytes()
            return StreamingResponse(
                _multipart(boundary, metadata, paths.output),
                media_type=f"multipart/form-data; boundary={boundary}",
                background=BackgroundTask(cleanup),
            )
        return FileResponse(
            paths.output,
            media_type="video/mp4",
            filename="blurred.mp4",
            headers={"X-SGBlur-Video-Job": job_id},
            background=BackgroundTask(cleanup),
        )

    @app.get("/jobs/{job_id}", tags=["jobs"], dependencies=auth)
    async def get_status(job_id: str) -> dict[str, Any]:
        """Job status and progress."""
        return job_status(await run_in_threadpool(get_job, job_id))

    @app.get("/jobs/{job_id}/video", tags=["jobs"], dependencies=auth, response_model=None)
    async def get_video(job_id: str) -> FileResponse:
        """Download the blurred video (supports HTTP range requests)."""
        job = await run_in_threadpool(get_job, job_id)
        require_results(job)
        return FileResponse(store.paths(job).output, media_type="video/mp4", filename="blurred.mp4")

    @app.get("/jobs/{job_id}/metadata", tags=["jobs"], dependencies=auth)
    async def get_metadata(job_id: str) -> JSONResponse:
        """Blurring metadata and Panoramax annotations."""
        job = await run_in_threadpool(get_job, job_id)
        require_results(job)
        return JSONResponse(json.loads(store.paths(job).metadata.read_text(encoding="utf-8")))

    @app.get("/jobs/{job_id}/frames", tags=["jobs"], dependencies=auth)
    async def list_frames(job_id: str) -> dict[str, Any]:
        """Best-frame pictures of signs (jobs created with ``frames=1``)."""
        job = await run_in_threadpool(get_job, job_id)
        require_results(job)
        index = store.paths(job).frames / "frames.json"
        if not index.exists():
            raise ApiError(404, "frames_not_requested", "This job was created without frames=1.")
        entries = json.loads(index.read_text(encoding="utf-8"))["frames"]
        for entry in entries:
            entry["url"] = f"/jobs/{job_id}/frames/{entry['n']}.jpg"
            entry.pop("file", None)
        return {"frames": entries}

    @app.get("/jobs/{job_id}/frames/{n}.jpg", tags=["jobs"], dependencies=auth, response_model=None)
    async def get_frame(job_id: str, n: int) -> FileResponse:
        """One blurred best-frame picture."""
        job = await run_in_threadpool(get_job, job_id)
        require_results(job)
        path = store.paths(job).frames / f"{n}.jpg"
        if n < 0 or not path.exists():
            raise ApiError(404, "frame_not_found", "Picture not found.")
        return FileResponse(path, media_type="image/jpeg")

    @app.delete("/jobs/{job_id}", tags=["jobs"], dependencies=auth, status_code=204)
    async def delete_job(job_id: str) -> Response:
        """Cancel the job if needed and delete all its files now."""
        job = await run_in_threadpool(get_job, job_id)
        if job.status in {"queued", "running"}:
            store.finish(job_id, "cancelled", error_code="cancelled", error_message="Deleted by the client.")
        await run_in_threadpool(delete_job_files, store.paths(job))
        store.mark_expired(job_id)
        return Response(status_code=204)

    if settings.web_ui:

        @app.get("/ui", tags=["ops"], response_class=HTMLResponse)
        async def web_ui() -> str:
            """Web page to upload a video and download the blurred result (calls this API)."""
            return _UI_PAGE

    @app.get("/metrics", tags=["ops"], response_class=PlainTextResponse)
    async def metrics() -> str:
        """Prometheus metrics computed from the job store."""
        return await run_in_threadpool(render_metrics, store)

    return app


async def _multipart(boundary: str, metadata: bytes, video: Path) -> AsyncIterator[bytes]:
    """``metadata`` (JSON) and ``video`` (MP4) parts, as SGBlur's ``metadata`` + ``image`` parts."""
    yield (
        f'--{boundary}\r\nContent-Disposition: form-data; name="metadata"; filename="metadata.json"\r\n'
        "Content-Type: application/json\r\n\r\n"
    ).encode()
    yield metadata
    yield (
        f'\r\n--{boundary}\r\nContent-Disposition: form-data; name="video"; filename="blurred.mp4"\r\n'
        "Content-Type: video/mp4\r\n\r\n"
    ).encode()
    with video.open("rb") as handle:
        while chunk := await run_in_threadpool(handle.read, _STREAM_CHUNK):
            yield chunk
    yield f"\r\n--{boundary}--\r\n".encode()
