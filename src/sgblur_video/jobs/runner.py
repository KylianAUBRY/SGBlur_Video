"""Run one job: analysis (local or remote), post-processing, rendering, metadata, cleanup.

:func:`run_job` is the entry point of the per-job child process started by the
worker. Whatever happens, the original upload is deleted before it returns;
on failure every file of the job is deleted.
"""

import json
import logging
import shutil
import time
import uuid
from collections.abc import Callable
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import httpx

from sgblur_video.config import Settings
from sgblur_video.core.detections_io import DetectionsFormatError
from sgblur_video.core.pipeline import load_detections, render_detections, write_metadata
from sgblur_video.core.probe import UnsupportedVideoError, VideoInfo, probe
from sgblur_video.core.render import RenderError
from sgblur_video.core.trim import FrameRange, trim_video
from sgblur_video.jobs.analysis import AnalysisKilledError, run_analysis
from sgblur_video.jobs.store import JobPaths, JobStore, Phase, utcnow
from sgblur_video.privacy.keep import KeepRecorder

logger = logging.getLogger(__name__)

#: Share of the job duration spent in analysis, used to compute the overall percentage.
ANALYSIS_SHARE = 0.7
_PROGRESS_INTERVAL_S = 1.0


class RemoteDetectError(RuntimeError):
    """The remote Detect API failed or returned an invalid stream."""


#: Error message of a job whose process was killed by the system.
KILLED_MESSAGE = (
    "Processing was killed by the system, most likely for lack of memory: give the service more "
    "memory, or send a shorter range or a smaller video."
)


class FrameRangeError(ValueError):
    """The requested frame range holds no frame of the video."""


class Progress:
    """Throttled progress reporter writing to the job store (also used by the analysis process)."""

    def __init__(self, store: JobStore, job_id: str) -> None:
        self._store = store
        self._job_id = job_id
        self._last = 0.0
        self._phase_started = time.monotonic()
        self._phase: Phase = "analyzing"

    def phase(self, phase: Phase) -> None:
        """Start a phase (progress then counts within it)."""
        self._phase = phase
        self._phase_started = time.monotonic()
        self._store.heartbeat(self._job_id)

    def __call__(self, done: int, total: int | None) -> None:
        """Record ``done`` of ``total`` frames of the current phase (at most once a second)."""
        now = time.monotonic()
        if now - self._last < _PROGRESS_INTERVAL_S and not (total and done >= total):
            return
        self._last = now
        fraction = min(1.0, done / total) if total else 0.0
        base, share = (
            (0.0, ANALYSIS_SHARE) if self._phase == "analyzing" else (ANALYSIS_SHARE, 1 - ANALYSIS_SHARE)
        )
        elapsed = now - self._phase_started
        eta = round(elapsed / done * (total - done)) if total and done else None
        self._store.progress(
            self._job_id,
            phase=self._phase,
            frames_done=done,
            frames_total=total,
            percent=100 * (base + share * fraction),
            eta_s=eta,
        )


def remote_detect(
    info: VideoInfo,
    output: Path,
    settings: Settings,
    progress: Callable[[int, int | None], None],
    *,
    client: httpx.Client | None = None,
) -> None:
    """Run pass 1 on the remote Detect API and save the streamed ``detections.jsonl``.

    Retries while the detector is busy (``503``, honouring ``Retry-After``).

    Raises:
        RemoteDetectError: On HTTP errors or an incomplete stream.
    """
    assert settings.detect_url is not None  # noqa: S101 - caller checks
    url = f"{str(settings.detect_url).rstrip('/')}/detect/"
    http = client or httpx.Client(timeout=httpx.Timeout(30.0, read=600.0))
    try:
        for attempt in range(60):
            with (
                info.path.open("rb") as video,
                http.stream(
                    "POST", url, files={"video": (f"input{info.path.suffix}", video, "video/mp4")}
                ) as response,
            ):
                if response.status_code == 503:
                    delay = float(response.headers.get("Retry-After", "10"))
                    logger.info("detector busy, retry %d in %.0f s", attempt + 1, delay)
                    time.sleep(min(delay, 60.0))
                    continue
                if response.status_code != 200:
                    response.read()
                    raise RemoteDetectError(f"Detect API returned {response.status_code}")
                frames = 0
                complete = False
                with output.open("w", encoding="utf-8") as handle:
                    for line in response.iter_lines():
                        if not line.strip():
                            continue
                        handle.write(line + "\n")
                        kind = json.loads(line).get("type")
                        if kind == "frame":
                            frames += 1
                            progress(frames, info.frame_count)
                        elif kind == "footer":
                            complete = True
                if not complete:
                    raise RemoteDetectError("Detect API stream ended without a footer")
                return
        raise RemoteDetectError("Detect API stayed busy")
    except httpx.HTTPError as exc:
        raise RemoteDetectError(f"Detect API unreachable: {exc}") from exc
    finally:
        if client is None:
            http.close()


def delete_job_files(paths: JobPaths) -> None:
    """Delete every file of a job (best effort, never raises)."""
    shutil.rmtree(paths.root, ignore_errors=True)


def run_job(job_id: str, settings: Settings, store: JobStore | None = None) -> bool:
    """Process a claimed job. Returns True on success.

    Args:
        job_id: Job to run (already marked ``running`` by the worker).
        settings: Settings.
        store: Job store (defaults to the one in ``DATA_DIR``).
    """
    store = store or JobStore(settings.data_dir)
    job = store.get(job_id)
    if job is None or job.status != "running":
        return False
    paths = store.paths(job)
    progress = Progress(store, job_id)
    started = time.monotonic()
    frame_range = FrameRange(
        start=int(job.params.get("start_frame") or 0),
        end=int(job.params["end_frame"]) if job.params.get("end_frame") is not None else None,
    )
    try:
        info = probe(paths.input, settings)
        progress.phase("analyzing")
        if not frame_range.is_whole:
            try:
                trim_video(info, settings, frame_range, paths.range_input)
            except RenderError as exc:
                raise FrameRangeError(str(exc)) from exc
            paths.input.unlink(missing_ok=True)  # only the requested range is kept
            # The output is encoded at the bit rate of the original, not of the near-lossless cut.
            info = replace(probe(paths.range_input, settings), bit_rate=info.bit_rate)
        if settings.detect_url is not None:
            remote_detect(info, paths.detections, settings, progress)
        else:
            # In a process of its own: its memory is returned to the system before rendering.
            run_analysis(job_id, info, settings, paths.detections)
        detections = load_detections(paths.detections, info, allow_partial=False)
        progress.phase("rendering")
        blurring_id = str(uuid.uuid4())
        keep = bool(job.params.get("keep")) and settings.keep_secret_key is not None
        result = render_detections(
            info,
            detections,
            paths.output,
            settings,
            debug_output=paths.debug if job.params.get("debug") else None,
            frames_dir=paths.frames if job.params.get("frames") else None,
            progress=progress,
            region_sink_factory=(
                (lambda plan: KeepRecorder(plan.chain_scores, settings.keep_max_confidence)) if keep else None
            ),
            blurring_id=blurring_id,
        )
        progress.phase("finalizing")
        if keep and isinstance(result.region_sink, KeepRecorder):
            assert settings.keep_secret_key is not None  # noqa: S101 - checked above
            result.region_sink.save(
                settings.effective_keep_dir,
                secret=settings.keep_secret_key.get_secret_value(),
                blurring_id=blurring_id,
            )
            result.metadata.stats["kept_regions"] = result.region_sink.regions
        result.metadata.stats["processing_s"] = round(time.monotonic() - started, 1)
        if not frame_range.is_whole:
            # Timestamps and frame numbers of the annotations count from the start of the range.
            result.metadata.video["frame_range"] = {"start": frame_range.start, "end": frame_range.end}
        write_metadata(result.metadata, paths.metadata)
        paths.input.unlink(missing_ok=True)
        paths.range_input.unlink(missing_ok=True)
        paths.detections.unlink(missing_ok=True)
        store.finish(
            job_id,
            "succeeded",
            expires_at=utcnow() + timedelta(minutes=settings.result_ttl_minutes),
            stats={
                "frames": result.render.frames,
                "elapsed_s": round(time.monotonic() - started, 1),
                "fps": round(result.render.frames / max(1e-6, time.monotonic() - started), 2),
                "signs": len(result.metadata.annotations),
                "blurred_frames": result.render.blurred_frames,
            },
        )
        logger.info("job %s succeeded in %.1f s", job_id, time.monotonic() - started)
        return True
    except UnsupportedVideoError as exc:
        _fail(store, job_id, paths, exc.code, str(exc))
    except (RemoteDetectError, DetectionsFormatError) as exc:
        _fail(store, job_id, paths, "detection_failed", str(exc))
    except FrameRangeError as exc:
        _fail(store, job_id, paths, "invalid_parameter", str(exc))
    except AnalysisKilledError:
        _fail(store, job_id, paths, "worker_crash", KILLED_MESSAGE)
    except Exception as exc:
        logger.exception("job %s failed", job_id)
        _fail(store, job_id, paths, "processing_error", type(exc).__name__)
    return False


def _fail(store: JobStore, job_id: str, paths: JobPaths, code: str, message: str) -> None:
    delete_job_files(paths)
    store.finish(job_id, "failed", error_code=code, error_message=message)
