"""Job worker: claims queued jobs and runs each one in a fresh child process.

Why one process per job (``docs/adr/0005-job-queue.md``): Ultralytics tracker
ids are process-global, 8K pipelines use a lot of memory that must be given
back, and a crash or a hard timeout must not take the worker down.

The worker also runs the **janitor** (privacy retention rules of
``docs/design/architecture.md``):

* results past ``RESULT_TTL_MINUTES`` are deleted and their job marked ``expired``;
* running jobs whose heartbeat stopped (crashed worker) are failed and their files deleted;
* ``keep=1`` archives past ``KEEP_TTL_HOURS`` are deleted;
* job folders without a database row, and old temporary files, are deleted;
* rows of jobs finished more than a day ago are purged.
"""

import logging
import multiprocessing
import multiprocessing.context
import shutil
import signal
import threading
import time
from datetime import timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from sgblur_video.config import Settings
from sgblur_video.jobs.runner import KILLED_MESSAGE, delete_job_files, run_job
from sgblur_video.jobs.store import Job, JobStore
from sgblur_video.privacy.keep import purge_expired

logger = logging.getLogger(__name__)

HEARTBEAT_TIMEOUT = timedelta(minutes=5)
ROWS_RETENTION = timedelta(days=1)
JANITOR_INTERVAL_S = 60.0
TMP_RETENTION_S = 3600.0
ORPHAN_GRACE_S = 60.0


def job_status(job: Job) -> dict[str, Any]:
    """Public JSON representation of a job (``GET /jobs/{id}``, callbacks)."""
    base = f"/jobs/{job.id}"
    return {
        "job_id": job.id,
        "status": job.status,
        "phase": job.phase,
        "progress": {
            "percent": job.percent,
            "frames_done": job.frames_done,
            "frames_total": job.frames_total,
        },
        "eta_s": job.eta_s,
        "created_at": job.created_at.isoformat(timespec="seconds"),
        "started_at": job.started_at.isoformat(timespec="seconds") if job.started_at else None,
        "finished_at": job.finished_at.isoformat(timespec="seconds") if job.finished_at else None,
        "expires_at": job.expires_at.isoformat(timespec="seconds") if job.expires_at else None,
        "error": {"code": job.error_code, "message": job.error_message} if job.error_code else None,
        "links": {"self": base, "video": f"{base}/video", "metadata": f"{base}/metadata"}
        | ({"debug": f"{base}/debug"} if job.params.get("debug") else {}),
    }


def callback_allowed(url: str, settings: Settings) -> bool:
    """Whether ``callback_url`` targets an allow-listed host over HTTP(S)."""
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and (parsed.hostname or "") in set(
        settings.callback_allowed_hosts
    )


def _child(job_id: str, settings: Settings) -> None:
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_job(job_id, settings)


class Worker:
    """Claim and run jobs.

    Args:
        settings: Settings (``DATA_DIR``, timeouts, retention).
        isolate: Run each job in a child process (``False`` only for tests).
        poll_interval: Seconds between two looks at the queue when it is empty.
    """

    def __init__(self, settings: Settings, *, isolate: bool = True, poll_interval: float = 1.0) -> None:
        self.settings = settings
        self.store = JobStore(settings.data_dir)
        self.isolate = isolate
        self.poll_interval = poll_interval
        self._context = multiprocessing.get_context("spawn")
        self._last_janitor = 0.0

    def run_forever(self, stop: threading.Event | None = None) -> None:
        """Process jobs until ``stop`` is set."""
        stop = stop or threading.Event()
        logger.info("worker started (data dir: %s)", self.settings.data_dir)
        self.janitor()
        while not stop.is_set():
            if time.monotonic() - self._last_janitor > JANITOR_INTERVAL_S:
                self.janitor()
            if not self.run_once():
                stop.wait(self.poll_interval)

    def run_once(self) -> bool:
        """Claim one queued job and run it to completion. Returns False if the queue was empty."""
        job = self.store.claim()
        if job is None:
            return False
        logger.info("job %s claimed", job.id)
        if self.isolate:
            self._run_isolated(job)
        else:
            run_job(job.id, self.settings, self.store)
        final = self.store.get(job.id)
        if final is not None:
            self._callback(final)
        return True

    def _run_isolated(self, job: Job) -> None:
        process: multiprocessing.context.SpawnProcess = self._context.Process(
            target=_child, args=(job.id, self.settings), name=f"job-{job.id[:8]}", daemon=True
        )
        process.start()
        deadline = time.monotonic() + self.settings.job_timeout_s
        while process.is_alive():
            process.join(timeout=1.0)
            current = self.store.get(job.id)
            if current is None or current.status == "cancelled":
                logger.info("job %s cancelled, stopping it", job.id)
                process.kill()
                break
            if time.monotonic() > deadline:
                logger.warning("job %s exceeded JOB_TIMEOUT_S, stopping it", job.id)
                process.kill()
                self.store.finish(
                    job.id, "failed", error_code="timeout", error_message="Job time limit exceeded."
                )
                break
            self.store.heartbeat(job.id)
        process.join(timeout=5.0)
        current = self.store.get(job.id)
        if current is not None and current.status == "running":
            # The child died without recording an outcome (crash, out of memory…).
            killed = process.exitcode == -signal.SIGKILL
            logger.warning("job %s: process ended with exit code %s", job.id, process.exitcode)
            self.store.finish(
                job.id,
                "failed",
                error_code="worker_crash",
                error_message=KILLED_MESSAGE if killed else "Processing stopped unexpectedly.",
            )
        if current is None or current.status in {"failed", "cancelled"}:
            delete_job_files(self.store.paths(job))

    def _callback(self, job: Job) -> None:
        url = job.params.get("callback_url")
        if not url or not callback_allowed(url, self.settings):
            return
        try:
            httpx.post(url, json=job_status(job), timeout=10.0)
        except httpx.HTTPError as exc:
            logger.warning("callback for job %s failed: %s", job.id, type(exc).__name__)

    def janitor(self) -> None:
        """Apply the retention rules (see module docstring)."""
        self._last_janitor = time.monotonic()
        for job in self.store.expired_results():
            delete_job_files(self.store.paths(job))
            self.store.mark_expired(job.id)
        for job in self.store.stale_running(HEARTBEAT_TIMEOUT):
            logger.warning("job %s lost its worker, failing it", job.id)
            delete_job_files(self.store.paths(job))
            self.store.finish(
                job.id, "failed", error_code="worker_crash", error_message="Processing stopped."
            )
        purge_expired(self.settings.effective_keep_dir, self.settings.keep_ttl_hours)
        known = self.store.all_ids()
        for folder in self.store.jobs_dir.iterdir():
            row = self.store.get(folder.name) if folder.name in known else None
            # The API moves an upload into place just before inserting its row: give it a grace period.
            orphan = row is None and _age_s(folder) > ORPHAN_GRACE_S
            if orphan or (row is not None and row.status in {"failed", "cancelled", "expired"}):
                shutil.rmtree(folder, ignore_errors=True)
        tmp = self.settings.effective_tmp_dir
        if tmp.exists():
            for path in tmp.iterdir():
                if _age_s(path) > TMP_RETENTION_S:
                    if path.is_dir():
                        shutil.rmtree(path, ignore_errors=True)
                    else:
                        path.unlink(missing_ok=True)
        self.store.purge_rows(ROWS_RETENTION)


def _age_s(path: Path) -> float:
    """Seconds since anything in ``path`` was last modified (uploads in progress stay young)."""
    newest = path.stat().st_mtime
    if path.is_dir():
        newest = max([newest, *(child.stat().st_mtime for child in path.rglob("*"))])
    return time.time() - newest
