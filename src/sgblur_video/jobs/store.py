"""SQLite job store (``docs/adr/0005-job-queue.md``).

One table, ``jobs``, in ``DATA_DIR/jobs.sqlite`` (WAL mode). The API inserts
and reads jobs; the worker claims them atomically and records progress. Rows
hold no personal data: ids, parameters, counters, timings and error codes.
Per-job files live in ``DATA_DIR/jobs/<id>/`` (see :class:`JobPaths`).
"""

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

Status = Literal["queued", "running", "succeeded", "failed", "cancelled", "expired"]
Phase = Literal["analyzing", "postprocessing", "rendering", "finalizing"]
FINISHED: frozenset[str] = frozenset({"succeeded", "failed", "cancelled", "expired"})

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    phase TEXT,
    params TEXT NOT NULL,
    input_name TEXT NOT NULL,
    frames_done INTEGER NOT NULL DEFAULT 0,
    frames_total INTEGER,
    percent REAL NOT NULL DEFAULT 0,
    eta_s INTEGER,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    expires_at TEXT,
    heartbeat_at TEXT,
    error_code TEXT,
    error_message TEXT,
    stats TEXT
);
CREATE INDEX IF NOT EXISTS jobs_status_created ON jobs (status, created_at);
"""


def utcnow() -> datetime:
    """Current UTC time (patched in tests)."""
    return datetime.now(UTC)


def _iso(moment: datetime | None) -> str | None:
    return moment.isoformat(timespec="seconds") if moment is not None else None


def _parse(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


@dataclass(frozen=True)
class JobPaths:
    """Files of one job inside ``DATA_DIR/jobs/<id>/``."""

    root: Path
    input_name: str = "input.mp4"

    @property
    def input(self) -> Path:
        """Uploaded original video (deleted as soon as rendering ends)."""
        return self.root / self.input_name

    @property
    def detections(self) -> Path:
        """``detections.jsonl``."""
        return self.root / "detections.jsonl"

    @property
    def output(self) -> Path:
        """Blurred video."""
        return self.root / "output.mp4"

    @property
    def metadata(self) -> Path:
        """Metadata JSON (annotations, statistics)."""
        return self.root / "metadata.json"

    @property
    def frames(self) -> Path:
        """Best-frame pictures folder."""
        return self.root / "frames"


@dataclass(frozen=True)
class Job:
    """A job row."""

    id: str
    status: Status
    phase: Phase | None
    params: dict[str, Any]
    input_name: str
    frames_done: int
    frames_total: int | None
    percent: float
    eta_s: int | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    expires_at: datetime | None
    heartbeat_at: datetime | None
    error_code: str | None
    error_message: str | None
    stats: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Job:
        """Build a job from a database row."""
        created = _parse(row["created_at"])
        assert created is not None  # noqa: S101 - NOT NULL column
        return cls(
            id=row["id"],
            status=row["status"],
            phase=row["phase"],
            params=json.loads(row["params"]),
            input_name=row["input_name"],
            frames_done=row["frames_done"],
            frames_total=row["frames_total"],
            percent=row["percent"],
            eta_s=row["eta_s"],
            created_at=created,
            started_at=_parse(row["started_at"]),
            finished_at=_parse(row["finished_at"]),
            expires_at=_parse(row["expires_at"]),
            heartbeat_at=_parse(row["heartbeat_at"]),
            error_code=row["error_code"],
            error_message=row["error_message"],
            stats=json.loads(row["stats"]) if row["stats"] else {},
        )


class JobStore:
    """Access to the job table. Every method opens its own short connection (safe across processes).

    Args:
        data_dir: ``DATA_DIR``; the database is ``jobs.sqlite`` inside it.
    """

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.jobs_dir = data_dir / "jobs"
        self.path = data_dir / "jobs.sqlite"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
        finally:
            connection.close()

    def paths(self, job: Job | str, input_name: str = "input.mp4") -> JobPaths:
        """Files of a job."""
        if isinstance(job, Job):
            return JobPaths(self.jobs_dir / job.id, job.input_name)
        return JobPaths(self.jobs_dir / job, input_name)

    def create(self, job_id: str, params: dict[str, Any], input_name: str, frames_total: int | None) -> Job:
        """Insert a queued job."""
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO jobs (id, status, params, input_name, frames_total, created_at)"
                " VALUES (?, 'queued', ?, ?, ?, ?)",
                (job_id, json.dumps(params), input_name, frames_total, _iso(utcnow())),
            )
        job = self.get(job_id)
        assert job is not None  # noqa: S101 - just inserted
        return job

    def get(self, job_id: str) -> Job | None:
        """Return a job, or None if unknown."""
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return Job.from_row(row) if row else None

    def claim(self) -> Job | None:
        """Atomically take the oldest queued job and mark it running."""
        now = _iso(utcnow())
        with self._connect() as connection:
            row = connection.execute(
                "UPDATE jobs SET status = 'running', phase = 'analyzing', started_at = ?, heartbeat_at = ?"
                " WHERE id = (SELECT id FROM jobs WHERE status = 'queued' ORDER BY created_at, rowid LIMIT 1)"
                " RETURNING *",
                (now, now),
            ).fetchone()
        return Job.from_row(row) if row else None

    def progress(
        self,
        job_id: str,
        *,
        phase: Phase,
        frames_done: int,
        frames_total: int | None,
        percent: float,
        eta_s: int | None,
    ) -> None:
        """Record progress (also refreshes the heartbeat). Ignored if the job is no longer running."""
        with self._connect() as connection:
            connection.execute(
                "UPDATE jobs SET phase = ?, frames_done = ?, frames_total = COALESCE(?, frames_total),"
                " percent = ?, eta_s = ?, heartbeat_at = ? WHERE id = ? AND status = 'running'",
                (phase, frames_done, frames_total, round(percent, 1), eta_s, _iso(utcnow()), job_id),
            )

    def heartbeat(self, job_id: str) -> None:
        """Mark a running job as alive."""
        with self._connect() as connection:
            connection.execute(
                "UPDATE jobs SET heartbeat_at = ? WHERE id = ? AND status = 'running'",
                (_iso(utcnow()), job_id),
            )

    def finish(
        self,
        job_id: str,
        status: Status,
        *,
        error_code: str | None = None,
        error_message: str | None = None,
        expires_at: datetime | None = None,
        stats: dict[str, Any] | None = None,
    ) -> None:
        """Move a job to a final status (no effect on jobs already finished)."""
        with self._connect() as connection:
            connection.execute(
                "UPDATE jobs SET status = ?, phase = NULL, finished_at = ?, expires_at = ?, error_code = ?,"
                " error_message = ?, stats = COALESCE(?, stats),"
                " percent = CASE WHEN ? = 'succeeded' THEN 100 ELSE percent END, eta_s = NULL"
                " WHERE id = ? AND status IN ('queued', 'running')",
                (
                    status,
                    _iso(utcnow()),
                    _iso(expires_at),
                    error_code,
                    error_message,
                    json.dumps(stats) if stats is not None else None,
                    status,
                    job_id,
                ),
            )

    def mark_expired(self, job_id: str) -> None:
        """Results deleted after their time-to-live."""
        with self._connect() as connection:
            connection.execute(
                "UPDATE jobs SET status = 'expired' WHERE id = ? AND status = 'succeeded'", (job_id,)
            )

    def counts(self) -> dict[str, int]:
        """Number of jobs per status."""
        with self._connect() as connection:
            rows = connection.execute("SELECT status, COUNT(*) AS n FROM jobs GROUP BY status").fetchall()
        return {row["status"]: row["n"] for row in rows}

    def expired_results(self) -> list[Job]:
        """Succeeded jobs whose results passed their time-to-live."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM jobs WHERE status = 'succeeded' AND expires_at < ?", (_iso(utcnow()),)
            ).fetchall()
        return [Job.from_row(row) for row in rows]

    def stale_running(self, older_than: timedelta) -> list[Job]:
        """Running jobs whose heartbeat stopped (crashed worker)."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM jobs WHERE status = 'running' AND heartbeat_at < ?",
                (_iso(utcnow() - older_than),),
            ).fetchall()
        return [Job.from_row(row) for row in rows]

    def purge_rows(self, older_than: timedelta) -> int:
        """Delete rows of jobs finished long ago; returns the number of rows deleted."""
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM jobs WHERE status IN ('failed', 'cancelled', 'expired') AND finished_at < ?",
                (_iso(utcnow() - older_than),),
            )
        return cursor.rowcount

    def all_ids(self) -> set[str]:
        """Ids of every job row."""
        with self._connect() as connection:
            return {row["id"] for row in connection.execute("SELECT id FROM jobs").fetchall()}

    def finished_jobs(self) -> list[Job]:
        """Jobs in a final status (for metrics)."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM jobs WHERE status IN ('succeeded', 'failed', 'cancelled', 'expired')"
            ).fetchall()
        return [Job.from_row(row) for row in rows]
