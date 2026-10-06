---
status: proposed
date: 2026-10-06
---

# Job queue: SQLite and worker processes, no Redis

## Context and Problem Statement

A video takes minutes to tens of minutes. The API must be asynchronous (job id,
progress, result download) and must never run inference in the HTTP process.
Which queue?

## Decision Drivers

- `docker compose up` and a native macOS install must work with minimal moving parts.
- One accelerator per host ⇒ one job at a time per worker in practice.
- Each job must run in a fresh process (Ultralytics tracker ids are process-global; memory of 8K pipelines must be released; crashes must not kill the worker).
- Progress and timings must be visible to the API and to `/metrics`.
- Panoramax contributors know Python and PostgreSQL, not necessarily Celery.

## Considered Options

1. Redis + RQ / arq / Celery / Dramatiq.
2. PostgreSQL-based queue (Procrastinate, or a table like the Panoramax backend).
3. SQLite (WAL) job table + `sgblur-video worker` processes.
4. In-process background tasks of FastAPI.

## Decision Outcome

Chosen option: **3**. A `jobs` table in `DATA_DIR/jobs.sqlite` (WAL mode);
the worker claims the oldest queued job with an atomic
`UPDATE … RETURNING`, spawns a child process (multiprocessing `spawn`) per job
with a hard timeout, and records heartbeats. A janitor (in the worker) fails
jobs with stale heartbeats, deletes their files, and purges expired results
and `keep=1` regions. Metrics are computed from the table.

### Consequences

- Good: no extra service; works identically natively and in Docker; trivially inspectable with `sqlite3`.
- Good: per-job process isolation solves tracker-id sharing, memory release and crash containment.
- Bad: API and worker must share a filesystem (same host or shared volume); horizontal scaling across hosts is limited to remote detection (`DETECT_URL`). Acceptable for Panoramax-sized instances; the store sits behind a small interface (`jobs/store.py`) so a PostgreSQL implementation can be added later.
- Option 4 rejected: a crash or restart of the API would lose running jobs, and inference would compete with request handling.
