"""Tests of the SQLite job store."""

from datetime import timedelta
from pathlib import Path

import pytest

from sgblur_video.jobs import store as store_module
from sgblur_video.jobs.store import JobStore


@pytest.fixture
def store(tmp_path: Path) -> JobStore:
    return JobStore(tmp_path)


def test_lifecycle(store: JobStore) -> None:
    job = store.create("a", {"keep": False}, "input.mp4", 100)
    assert (job.status, job.frames_total, job.params) == ("queued", 100, {"keep": False})
    claimed = store.claim()
    assert claimed is not None
    assert (claimed.id, claimed.status, claimed.phase) == ("a", "running", "analyzing")
    assert store.claim() is None
    store.progress("a", phase="rendering", frames_done=50, frames_total=100, percent=85.0, eta_s=12)
    running = store.get("a")
    assert running is not None
    assert (running.phase, running.frames_done, running.percent, running.eta_s) == ("rendering", 50, 85.0, 12)
    store.finish("a", "succeeded", stats={"frames": 100})
    done = store.get("a")
    assert done is not None
    assert (done.status, done.percent, done.stats, done.phase) == ("succeeded", 100, {"frames": 100}, None)
    store.finish("a", "failed")  # no effect once finished
    assert store.get("a").status == "succeeded"  # type: ignore[union-attr]


def test_claims_oldest_first_and_counts(store: JobStore) -> None:
    for name in ("first", "second", "third"):
        store.create(name, {}, "input.mp4", None)
    assert store.claim().id == "first"  # type: ignore[union-attr]
    assert store.counts() == {"queued": 2, "running": 1}


def test_cancelled_job_ignores_late_progress(store: JobStore) -> None:
    store.create("a", {}, "input.mp4", 10)
    store.claim()
    store.finish("a", "cancelled")
    store.progress("a", phase="rendering", frames_done=5, frames_total=10, percent=50, eta_s=1)
    store.finish("a", "succeeded")
    job = store.get("a")
    assert job is not None
    assert (job.status, job.frames_done) == ("cancelled", 0)


def test_expiry_stale_and_purge(store: JobStore, monkeypatch: pytest.MonkeyPatch) -> None:
    now = store_module.utcnow()
    store.create("done", {}, "input.mp4", 1)
    store.claim()
    store.finish("done", "succeeded", expires_at=now + timedelta(minutes=1))
    store.create("crashed", {}, "input.mp4", 1)
    store.claim()
    assert store.expired_results() == []
    later = now + timedelta(minutes=10)
    monkeypatch.setattr(store_module, "utcnow", lambda: later)
    assert [j.id for j in store.expired_results()] == ["done"]
    assert [j.id for j in store.stale_running(timedelta(minutes=5))] == ["crashed"]
    store.mark_expired("done")
    store.finish("crashed", "failed", error_code="worker_crash")
    monkeypatch.setattr(store_module, "utcnow", lambda: later + timedelta(days=2))
    assert store.purge_rows(timedelta(days=1)) == 2
    assert store.all_ids() == set()


def test_paths(store: JobStore) -> None:
    job = store.create("a", {}, "input.mov", None)
    paths = store.paths(job)
    assert paths.input.name == "input.mov"
    assert paths.root == store.jobs_dir / "a"
    assert {paths.output.name, paths.metadata.name, paths.frames.name} == {
        "output.mp4",
        "metadata.json",
        "frames",
    }
