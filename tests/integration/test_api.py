"""Blur API and worker end to end, with the scripted fake detector (no model download)."""

import io
import json
import threading
import zipfile
from pathlib import Path

import av
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from sgblur_video.api.blur_api import create_app
from sgblur_video.config import Settings
from sgblur_video.core.analyze import analyze
from sgblur_video.core.probe import VideoInfo
from sgblur_video.jobs import runner
from sgblur_video.jobs.store import JobStore
from sgblur_video.jobs.worker import Worker
from sgblur_video.privacy.keep import archive_path, decrypt
from tests.privacy.synthetic import FRAMES, FakeDetector, scenario, write_video

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def fake_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Analysis with the scripted detector, in the job process (the real one runs in a child)."""

    def _analysis(job_id: str, info: VideoInfo, settings: Settings, output: Path) -> None:
        progress = runner.Progress(JobStore(settings.data_dir), job_id)
        analyze(info, FakeDetector(scenario()), settings, output, model={"name": "fake"}, progress=progress)

    monkeypatch.setattr(runner, "run_analysis", _analysis)


@pytest.fixture
def settings(tmp_path: Path, repo_root: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        models_file=repo_root / "models" / "registry.yaml",
        tracker_config=repo_root / "configs" / "trackers" / "tracktrack-recall.yaml",
        encoder="libx264",
        keep_secret_key="test-secret",
        callback_allowed_hosts=["callback.test"],
        sync_max_duration_s=10,
    )


@pytest.fixture
def video_bytes(tmp_path: Path) -> bytes:
    path = tmp_path / "synthetic.mp4"
    write_video(path, scenario())
    return path.read_bytes()


def _client(settings: Settings) -> TestClient:
    return TestClient(create_app(settings))


def _submit(client: TestClient, video: bytes, **params: int | str) -> dict:
    response = client.post("/blur/", params=params, files={"video": ("trip.mp4", video, "video/mp4")})
    assert response.status_code == 202, response.text
    assert response.headers["location"] == f"/jobs/{response.json()['job_id']}"
    return response.json()


def _frames(data: bytes) -> int:
    with av.open(io.BytesIO(data)) as container:
        return sum(1 for _ in container.decode(video=0))


def test_full_asynchronous_flow(settings: Settings, video_bytes: bytes) -> None:
    client = _client(settings)
    job = _submit(client, video_bytes, frames=1, keep=1)
    assert job["status"] == "queued"
    assert client.get(f"/jobs/{job['job_id']}/video").json()["code"] == "job_not_ready"
    assert Worker(settings, isolate=False).run_once()

    status = client.get(f"/jobs/{job['job_id']}").json()
    assert status["status"] == "succeeded", status
    assert status["progress"]["percent"] == 100
    assert "debug" not in status["links"]
    assert client.get(f"/jobs/{job['job_id']}/debug").json()["code"] == "debug_not_requested"
    video = client.get(f"/jobs/{job['job_id']}/video")
    assert video.status_code == 200
    assert video.headers["content-type"] == "video/mp4"
    assert _frames(video.content) == FRAMES
    partial = client.get(f"/jobs/{job['job_id']}/video", headers={"Range": "bytes=0-99"})
    assert partial.status_code == 206
    assert len(partial.content) == 100

    metadata = client.get(f"/jobs/{job['job_id']}/metadata").json()
    assert metadata["service_name"] == "SGBlur-Video"
    assert len(metadata["annotations"]) == 2
    frames = client.get(f"/jobs/{job['job_id']}/frames").json()["frames"]
    assert frames
    assert client.get(frames[0]["url"]).headers["content-type"] == "image/jpeg"

    # keep=1: low-confidence regions (the 0.3-score plate) are stored encrypted.
    archive = archive_path(settings.effective_keep_dir, metadata["blurring_id"])
    content = decrypt(archive.read_bytes(), secret="test-secret", blurring_id=metadata["blurring_id"])
    assert json.loads(zipfile.ZipFile(io.BytesIO(content)).read("manifest.json"))["regions"]

    # The original upload is gone as soon as the job ends.
    paths = JobStore(settings.data_dir).paths(job["job_id"])
    assert not any(p.name.startswith("input") for p in paths.root.iterdir())

    assert client.delete(f"/jobs/{job['job_id']}").status_code == 204
    assert not paths.root.exists()
    assert client.get(f"/jobs/{job['job_id']}/video").json()["code"] == "job_expired"


def test_synchronous_multipart_response(settings: Settings, video_bytes: bytes) -> None:
    stop = threading.Event()
    worker = threading.Thread(
        target=Worker(settings, isolate=False, poll_interval=0.1).run_forever, args=(stop,)
    )
    worker.start()
    try:
        client = _client(settings)
        response = client.post(
            "/blur/",
            params={"sync": 1},
            files={"video": ("trip.mp4", video_bytes, "video/mp4")},
            headers={"Accept": "multipart/form-data"},
        )
    finally:
        stop.set()
        worker.join(timeout=30)
    assert response.status_code == 200, response.text
    content_type = response.headers["content-type"]
    assert content_type.startswith("multipart/form-data; boundary=")
    boundary = content_type.split("boundary=")[1].encode()
    parts = response.content.split(b"--" + boundary)
    metadata_part = next(p for p in parts if b'name="metadata"' in p)
    assert b"Content-Type: application/json" in metadata_part
    metadata = json.loads(metadata_part.split(b"\r\n\r\n", 1)[1].rsplit(b"\r\n", 1)[0])
    assert metadata["service_name"] == "SGBlur-Video"
    video_part = next(p for p in parts if b'name="video"' in p)
    assert _frames(video_part.split(b"\r\n\r\n", 1)[1].rsplit(b"\r\n", 1)[0]) == FRAMES
    # Synchronous results are deleted once sent.
    assert not any((Path(settings.data_dir) / "jobs").iterdir())


def test_input_errors(settings: Settings, video_bytes: bytes) -> None:
    client = _client(settings)
    not_video = client.post("/blur/", files={"video": ("notes.mp4", b"hello", "video/mp4")})
    assert (not_video.status_code, not_video.json()["code"]) == (415, "unsupported_media_type")
    raw_360 = client.post("/blur/", files={"video": ("VID.insv", video_bytes, "video/mp4")})
    assert raw_360.status_code == 415
    missing = client.post("/blur/", files={"picture": ("a.mp4", video_bytes, "video/mp4")})
    assert (missing.status_code, missing.json()["code"]) == (422, "invalid_parameter")
    bad_query = client.post("/blur/", params={"keep": "2"}, files={"video": ("a.mp4", video_bytes)})
    assert (bad_query.status_code, bad_query.json()["code"]) == (422, "invalid_parameter")
    assert bad_query.json()["detail"].startswith("keep:")
    callback = client.post(
        "/blur/", params={"callback_url": "http://169.254.169.254/"}, files={"video": ("a.mp4", video_bytes)}
    )
    assert callback.json()["code"] == "callback_not_allowed"
    assert client.get("/jobs/not-a-uuid").status_code == 404
    assert client.get("/jobs/3f0d2c3e-8a51-4c1f-9a4e-0d6f8f4f5b21").json()["code"] == "job_not_found"
    # Nothing is left behind by rejected uploads.
    assert not any(settings.effective_tmp_dir.iterdir())
    assert not any((settings.data_dir / "jobs").iterdir())


def test_limits_and_auth(settings: Settings, video_bytes: bytes) -> None:
    small = settings.model_copy(update={"max_upload_bytes": 1000, "keep_secret_key": None})
    client = _client(small)
    assert client.post("/blur/", files={"video": ("a.mp4", video_bytes)}).json()["code"] == "file_too_large"
    assert client.post("/blur/", params={"keep": 1}, files={"video": ("a.mp4", b"x")}).json()["code"] == (
        "keep_unavailable"
    )
    full = settings.model_copy(update={"queue_max": 1})
    client = _client(full)
    _submit(client, video_bytes)
    busy = client.post("/blur/", files={"video": ("a.mp4", video_bytes)})
    assert (busy.status_code, busy.headers["retry-after"]) == (503, "60")
    secured = _client(settings.model_copy(update={"api_token": SecretStr("t0ken")}))
    assert secured.get("/").status_code == 200  # health stays public
    assert secured.get("/ui").status_code == 200  # the page is static; the user types the token in it
    assert secured.get("/jobs/3f0d2c3e-8a51-4c1f-9a4e-0d6f8f4f5b21").status_code == 401
    authorised = secured.get(
        "/jobs/3f0d2c3e-8a51-4c1f-9a4e-0d6f8f4f5b21", headers={"Authorization": "Bearer t0ken"}
    )
    assert authorised.status_code == 404


def test_health_metrics_and_janitor(settings: Settings, video_bytes: bytes) -> None:
    client = _client(settings)
    health = client.get("/").json()
    assert (health["name"], health["model"]["name"], health["queue"]["queued"]) == (
        "SGBlur-Video",
        "yolo26s",
        0,
    )
    job = _submit(client, video_bytes)
    Worker(settings, isolate=False).run_once()
    metrics = client.get("/metrics").text
    assert 'sgblur_video_jobs{status="succeeded"} 1' in metrics
    assert f"sgblur_video_frames_processed_total {FRAMES}" in metrics
    # Results past their TTL are deleted by the janitor.
    expired = settings.model_copy(update={"result_ttl_minutes": 1})
    store = JobStore(expired.data_dir)
    with store._connect() as connection:
        connection.execute("UPDATE jobs SET expires_at = '2000-01-01T00:00:00+00:00'")
    Worker(expired, isolate=False).janitor()
    assert client.get(f"/jobs/{job['job_id']}").json()["status"] == "expired"
    assert not store.paths(job["job_id"]).root.exists()


@pytest.fixture
def isolated_settings(settings: Settings) -> Settings:
    return settings.model_copy(update={"job_timeout_s": 120})


def test_isolated_job_failure_cleans_up(isolated_settings: Settings, video_bytes: bytes) -> None:
    """A job running in a child process that fails leaves no file behind (spawn path)."""
    client = _client(isolated_settings)
    job = _submit(client, video_bytes)
    store = JobStore(isolated_settings.data_dir)
    for path in store.paths(job["job_id"]).root.iterdir():
        path.write_bytes(b"corrupted after upload")
    assert Worker(isolated_settings, isolate=True).run_once()
    status = client.get(f"/jobs/{job['job_id']}").json()
    assert status["status"] == "failed"
    assert status["error"]["code"] == "unsupported_media_type"
    assert not store.paths(job["job_id"]).root.exists()


def test_web_ui(settings: Settings) -> None:
    page = _client(settings).get("/ui")
    assert page.status_code == 200
    assert page.headers["content-type"].startswith("text/html")
    # Self-contained: no external script, style or font.
    assert "SGBlur-Video" in page.text
    assert "http://" not in page.text
    assert "https://" not in page.text
    assert _client(settings.model_copy(update={"web_ui": False})).get("/ui").status_code == 404


def test_frame_range(settings: Settings, video_bytes: bytes) -> None:
    client = _client(settings)
    job = _submit(client, video_bytes, start_frame=10, end_frame=30)
    assert job["progress"]["frames_total"] == 20
    assert Worker(settings, isolate=False).run_once()
    status = client.get(f"/jobs/{job['job_id']}").json()
    assert status["status"] == "succeeded", status
    assert _frames(client.get(f"/jobs/{job['job_id']}/video").content) == 20
    metadata = client.get(f"/jobs/{job['job_id']}/metadata").json()
    assert metadata["video"]["frame_range"] == {"start": 10, "end": 30}
    paths = JobStore(settings.data_dir).paths(job["job_id"])
    assert not any(p.name.startswith("input") for p in paths.root.iterdir())

    for params, message in (
        ({"start_frame": 5, "end_frame": 5}, "greater than start_frame"),
        ({"start_frame": FRAMES}, "past the end"),
    ):
        response = client.post("/blur/", params=params, files={"video": ("a.mp4", video_bytes)})
        assert (response.status_code, response.json()["code"]) == (422, "invalid_parameter")
        assert message in response.json()["detail"]
    assert not any(settings.effective_tmp_dir.iterdir())


def test_debug_video(settings: Settings, video_bytes: bytes) -> None:
    client = _client(settings)
    job = _submit(client, video_bytes, debug=1, start_frame=10, end_frame=30)
    assert Worker(settings, isolate=False).run_once()
    status = client.get(f"/jobs/{job['job_id']}").json()
    assert status["status"] == "succeeded", status
    assert status["links"]["debug"] == f"/jobs/{job['job_id']}/debug"
    debug = client.get(status["links"]["debug"])
    assert debug.status_code == 200
    assert debug.headers["content-type"] == "video/mp4"
    assert _frames(debug.content) == 20

    disabled = _client(settings.model_copy(update={"debug_videos": False}))
    response = disabled.post("/blur/", params={"debug": 1}, files={"video": ("a.mp4", video_bytes)})
    assert (response.status_code, response.json()["code"]) == (422, "debug_unavailable")
