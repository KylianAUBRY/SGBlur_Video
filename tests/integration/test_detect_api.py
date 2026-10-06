"""Detect API with the real model on CPU (downloads the SGBlur weights once)."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from sgblur_video.api.detect_api import create_app
from sgblur_video.config import Settings
from tests.privacy.synthetic import FRAMES, scenario, write_video

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def test_detect_streams_a_complete_detections_file(tmp_path: Path, repo_root: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "data",
        models_file=repo_root / "models" / "registry.yaml",
        tracker_config=repo_root / "configs" / "trackers" / "tracktrack-recall.yaml",
        device="cpu",
    )
    video = tmp_path / "synthetic.mp4"
    write_video(video, scenario())
    client = TestClient(create_app(settings))
    assert client.get("/").json()["busy"] is False
    rejected = client.post("/detect/", files={"video": ("notes.mp4", b"hello", "video/mp4")})
    assert rejected.status_code == 415
    response = client.post("/detect/", files={"video": ("synthetic.mp4", video.read_bytes(), "video/mp4")})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    lines = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    if not lines:
        pytest.skip("model download unavailable")
    assert lines[0]["type"] == "header"
    assert lines[-1] == lines[-1] | {"type": "footer", "complete": True, "frames": FRAMES}
    assert sum(line["type"] == "frame" for line in lines) == FRAMES
    assert not any((settings.effective_tmp_dir).iterdir())  # received video deleted
    assert client.get("/").json()["busy"] is False
