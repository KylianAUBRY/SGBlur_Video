"""End-to-end CLI tests on a synthetic video with a fake detector (no model download)."""

import json
from pathlib import Path

import av
import pytest
from typer.testing import CliRunner

from sgblur_video.cli import app
from sgblur_video.config import Settings
from sgblur_video.core import pipeline
from sgblur_video.core.detections_io import DetectionsFormatError
from sgblur_video.core.pipeline import run_render
from tests.privacy.synthetic import FRAMES, FakeDetector, scenario, write_video

pytestmark = pytest.mark.integration

runner = CliRunner()


@pytest.fixture
def fake_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the YOLO model by the scripted fake detector."""

    class _Entry:
        name, version, sha256, tag = "fake", "0", "0" * 64, "fake/0"

    def _load(_settings: Settings, _name: str | None = None) -> pipeline.LoadedModel:
        return pipeline.LoadedModel(detector=FakeDetector(scenario()), entry=_Entry(), device="cpu")  # type: ignore[arg-type]

    monkeypatch.setattr(pipeline, "load_model", _load)


@pytest.fixture
def video(tmp_path: Path) -> Path:
    path = tmp_path / "synthetic.mp4"
    write_video(path, scenario())
    return path


def _frames(path: Path) -> int:
    with av.open(str(path)) as container:
        return sum(1 for _ in container.decode(video=0))


@pytest.mark.usefixtures("fake_model")
def test_detect_then_render_with_debug(
    tmp_path: Path, video: Path, repo_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRACKER_CONFIG", str(repo_root / "configs" / "trackers" / "bytetrack-recall.yaml"))
    monkeypatch.setenv("ENCODER", "libx264")
    detections = tmp_path / "detections.jsonl"
    result = runner.invoke(app, ["detect", str(video), "--out", str(detections)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output[result.output.index("{") :])["complete"] is True
    output = tmp_path / "out.mp4"
    result = runner.invoke(app, ["render", str(video), str(detections), str(output), "--debug"])
    assert result.exit_code == 0, result.output
    assert _frames(output) == FRAMES
    assert _frames(tmp_path / "out.debug.mp4") == FRAMES


@pytest.mark.usefixtures("fake_model")
def test_blur_command_with_max_frames(tmp_path: Path, video: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENCODER", "libx264")
    output = tmp_path / "out.mp4"
    result = runner.invoke(
        app,
        [
            "blur",
            str(video),
            str(output),
            "--max-frames",
            "20",
            "--keep-detections",
            str(tmp_path / "d.jsonl"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert _frames(output) == 20
    assert (tmp_path / "d.jsonl").exists()


@pytest.mark.usefixtures("fake_model")
def test_incomplete_or_foreign_detections_are_refused(tmp_path: Path, video: Path) -> None:
    settings = Settings(encoder="libx264")
    detections = tmp_path / "d.jsonl"
    pipeline.run_detect(video, detections, settings, max_frames=10)
    with pytest.raises(DetectionsFormatError, match="incomplete"):
        run_render(video, detections, tmp_path / "out.mp4", settings)
    result = run_render(video, detections, tmp_path / "out.mp4", settings, allow_partial=True)
    assert result.render.frames == 10
    other = tmp_path / "other.mp4"
    with av.open(str(other), "w") as container:
        stream = container.add_stream("libx264", rate=30)
        stream.width, stream.height, stream.pix_fmt = 320, 240, "yuv420p"
        for i in range(3):
            frame = av.VideoFrame(320, 240, "yuv420p")
            frame.pts = i
            container.mux(stream.encode(frame))
        container.mux(stream.encode(None))
    with pytest.raises(DetectionsFormatError, match="another size"):
        run_render(other, detections, tmp_path / "out2.mp4", settings, allow_partial=True)
