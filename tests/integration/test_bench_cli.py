"""The benchmark and dataset commands with the real model, on a short synthetic clip (CPU)."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from sgblur_video.bench.dataset import Dataset
from sgblur_video.cli import app
from tests.privacy.synthetic import FPS, HEIGHT, WIDTH, scenario, write_video

pytestmark = [pytest.mark.integration, pytest.mark.slow]

runner = CliRunner()


@pytest.fixture
def env(tmp_path: Path, repo_root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("MODELS_FILE", str(repo_root / "models" / "registry.yaml"))
    monkeypatch.setenv("TRACKER_CONFIG", str(repo_root / "configs" / "trackers" / "tracktrack-recall.yaml"))
    monkeypatch.setenv("DEVICE", "cpu")
    monkeypatch.setenv("ENCODER", "libx264")
    video = tmp_path / "synthetic.mp4"
    write_video(video, scenario())
    return video


def _invoke(args: list[str]) -> str:
    result = runner.invoke(app, args)
    if result.exit_code != 0 and "cannot download" in result.output:
        pytest.skip(f"model download unavailable: {result.output}")
    assert result.exit_code == 0, result.output
    return result.output


def test_annotate_export(env: Path, tmp_path: Path) -> None:
    dataset = Dataset(tmp_path / "dataset")
    export = ["annotate", "export", str(env), "--dataset", str(dataset.root), "--id", "syn"]
    output = _invoke([*export, "--start", "0.5", "--duration", "0.5", "--proxy-width", str(WIDTH // 2)])
    assert "syn: 15 frames" in output
    entry = dataset.load_manifest().get("syn")
    assert (entry.frames, entry.width, entry.proxy_width, entry.proxy_height) == (
        15,
        WIDTH,
        WIDTH // 2,
        HEIGHT // 2,
    )
    assert entry.start_s == 0.5
    assert entry.fps == FPS
    assert dataset.preannotation("syn").read_text(encoding="utf-8").startswith("<?xml")
    labels = json.loads((dataset.root / "cvat-labels.json").read_text(encoding="utf-8"))
    assert [label["name"] for label in labels] == ["face", "plate"]
    assert list(dataset.cache_dir("syn").glob("*.jsonl"))
    output = _invoke(
        ["annotate", "preannotate", "--dataset", str(dataset.root), "--id", "syn", "--min-frames", "1"]
    )
    assert "syn: " in output
    assert "pre-annotated tracks" in output
    # An annotated clip is never overwritten.
    dataset.ground_truth_path("syn").write_text("{}", encoding="utf-8")
    result = runner.invoke(app, export)
    assert result.exit_code == 1
    assert "already annotated" in result.output


def test_benchmark_speed(env: Path) -> None:
    output = _invoke(["benchmark", "speed", str(env), "--frames", "2", "--profile", "fast"])
    assert "| yolo26s/0.1.0 | cpu | fast | 1 |" in output
    assert f"{WIDTH}x{HEIGHT} flat h264" in output
