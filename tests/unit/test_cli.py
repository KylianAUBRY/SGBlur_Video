"""Tests of the command-line interface skeleton."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from sgblur_video import __version__
from sgblur_video.cli import app

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("blur", "detect", "render", "signs", "benchmark", "worker", "serve", "models"):
        assert command in result.output


def test_models_list(repo_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODELS_FILE", str(repo_root / "models" / "registry.yaml"))
    result = runner.invoke(app, ["models", "list"])
    assert result.exit_code == 0
    assert "yolo26s" in result.output
    assert "direction,sign,plate,face" in result.output


def test_models_list_with_broken_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODELS_FILE", str(tmp_path / "missing.yaml"))
    result = runner.invoke(app, ["models", "list"])
    assert result.exit_code == 1


def test_config_masks_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KEEP_SECRET_KEY", "do-not-print")
    result = runner.invoke(app, ["config"])
    assert result.exit_code == 0
    assert "do-not-print" not in result.output
    assert json.loads(result.output)["api_name"] == "SGBlur-Video"


def test_config_hides_the_home_folder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODELS_DIR", str(Path.home() / "models-here"))
    result = runner.invoke(app, ["config"])
    assert result.exit_code == 0
    assert str(Path.home()) not in result.output
    assert json.loads(result.output)["models_dir"] == "~/models-here"


def test_benchmark_commands(tmp_path: Path, repo_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result = runner.invoke(app, ["benchmark", "--help"])
    assert result.exit_code == 0
    for command in ("privacy", "trackers", "speed"):
        assert command in result.output
    result = runner.invoke(app, ["benchmark", "trackers"])
    assert result.exit_code == 1
    assert "give --video and/or --dataset" in result.output
    monkeypatch.setenv("MODELS_FILE", str(repo_root / "models" / "registry.yaml"))
    thresholds = str(repo_root / "benchmarks" / "privacy-thresholds.yaml")
    result = runner.invoke(
        app, ["benchmark", "privacy", "--dataset", str(tmp_path), "--thresholds", thresholds]
    )
    assert result.exit_code == 1
    assert "no annotated clip" in result.output
    result = runner.invoke(
        app,
        ["benchmark", "privacy", "--dataset", str(tmp_path), "--thresholds", thresholds, "--sweep", "X"],
    )
    assert result.exit_code == 1


def test_annotate_errors_are_reported(tmp_path: Path) -> None:
    export = tmp_path / "export.xml"
    export.write_text("<annotations/>", encoding="utf-8")
    result = runner.invoke(app, ["annotate", "import", str(export), "--dataset", str(tmp_path), "--id", "a"])
    assert result.exit_code == 1
    assert "no clip 'a'" in result.output
    video = tmp_path / "in.mp4"
    video.write_bytes(b"not a video")
    result = runner.invoke(
        app, ["annotate", "export", str(video), "--dataset", str(tmp_path), "--id", "Bad Id"]
    )
    assert result.exit_code == 1
    assert "invalid clip id" in result.output


def test_invalid_video_is_reported_without_traceback(tmp_path: Path) -> None:
    video = tmp_path / "in.mp4"
    video.write_bytes(b"not a video")
    result = runner.invoke(app, ["blur", str(video), str(tmp_path / "out.mp4")])
    assert result.exit_code == 1
    assert "not a readable video" in result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)
