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


def test_planned_commands_exit_with_code_2(tmp_path: Path) -> None:
    video = tmp_path / "in.mp4"
    video.write_bytes(b"")
    result = runner.invoke(app, ["blur", str(video), str(tmp_path / "out.mp4")])
    assert result.exit_code == 2
    assert "step 4" in result.output
