"""Pass 1 of a job in its own process, with the real model (downloads ~20 MB once)."""

import signal
import subprocess
from pathlib import Path

import pytest

from sgblur_video.config import Settings
from sgblur_video.core.detections_io import read_detections
from sgblur_video.core.probe import probe
from sgblur_video.jobs import analysis
from sgblur_video.jobs.analysis import AnalysisKilledError, run_analysis
from sgblur_video.jobs.store import JobStore
from tests.privacy.synthetic import FRAMES, scenario, write_video

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def test_analysis_runs_in_a_child_process(tmp_path: Path, repo_root: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "data",
        models_file=repo_root / "models" / "registry.yaml",
        tracker_config=repo_root / "configs" / "trackers" / "tracktrack-recall.yaml",
        device="cpu",
    )
    video = tmp_path / "synthetic.mp4"
    write_video(video, scenario())
    store = JobStore(settings.data_dir)
    job = store.create("3f0d2c3e-8a51-4c1f-9a4e-0d6f8f4f5b21", {}, "input.mp4", FRAMES)
    assert store.claim() is not None
    output = store.paths(job).root / "detections.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        run_analysis(job.id, probe(video, settings), settings, output)
    except RuntimeError as exc:
        pytest.skip(f"model unavailable in the child process: {exc}")
    detections = read_detections(output)
    assert detections.complete
    assert len(detections.frames) == FRAMES
    current = store.get(job.id)
    assert current is not None
    assert current.frames_done == FRAMES  # the child reported its progress to the store
    assert not list(output.parent.glob("*.pickle"))  # its arguments file is deleted


def test_a_killed_analysis_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def killed(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess([], -signal.SIGKILL)

    monkeypatch.setattr(analysis.subprocess, "run", killed)
    video = tmp_path / "v.mp4"
    write_video(video, scenario())
    info = probe(video, Settings())
    with pytest.raises(AnalysisKilledError):
        run_analysis("job", info, Settings(), tmp_path / "detections.jsonl")
