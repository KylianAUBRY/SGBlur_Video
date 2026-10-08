"""The overlapped analysis stages: same output as the scripted detector run directly, errors surface."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from sgblur_video.config import Settings
from sgblur_video.core import track
from sgblur_video.core.analyze import analyze
from sgblur_video.core.detect import Detection, DetectionPass, PreparedFrame
from sgblur_video.core.probe import VideoInfo, probe
from tests.privacy.synthetic import FRAMES, FakeDetector, scenario, write_video


class StagedFake(FakeDetector):
    """The scripted detector, split into ``prepare`` (run in the preparation thread) and ``infer``."""

    def __init__(self, fail_prepare_at: int | None = None) -> None:
        super().__init__(scenario())
        self.fail_prepare_at = fail_prepare_at
        self.prepared = 0
        self._plan: Sequence[DetectionPass] = ()
        self._index = 0

    def prepare(self, image: npt.NDArray[np.uint8], plan: Sequence[DetectionPass]) -> PreparedFrame:
        """Keep the frame as the input of every pass (fails at frame ``fail_prepare_at``)."""
        if self.prepared == self.fail_prepare_at:
            msg = "prepare failed"
            raise ValueError(msg)
        self.prepared += 1
        self._plan = plan
        return PreparedFrame(inputs=tuple((p, image) for p in plan))

    def infer(self, prepared: PreparedFrame) -> list[Detection]:
        """Scripted detections of the next frame (inference sees frames in order)."""
        detections = super().detect(prepared.inputs[0][1], self._plan, self._index)
        self._index += 1
        return detections


class FailingFake(FakeDetector):
    """The scripted detector, failing on frame 3."""

    def detect(
        self, image: npt.NDArray[np.uint8], plan: Sequence[DetectionPass], frame_index: int
    ) -> list[Detection]:
        """Scripted detections, or an error on frame 3."""
        if frame_index == 3:
            msg = "inference failed"
            raise RuntimeError(msg)
        return super().detect(image, plan, frame_index)


@pytest.fixture
def video(tmp_path: Path, repo_root: Path) -> tuple[VideoInfo, Settings]:
    settings = Settings(tracker_config=repo_root / "configs" / "trackers" / "tracktrack-recall.yaml")
    path = tmp_path / "synthetic.mp4"
    write_video(path, scenario())
    return probe(path, settings), settings


def test_staged_detector_gives_the_same_detections(video: tuple[VideoInfo, Settings], tmp_path: Path) -> None:
    info, settings = video
    plain, staged = tmp_path / "plain.jsonl", tmp_path / "staged.jsonl"
    analyze(info, FakeDetector(scenario()), settings, plain, model={"name": "fake"})
    detector = StagedFake()
    footer = analyze(info, detector, settings, staged, model={"name": "fake"})
    assert footer.frames == FRAMES
    assert detector.prepared == FRAMES

    def frames(path: Path) -> list[str]:
        return [line for line in path.read_text(encoding="utf-8").splitlines() if '"type": "frame"' in line]

    # Track ids are numbered per process, so compare boxes, scores and classes in order.
    def strip(lines: list[str]) -> list[str]:
        import json

        return [
            json.dumps([(d["class"], d["score"], d["box"]) for d in json.loads(line)["detections"]])
            for line in lines
        ]

    assert strip(frames(staged)) == strip(frames(plain))


def test_progress_is_reported_for_every_frame(video: tuple[VideoInfo, Settings], tmp_path: Path) -> None:
    info, settings = video
    calls: list[int] = []
    analyze(
        info,
        StagedFake(),
        settings,
        tmp_path / "d.jsonl",
        model={"name": "fake"},
        progress=lambda done, _total: calls.append(done),
    )
    assert calls == list(range(1, FRAMES + 1))


@pytest.mark.parametrize(
    ("detector", "error"),
    [
        (lambda: FailingFake(scenario()), "inference failed"),
        (lambda: StagedFake(fail_prepare_at=5), "prepare failed"),
    ],
)
def test_stage_errors_are_raised(
    video: tuple[VideoInfo, Settings], tmp_path: Path, detector: object, error: str
) -> None:
    info, settings = video
    with pytest.raises((RuntimeError, ValueError), match=error):
        analyze(info, detector(), settings, tmp_path / "d.jsonl", model={"name": "fake"})  # type: ignore[operator]


def test_tracking_errors_are_raised(
    video: tuple[VideoInfo, Settings], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    info, settings = video
    original = track.GroupTrackers.update
    calls = []

    def flaky(self: track.GroupTrackers, *args: object, **kwargs: object) -> None:
        calls.append(1)
        if len(calls) == 4:
            msg = "tracking failed"
            raise RuntimeError(msg)
        original(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(track.GroupTrackers, "update", flaky)
    with pytest.raises(RuntimeError, match="tracking failed"):
        analyze(info, StagedFake(), settings, tmp_path / "d.jsonl", model={"name": "fake"})
