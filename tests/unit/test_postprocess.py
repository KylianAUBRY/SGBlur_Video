"""Tests of the blur plan: every frame is blurred on its own detections, like a picture."""

from collections.abc import Mapping

import pytest

from sgblur_video.config import Settings
from sgblur_video.core.detections_io import DetectionRecord, Detections, Footer, FrameRecord, Header
from sgblur_video.core.geometry import Box
from sgblur_video.core.postprocess import MIN_BLUR_SIZE, build_blur_plan

Det = tuple[str, float, Box, str | None]


def _detections(frames: Mapping[int, list[Det]], count: int) -> Detections:
    header = Header(
        created_at="2026-10-08T00:00:00Z", video={}, model={}, detection={}, tracking={}, software={}
    )
    records = [
        FrameRecord(
            index=i,
            pts=i,
            time=i / 30,
            detections=[
                DetectionRecord.model_validate({"class": c, "score": s, "box": b, "track_id": t})
                for c, s, b, t in frames.get(i, [])
            ],
        )
        for i in range(count)
    ]
    return Detections(header, records, Footer(frames=count, complete=True, elapsed_s=0))


def _plan(frames: Mapping[int, list[Det]], count: int = 20, *, wrap_width: int | None = None):
    return build_blur_plan(
        _detections(frames, count), Settings(), frame_size=(1000, 500), wrap_width=wrap_width
    )


def test_each_detection_is_blurred_on_its_frame_only() -> None:
    frames = {
        3: [("face", 0.4, (100, 100, 140, 150), None), ("plate", 0.9, (300, 300, 380, 320), None)],
        7: [("face", 0.6, (110, 100, 150, 150), None)],
    }
    plan = _plan(frames)
    assert set(plan.frames) == {3, 7}  # nothing carried to the frames around
    shapes = {s.cls: s for s in plan.shapes(3)}
    assert shapes["face"].box == pytest.approx((100, 100, 140, 150))  # exactly the detected box
    assert shapes["face"].score == pytest.approx(0.4)
    assert shapes["plate"].box == pytest.approx((300, 300, 380, 320))
    assert plan.stats == {"boxes_face": 2, "boxes_plate": 1, "frames_with_blur": 2}


def test_detections_below_conf_detect_are_not_blurred() -> None:
    frames = {0: [("face", 0.29, (0, 0, 40, 40), None), ("plate", 0.30, (100, 100, 160, 120), None)]}
    plan = _plan(frames)
    assert [s.cls for s in plan.shapes(0)] == ["plate"]
    assert plan.stats["below_conf"] == 1


def test_signs_are_never_in_the_blur_plan() -> None:
    frames = {0: [("sign", 0.99, (0, 0, 50, 50), "signage:1"), ("direction", 0.99, (60, 0, 90, 50), None)]}
    assert not _plan(frames).frames


def test_boxes_smaller_than_sgblur_minimum_are_not_blurred() -> None:
    small = MIN_BLUR_SIZE - 1
    frames = {
        0: [
            ("face", 0.9, (0, 0, small, 40), None),
            ("plate", 0.9, (100, 100, 140, 100 + small), None),
            ("face", 0.9, (200, 200, 200 + MIN_BLUR_SIZE, 200 + MIN_BLUR_SIZE), None),
        ]
    }
    plan = _plan(frames)
    assert [s.box for s in plan.shapes(0)] == [(200, 200, 200 + MIN_BLUR_SIZE, 200 + MIN_BLUR_SIZE)]
    assert plan.stats["too_small"] == 2


def test_boxes_outside_the_frame_are_dropped() -> None:
    frames = {0: [("face", 0.9, (1100, 100, 1150, 150), None), ("plate", 0.9, (100, 600, 150, 650), None)]}
    assert not _plan(frames).frames


def test_360_boxes_crossing_the_seam_are_kept_whole() -> None:
    frames = {
        0: [("plate", 0.9, (980, 100, 1030, 120), None)],
        1: [("plate", 0.9, (-20, 100, 30, 120), None)],
    }
    plan = _plan(frames, wrap_width=1000)
    assert plan.shapes(0)[0].box == pytest.approx((980, 100, 1030, 120))
    assert plan.shapes(1)[0].box == pytest.approx((980, 100, 1030, 120))  # normalised: 0 <= x1 < width
    assert plan.wrap_width == 1000
