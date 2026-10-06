"""Tests of the blur plan: linking, selection, gap filling, smoothing, padding, margins, shapes."""

from collections.abc import Mapping

import pytest

from sgblur_video.config import Settings
from sgblur_video.core.detections_io import DetectionRecord, Detections, Footer, FrameRecord, Header
from sgblur_video.core.geometry import Box, iomin
from sgblur_video.core.postprocess import (
    Fragment,
    Observation,
    build_blur_plan,
    link_fragments,
)

Det = tuple[str, float, Box, str | None]


def _detections(frames: Mapping[int, list[Det]], count: int) -> Detections:
    header = Header(
        created_at="2026-10-06T00:00:00Z", video={}, model={}, detection={}, tracking={}, software={}
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


def _settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {"blur_temporal_padding_frames": 0, "blur_box_margin": 0.0}
    return Settings(**(defaults | overrides))  # type: ignore[arg-type]


def _plan(frames: Mapping[int, list[Det]], count: int = 100, **overrides: object):
    return build_blur_plan(
        _detections(frames, count), _settings(**overrides), frame_size=(1000, 1000), fps=30
    )


def _covered(plan, frame: int, box: Box) -> bool:
    return any(iomin(shape.box, box) >= 0.999 for shape in plan.shapes(frame))


def test_tracked_gap_is_interpolated() -> None:
    frames = {0: [("face", 0.9, (0, 0, 10, 10), "face:1")], 10: [("face", 0.9, (100, 0, 110, 10), "face:1")]}
    plan = _plan(frames)
    shape = plan.shapes(5)[0]
    assert shape.source == "interpolated"
    assert _covered(plan, 5, (50, 0, 60, 10))
    assert all(plan.shapes(i) for i in range(11))
    assert not plan.shapes(11)


def test_gap_longer_than_limit_is_not_interpolated() -> None:
    frames = {0: [("plate", 0.9, (0, 0, 10, 10), "plate:1")], 90: [("plate", 0.9, (0, 0, 10, 10), "plate:1")]}
    plan = _plan(frames, max_interpolation_gap_s=1.0)  # 30 frames
    assert not plan.shapes(45)


def test_low_score_detections_of_a_confirmed_track_are_blurred() -> None:
    frames = {
        0: [("face", 0.12, (0, 0, 10, 10), "face:1")],
        1: [("face", 0.9, (0, 0, 10, 10), "face:1")],
        2: [("face", 0.12, (0, 0, 10, 10), "face:1")],
    }
    plan = _plan(frames)
    assert all(plan.shapes(i) for i in range(3))


def test_isolated_low_score_noise_is_not_blurred_but_isolated_hits_are() -> None:
    frames = {5: [("face", 0.11, (0, 0, 10, 10), None)], 50: [("plate", 0.5, (500, 500, 540, 520), None)]}
    plan = _plan(frames, blur_temporal_padding_frames=3)
    assert not plan.shapes(5)
    assert plan.shapes(50)[0].source == "orphan"
    assert [bool(plan.shapes(i)) for i in range(46, 55)] == [
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
    ]
    assert plan.stats["chains_below_threshold"] == 1


def test_flickering_orphans_are_linked_into_one_chain() -> None:
    # A small plate detected every other frame and never tracked.
    frames = {i: [("plate", 0.4, (100 + 2 * i, 100, 130 + 2 * i, 110), None)] for i in range(0, 40, 2)}
    plan = _plan(frames)
    assert plan.stats["chains"] == 1
    assert all(len(plan.shapes(i)) == 1 for i in range(39))
    assert plan.shapes(1)[0].source == "interpolated"


def test_padding_extrapolates_motion_and_grows() -> None:
    frames = {i: [("face", 0.9, (100 + 10 * i, 100, 120 + 10 * i, 120), "face:1")] for i in range(20, 30)}
    plan = _plan(frames, blur_temporal_padding_frames=5, blur_padding_growth=0.1)
    before = plan.shapes(15)[0]
    assert before.source == "padded"
    # 5 frames before the first detection, the face was ~50 px to the left.
    assert _covered(plan, 15, (250, 100, 270, 120))
    assert before.box[2] - before.box[0] > 20  # grown
    assert _covered(plan, 34, (440, 100, 460, 120))
    assert not plan.shapes(14)
    assert not plan.shapes(35)


def test_padding_is_clipped_to_the_video() -> None:
    frames = {0: [("face", 0.9, (0, 0, 10, 10), "face:1")]}
    plan = _plan(frames, count=3, blur_temporal_padding_frames=15)
    assert set(plan.frames) == {0, 1, 2}


def test_smoothing_never_shrinks_boxes() -> None:
    boxes = [(100, 100, 120, 120), (100, 100, 160, 160), (100, 100, 120, 120)]
    frames = {i: [("plate", 0.9, b, "plate:1")] for i, b in enumerate(boxes)}
    plan = _plan(frames)
    for i, box in enumerate(boxes):
        assert _covered(plan, i, box)


def test_margin_and_shapes() -> None:
    frames = {
        0: [("face", 0.9, (100, 100, 200, 200), "face:1"), ("plate", 0.9, (300, 300, 400, 340), "plate:2")]
    }
    plan = _plan(frames, blur_box_margin=0.1)
    shapes = {s.cls: s for s in plan.shapes(0)}
    assert shapes["face"].kind == "ellipse"
    assert shapes["plate"].kind == "rect"
    assert shapes["face"].box == pytest.approx((90, 90, 210, 210))


def test_signs_are_never_in_the_blur_plan() -> None:
    frames = {0: [("sign", 0.99, (0, 0, 50, 50), "signage:1"), ("direction", 0.99, (60, 0, 90, 50), None)]}
    assert not _plan(frames).frames


def test_linking_respects_groups_distance_and_time() -> None:
    def fragment(group: str, frame: int, box: Box) -> Fragment:
        return Fragment(None, group, [Observation(frame, box, 0.5, group)])

    fragments = [
        fragment("face", 0, (0, 0, 10, 10)),
        fragment("plate", 1, (0, 0, 10, 10)),  # other group
        fragment("face", 2, (500, 500, 510, 510)),  # too far
        fragment("face", 3, (2, 0, 12, 10)),  # same face
        fragment("face", 200, (4, 0, 14, 10)),  # too late
    ]
    chains = link_fragments(fragments, max_gap=30, max_distance=1.0)
    sizes = sorted(len(chain.observations) for chain in chains)
    assert sizes == [1, 1, 1, 2]
