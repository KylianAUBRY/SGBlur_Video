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


def test_short_tracked_gap_is_interpolated() -> None:
    frames = {0: [("face", 0.9, (0, 0, 10, 10), "face:1")], 6: [("face", 0.9, (30, 0, 40, 10), "face:1")]}
    plan = _plan(frames)
    shape = plan.shapes(3)[0]
    assert shape.source == "interpolated"
    assert shape.box == pytest.approx((15, 0, 25, 10))
    assert all(plan.shapes(i) for i in range(7))
    assert not plan.shapes(7)


def test_long_gaps_are_not_interpolated_by_default() -> None:
    # 10 missing frames (0.33 s at 30 fps) is longer than MAX_INTERPOLATION_GAP_S (0.3 s, 9 frames).
    frames = {0: [("face", 0.9, (0, 0, 10, 10), "face:1")], 11: [("face", 0.9, (0, 0, 10, 10), "face:1")]}
    assert not _plan(frames).shapes(5)


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


def test_padding_ignores_unreliable_motion() -> None:
    # Two detections 30 px apart (an intermittent plate, or two plates linked together): padding
    # used to follow that "motion" 300 px away over 10 frames; it stays on the boxes.
    frames = {
        20: [("plate", 0.9, (400, 500, 440, 520), None)],
        21: [("plate", 0.9, (430, 500, 470, 520), None)],
    }
    plan = _plan(frames, blur_temporal_padding_frames=10)
    assert _covered(plan, 31, (430, 500, 470, 520))
    assert all(350 < shape.box[0] < 450 for i in (10, 31) for shape in plan.shapes(i))


def test_padding_speed_is_capped() -> None:
    # Boxes jumping 3 box sizes per frame are followed at most 0.5 box size (10 px) per frame.
    frames = {20 + i: [("plate", 0.9, (100 + 60 * i, 100, 120 + 60 * i, 120), "plate:1")] for i in range(5)}
    plan = _plan(frames, blur_temporal_padding_frames=10)
    last = (340, 100, 360, 120)
    shape = plan.shapes(34)[0]
    assert shape.source == "padded"
    assert shape.box[0] <= last[0] + 10 * 10
    assert _covered(plan, 25, (last[0] + 10, 100, last[2] + 10, 120))


def test_padding_does_not_follow_the_growth_of_the_box() -> None:
    # An approaching plate whose box grows by 10 px per frame: following that growth made padded
    # boxes several times larger than the plate (190 px plate, 830 px padded box on 8K video).
    frames = {i: [("plate", 0.9, (500, 500, 540 + 10 * (i - 20), 520), "plate:1")] for i in range(20, 25)}
    plan = _plan(frames, blur_temporal_padding_frames=10, blur_padding_growth=0.0)
    shape = plan.shapes(34)[0]
    assert shape.source == "padded"
    assert shape.box[2] - shape.box[0] == pytest.approx(80)


def test_jump_between_detections_is_not_interpolated() -> None:
    # 50 box sizes apart: two faces tracked as one, not one face crossing the frame.
    frames = {0: [("face", 0.9, (0, 0, 10, 10), "face:1")], 10: [("face", 0.9, (500, 0, 510, 10), "face:1")]}
    assert not _plan(frames).shapes(5)
    assert _plan(frames, max_interpolation_jump=100).shapes(5)[0].source == "interpolated"


def test_each_part_of_a_jumping_chain_must_reach_the_blur_threshold() -> None:
    # A tracker linked a real face (0.6) and a low-score false positive far away (0.11): only the
    # face is blurred.
    frames = {
        0: [("face", 0.6, (0, 0, 10, 10), "face:1")],
        5: [("face", 0.11, (500, 500, 510, 510), "face:1")],
    }
    plan = _plan(frames, blur_temporal_padding_frames=2)
    assert plan.shapes(0)
    assert not plan.shapes(5)
    assert all(shape.box[0] < 100 for i in range(10) for shape in plan.shapes(i))
    assert plan.stats["parts_below_threshold"] == 1


def test_padding_is_not_merged_across_a_jump() -> None:
    # The padding after the first box must not be merged with the second box into one large box.
    frames = {
        0: [("plate", 0.9, (0, 0, 10, 10), "plate:1")],
        3: [("plate", 0.9, (500, 0, 510, 10), "plate:1")],
    }
    plan = _plan(frames, blur_temporal_padding_frames=5)
    shapes = plan.shapes(3)
    assert len(shapes) == 2
    assert all(shape.box[2] - shape.box[0] < 20 for shape in shapes)


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
