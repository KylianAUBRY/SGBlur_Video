from typing import Literal

import pytest

from sgblur_video.bench.dataset import GroundTruth, GtBox, GtTrack
from sgblur_video.bench.metrics import check_gate, coverage, evaluate_clip, size_bucket, summarize
from sgblur_video.core.postprocess import BlurPlan, BlurShape

THRESHOLDS = {"max_leakage_rate_readable": 0.01, "max_consecutive_exposed_frames_readable": 3}


def _shape(
    box: tuple[float, float, float, float], kind: Literal["rect", "ellipse"] = "rect", track: str = "face:1"
) -> BlurShape:
    return BlurShape(kind, box, "face", "detected", track)


def test_coverage_of_rectangles_and_ellipses() -> None:
    truth = (10.0, 10.0, 30.0, 30.0)
    assert coverage(truth, [_shape((0.0, 0.0, 40.0, 40.0))]) == 1.0
    assert coverage(truth, [_shape((0.0, 0.0, 20.0, 40.0))]) == pytest.approx(0.5)
    assert coverage(truth, []) == 0.0
    # The ellipse inscribed in a box leaves its corners out...
    assert coverage(truth, [_shape(truth, "ellipse")]) == pytest.approx(0.785, abs=0.03)
    assert coverage(truth, [_shape((100.0, 100.0, 120.0, 120.0), "ellipse")]) == 0.0
    # ...which a face (an oval) does not need: faces are measured on the oval of their box.
    assert coverage(truth, [_shape(truth, "ellipse")], oval=True) == 1.0


def test_coverage_across_the_360_seam() -> None:
    truth = (0.0, 10.0, 10.0, 20.0)
    shape = _shape((990.0, 5.0, 1015.0, 25.0))  # crosses the seam of a 1000 px wide frame
    assert coverage(truth, [shape]) == 0.0
    assert coverage(truth, [shape], wrap_width=1000) == 1.0


def test_size_buckets() -> None:
    assert size_bucket((0, 0, 10, 8)) == "<16px"
    assert size_bucket((0, 0, 10, 16)) == "16-32px"
    assert size_bucket((0, 0, 10, 50)) == "32-96px"
    assert size_bucket((0, 0, 10, 200)) == ">96px"


def _truth(frames: range, *, readable: bool = True) -> GroundTruth:
    boxes = [GtBox(frame=f, box=(10.0, 10.0, 30.0, 30.0), readable=readable) for f in frames]
    return GroundTruth(
        clip_id="c", frames=20, width=100, height=100, tracks=[GtTrack(id="face:0", cls="face", boxes=boxes)]
    )


def _plan(blurred: set[int], *, chain_of: dict[int, str] | None = None) -> BlurPlan:
    chain_of = chain_of or {}
    plan = BlurPlan(frame_count=20)
    for f in blurred:
        plan.frames[f] = [_shape((5.0, 5.0, 35.0, 35.0), track=chain_of.get(f, "face:1"))]
    return plan


def test_metrics_of_a_clip_with_exposed_frames() -> None:
    truth = _truth(range(10))
    plan = _plan({0, 1, 2, 6, 7, 9}, chain_of={9: "face:2"})  # frames 3-5 and 8 exposed
    evaluation = evaluate_clip(truth, plan, coverage_threshold=0.9, fps=3.0)
    summary = summarize([evaluation])
    assert summary["overall"] == {"object_frames": 10, "unprotected": 4, "leakage_rate": 0.4}
    assert summary["readable"]["unprotected"] == 4
    assert summary["by_class"]["face"]["object_frames"] == 10
    assert summary["by_size"] == {"16-32px": {"object_frames": 10, "unprotected": 4, "leakage_rate": 0.4}}
    assert summary["tracks_ever_leaked"] == 1
    assert summary["longest_exposure_frames"] == 3
    assert summary["longest_exposure_frames_readable"] == 3
    # Window of 1 frame at 3 fps: only frame 8 is protected right before and right after.
    assert summary["transient_exposures"] == 1
    assert summary["mean_chains_per_track"] == 2
    # 30×30 blurred box minus the 20×20 ground truth, on 6 of 20 frames of 100×100 pixels.
    assert summary["over_blur_ratio"] == pytest.approx(0.05 * 6 / 20, abs=0.003)
    assert check_gate(summary, THRESHOLDS) == ["leakage rate of readable objects 40.00% > 1.00%"]


def test_gate_on_consecutive_exposure_and_unreadable_objects() -> None:
    long_exposure = summarize(
        [evaluate_clip(_truth(range(10)), _plan({0, 9}), coverage_threshold=0.9, fps=30)]
    )
    failures = check_gate(long_exposure, {**THRESHOLDS, "max_leakage_rate_readable": 1.0})
    assert failures == ["a readable object stayed exposed 8 consecutive frames > 3"]
    unreadable = summarize(
        [evaluate_clip(_truth(range(10), readable=False), _plan(set()), coverage_threshold=0.9, fps=30)]
    )
    assert unreadable["overall"]["leakage_rate"] == 1.0
    assert unreadable["readable"]["object_frames"] == 0
    assert check_gate(unreadable, THRESHOLDS) == []


def test_summary_without_ground_truth_objects() -> None:
    empty = GroundTruth(clip_id="c", frames=5, width=10, height=10, tracks=[])
    summary = summarize([evaluate_clip(empty, BlurPlan(frame_count=5), coverage_threshold=0.9, fps=30)])
    assert summary["overall"]["leakage_rate"] == 0.0
    assert summary["mean_chains_per_track"] == 0.0
