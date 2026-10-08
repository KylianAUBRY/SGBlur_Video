"""Privacy metrics of a blur plan against human ground truth (``docs/design/testing-strategy.md``).

For each ground-truth object-frame (box ``G``) and the rectangles ``M`` the
renderer blurs on that frame (with their copies across the 360° seam):

* **coverage** = |G ∩ M| / |G|, estimated on a grid of up to 16 × 16 points of
  ``G`` (blurring more than needed is harmless, so this is not an IoU);
* the object-frame is **protected** when coverage ≥ ``coverage_threshold``.

The metrics measure detection and the blur plan. That the renderer
really destroys the pixels of every planned shape is checked separately by the
synthetic oracle (``tests/privacy``).
"""

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from sgblur_video.bench.dataset import GroundTruth
from sgblur_video.core.geometry import Box, height, width
from sgblur_video.core.postprocess import BlurPlan, BlurShape
from sgblur_video.video360.wrap import copies

#: Size buckets on the ground-truth box height, in pixels.
SIZE_BUCKETS: tuple[tuple[float, float, str], ...] = (
    (0, 16, "<16px"),
    (16, 32, "16-32px"),
    (32, 96, "32-96px"),
    (96, math.inf, ">96px"),
)
_GRID = 16
_RASTER_WIDTH = 480


def size_bucket(box: Box) -> str:
    """Size bucket of a box (by height)."""
    h = height(box)
    return next(label for low, high, label in SIZE_BUCKETS if low <= h < high)


def _boxes(shape: BlurShape, wrap_width: int | None) -> list[Box]:
    return copies(shape.box, wrap_width) if wrap_width else [shape.box]


def _inside(box: Box, xs: npt.NDArray[np.float64], ys: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
    return np.asarray((xs >= box[0]) & (xs <= box[2]) & (ys >= box[1]) & (ys <= box[3]), dtype=np.bool_)


def coverage(truth: Box, shapes: Sequence[BlurShape], wrap_width: int | None = None) -> float:
    """Share of a ground-truth box covered by the blurred shapes.

    Args:
        truth: Ground-truth box.
        shapes: Shapes blurred on the frame.
        wrap_width: Frame width of a 360° video (shapes are also blurred one turn left and right).

    Returns:
        Coverage between 0 and 1.
    """
    nx = max(1, min(_GRID, math.ceil(width(truth))))
    ny = max(1, min(_GRID, math.ceil(height(truth))))
    xs, ys = np.meshgrid(
        truth[0] + (np.arange(nx) + 0.5) * width(truth) / nx,
        truth[1] + (np.arange(ny) + 0.5) * height(truth) / ny,
    )
    covered = np.zeros_like(xs, dtype=bool)
    for shape in shapes:
        for box in _boxes(shape, wrap_width):
            if box[2] < truth[0] or box[0] > truth[2] or box[3] < truth[1] or box[1] > truth[3]:
                continue
            covered |= _inside(box, xs, ys)
    return float(covered.mean())


@dataclass(frozen=True)
class ObjectFrame:
    """One ground-truth object on one frame, and whether it was protected."""

    track: str
    cls: str
    frame: int
    size: str
    readable: bool
    coverage: float
    protected: bool


@dataclass
class ClipEvaluation:
    """Metrics inputs of one clip.

    Attributes:
        clip_id: Clip identifier.
        object_frames: Every ground-truth object-frame with its coverage.
        over_blur_ratio: Mean share of frame pixels blurred outside every ground-truth box.
        fps: Frame rate (for transient exposures).
    """

    clip_id: str
    object_frames: list[ObjectFrame] = field(default_factory=list)
    over_blur_ratio: float = 0.0
    fps: float = 30.0


def _over_blur(truth: GroundTruth, plan: BlurPlan) -> float:
    """Mean share of each frame blurred outside the ground-truth boxes (rasterised at 480 px wide)."""
    factor = min(1.0, _RASTER_WIDTH / truth.width)
    size = (max(1, round(truth.height * factor)), max(1, round(truth.width * factor)))
    gt_by_frame: dict[int, list[Box]] = defaultdict(list)
    for track in truth.tracks:
        for gt in track.boxes:
            gt_by_frame[gt.frame].append(gt.box)
    total = 0.0
    for frame in range(truth.frames):
        shapes = plan.shapes(frame)
        if not shapes:
            continue
        mask = np.zeros(size, dtype=np.uint8)
        for shape in shapes:
            for box in _boxes(shape, plan.wrap_width):
                cv2.rectangle(
                    mask,
                    (round(box[0] * factor), round(box[1] * factor)),
                    (round(box[2] * factor), round(box[3] * factor)),
                    1,
                    -1,
                )
        for known in gt_by_frame.get(frame, []):
            cv2.rectangle(
                mask,
                (math.floor(known[0] * factor), math.floor(known[1] * factor)),
                (math.ceil(known[2] * factor), math.ceil(known[3] * factor)),
                0,
                -1,
            )
        total += float(mask.mean())
    return total / max(1, truth.frames)


def evaluate_clip(
    truth: GroundTruth, plan: BlurPlan, *, coverage_threshold: float, fps: float
) -> ClipEvaluation:
    """Compare a blur plan with the ground truth of one clip.

    Args:
        truth: Ground truth (full-resolution pixels).
        plan: Blur plan computed on the same clip.
        coverage_threshold: Coverage above which an object-frame counts as protected.
        fps: Clip frame rate.

    Returns:
        Per object-frame results and per-clip measures.
    """
    evaluation = ClipEvaluation(clip_id=truth.clip_id, fps=fps)
    for track in truth.tracks:
        for gt in track.boxes:
            value = coverage(gt.box, plan.shapes(gt.frame), plan.wrap_width)
            evaluation.object_frames.append(
                ObjectFrame(
                    track=track.id,
                    cls=track.cls,
                    frame=gt.frame,
                    size=size_bucket(gt.box),
                    readable=gt.readable,
                    coverage=round(value, 4),
                    protected=value >= coverage_threshold,
                )
            )
    evaluation.over_blur_ratio = _over_blur(truth, plan)
    return evaluation


def _rate(items: Sequence[ObjectFrame]) -> dict[str, float | int]:
    unprotected = sum(1 for o in items if not o.protected)
    return {
        "object_frames": len(items),
        "unprotected": unprotected,
        "leakage_rate": round(unprotected / len(items), 5) if items else 0.0,
    }


def _longest_run(frames: Iterable[int]) -> int:
    """Longest run of consecutive frame indices."""
    longest = run = 0
    previous: int | None = None
    for frame in sorted(frames):
        run = run + 1 if previous is not None and frame == previous + 1 else 1
        longest = max(longest, run)
        previous = frame
    return longest


def summarize(evaluations: Sequence[ClipEvaluation]) -> dict[str, Any]:
    """Aggregate the metrics of several clips.

    Returns:
        A JSON-serialisable summary: leakage overall, per class, per size bucket and for
        ``readable`` objects; tracks ever leaked; longest exposures; transient exposures;
        over-blur ratio.
    """
    frames = [o for e in evaluations for o in e.object_frames]
    tracks: dict[tuple[str, str], list[ObjectFrame]] = defaultdict(list)
    for evaluation in evaluations:
        for o in evaluation.object_frames:
            tracks[(evaluation.clip_id, o.track)].append(o)
    fps_of = {e.clip_id: e.fps for e in evaluations}

    transient = 0
    longest = longest_readable = 0
    ever_leaked = readable_ever_leaked = 0
    readable_tracks = 0
    for (clip_id, _track), items in tracks.items():
        window = max(1, round(fps_of[clip_id] / 3))
        protected = {o.frame for o in items if o.protected}
        exposed = [o.frame for o in items if not o.protected]
        exposed_readable = [o.frame for o in items if not o.protected and o.readable]
        if any(o.readable for o in items):
            readable_tracks += 1
        if exposed:
            ever_leaked += 1
        if exposed_readable:
            readable_ever_leaked += 1
        longest = max(longest, _longest_run(exposed))
        longest_readable = max(longest_readable, _longest_run(exposed_readable))
        transient += sum(
            1
            for f in exposed
            if any(f - window <= p < f for p in protected) and any(f < p <= f + window for p in protected)
        )
    return {
        "clips": len(evaluations),
        "overall": _rate(frames),
        "by_class": {
            cls: _rate([o for o in frames if o.cls == cls]) for cls in sorted({o.cls for o in frames})
        },
        "by_size": {
            label: _rate([o for o in frames if o.size == label])
            for _low, _high, label in SIZE_BUCKETS
            if any(o.size == label for o in frames)
        },
        "readable": _rate([o for o in frames if o.readable]),
        "tracks": len(tracks),
        "tracks_ever_leaked": ever_leaked,
        "readable_tracks": readable_tracks,
        "readable_tracks_ever_leaked": readable_ever_leaked,
        "longest_exposure_frames": longest,
        "longest_exposure_frames_readable": longest_readable,
        "transient_exposures": transient,
        "over_blur_ratio": round(sum(e.over_blur_ratio for e in evaluations) / max(1, len(evaluations)), 5),
    }


def check_gate(summary: dict[str, Any], thresholds: dict[str, float]) -> list[str]:
    """Failures of the privacy gate (``benchmarks/privacy-thresholds.yaml``, ``annotated_dataset``).

    Returns:
        Human-readable failures; empty when the gate passes.
    """
    failures = []
    rate = float(summary["readable"]["leakage_rate"])
    if rate > thresholds["max_leakage_rate_readable"]:
        failures.append(
            f"leakage rate of readable objects {rate:.2%} > {thresholds['max_leakage_rate_readable']:.2%}"
        )
    run = int(summary["longest_exposure_frames_readable"])
    if run > thresholds["max_consecutive_exposed_frames_readable"]:
        failures.append(
            f"a readable object stayed exposed {run} consecutive frames "
            f"> {int(thresholds['max_consecutive_exposed_frames_readable'])}"
        )
    return failures
