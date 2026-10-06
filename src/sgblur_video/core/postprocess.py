"""Turn ``detections.jsonl`` into a blur plan: what to blur on every frame.

This module is where most of the privacy guarantees live
(``docs/design/pipeline.md`` §3). It runs on CPU, without the video, and is
fully unit-tested with synthetic tracks.

Steps for ``blur`` classes (faces, plates):

1. **Fragments** — every tracker track is a fragment, and so is every
   detection the tracker left alone (orphan).
2. **Offline linking** — the whole video is known, so fragments of the same
   class group are chained when one starts shortly after another ends
   (``LINK_MAX_GAP_S``) near where the previous one was heading
   (``LINK_MAX_DISTANCE`` box sizes). Small objects far from the camera are
   often detected only every other frame and trackers fail to follow them;
   linking turns those flickering detections back into continuous chains.
3. **Selection** — a chain is blurred if it contains a detection ≥ ``CONF_BLUR``
   (so lower-score detections of a confirmed object are blurred too).
4. **Gap filling** — frames between two detections of a chain (up to
   ``MAX_INTERPOLATION_GAP_S``) get a linearly interpolated box.
5. **Envelope smoothing** — a 5-frame moving average removes jitter; the final
   box is the union of the smoothed and original boxes, so it never shrinks.
6. **Temporal padding** — ``BLUR_TEMPORAL_PADDING_FRAMES`` frames before the
   first and after the last detection of each chain segment are blurred too,
   with the box extrapolated from the chain's velocity and growing by
   ``BLUR_PADDING_GROWTH`` per frame.
7. **Margin and shape** — boxes are enlarged by ``BLUR_BOX_MARGIN``; faces are
   blurred with the ellipse circumscribing that box, other classes with the box.

360° seam stitching (step 7) and sign deduplication (step 5) plug in here later.
"""

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from sgblur_video.config import ClassAction, Settings
from sgblur_video.core.detect import class_groups
from sgblur_video.core.detections_io import Detections
from sgblur_video.core.geometry import Box, area, center, clip, expand, height, lerp, union_box, width

Source = Literal["detected", "interpolated", "padded", "orphan"]

#: Classes blurred with an ellipse; every other blurred class uses a rectangle.
ELLIPSE_CLASSES = frozenset({"face"})

#: Window of the envelope smoothing, in frames (odd).
SMOOTHING_WINDOW = 5

#: Detections used to estimate a chain's velocity at each end.
VELOCITY_SAMPLES = 5

#: Two fragments are linked only if their box areas differ by less than this factor.
LINK_MAX_AREA_RATIO = 4.0

# When a frame holds several boxes of one chain, keep the most reliable source.
_SOURCE_PRIORITY: dict[Source, int] = {"detected": 3, "orphan": 3, "interpolated": 2, "padded": 1}


@dataclass(frozen=True)
class BlurShape:
    """One region to blur on one frame.

    Attributes:
        kind: ``ellipse`` (circumscribing ``box``) or ``rect``.
        box: Margin-enlarged box in coded-frame pixels (may extend past the frame edges).
        cls: Class name.
        source: ``detected``, ``interpolated``, ``padded``, or ``orphan`` (isolated detection).
        track_id: Chain identifier (tracker id of its first track, or ``<group>:~N``).
    """

    kind: Literal["ellipse", "rect"]
    box: Box
    cls: str
    source: Source
    track_id: str


@dataclass
class BlurPlan:
    """Regions to blur, per frame index.

    Attributes:
        frame_count: Number of frames covered by the plan.
        frames: Shapes per frame index (frames without shapes are absent).
        stats: Counters for the job metadata.
        chain_scores: Best detection score of each blurred chain (``keep=1`` keeps low-score chains).
    """

    frame_count: int
    frames: dict[int, list[BlurShape]] = field(default_factory=dict)
    stats: dict[str, int] = field(default_factory=dict)
    chain_scores: dict[str, float] = field(default_factory=dict)

    def shapes(self, index: int) -> list[BlurShape]:
        """Shapes to blur on frame ``index`` (empty list if none)."""
        return self.frames.get(index, [])


@dataclass(frozen=True)
class Observation:
    """One detection of an object on one frame."""

    frame: int
    box: Box
    score: float
    cls: str


@dataclass
class Fragment:
    """Consecutive observations the tracker attributed to one object (or a single orphan).

    Attributes:
        key: Tracker id (``face:7``) or ``None`` for an orphan.
        group: Class group (``face``, ``plate``, ``signage``).
        observations: Observations sorted by frame.
    """

    key: str | None
    group: str
    observations: list[Observation]

    @property
    def start(self) -> int:
        """First frame."""
        return self.observations[0].frame

    @property
    def end(self) -> int:
        """Last frame."""
        return self.observations[-1].frame


@dataclass
class Chain:
    """Fragments linked into one object over time.

    Fragments are appended in time order and never overlap, so ``observations``
    stays sorted by frame.

    Attributes:
        id: Stable identifier used in the debug video and statistics.
        group: Class group.
        observations: All observations of the linked fragments, sorted by frame.
        fragments: Number of linked fragments.
    """

    id: str
    group: str
    observations: list[Observation] = field(default_factory=list)
    fragments: int = 0

    @property
    def end(self) -> int:
        """Last frame."""
        return self.observations[-1].frame

    @property
    def cls(self) -> str:
        """Majority class of the chain."""
        return Counter(o.cls for o in self.observations).most_common(1)[0][0]

    def append(self, fragment: Fragment) -> None:
        """Link a fragment that starts after the chain ends."""
        self.observations.extend(fragment.observations)
        self.fragments += 1


def _velocity(observations: Sequence[Observation]) -> Box:
    """Mean per-frame displacement of each box coordinate (zero for a single observation)."""
    frames = observations[-1].frame - observations[0].frame if observations else 0
    if frames <= 0:
        return (0.0, 0.0, 0.0, 0.0)
    first, last = observations[0].box, observations[-1].box
    return (
        (last[0] - first[0]) / frames,
        (last[1] - first[1]) / frames,
        (last[2] - first[2]) / frames,
        (last[3] - first[3]) / frames,
    )


def _shift(box: Box, velocity: Box, frames: int) -> Box:
    return (
        box[0] + velocity[0] * frames,
        box[1] + velocity[1] * frames,
        box[2] + velocity[2] * frames,
        box[3] + velocity[3] * frames,
    )


def fragments_from_detections(detections: Detections, groups: Mapping[str, str]) -> list[Fragment]:
    """Group detections of the given classes into tracker fragments and orphan fragments.

    Args:
        detections: Content of ``detections.jsonl``.
        groups: Class name to group name, for the classes to keep.

    Returns:
        Fragments sorted by start frame.
    """
    tracked: dict[str, list[Observation]] = {}
    fragments: list[Fragment] = []
    for frame in detections.frames:
        for det in frame.detections:
            if det.class_ not in groups:
                continue
            obs = Observation(frame.index, det.box, det.score, det.class_)
            if det.track_id is None:
                fragments.append(Fragment(None, groups[det.class_], [obs]))
            else:
                tracked.setdefault(det.track_id, []).append(obs)
    for key, observations in tracked.items():
        observations.sort(key=lambda o: o.frame)
        fragments.append(Fragment(key, groups[observations[0].cls], observations))
    fragments.sort(key=lambda f: (f.start, f.end))
    return fragments


def link_fragments(fragments: Sequence[Fragment], *, max_gap: int, max_distance: float) -> list[Chain]:
    """Chain fragments that are the same object seen with interruptions.

    A fragment is appended to the chain of the same group that ended at most
    ``max_gap`` frames before it starts, whose extrapolated position is the
    closest to the fragment's first box, provided the distance between centres
    is at most ``max_distance`` box sizes (plus 10 % per frame of gap) and the
    areas differ by less than ``LINK_MAX_AREA_RATIO``.

    Args:
        fragments: Fragments sorted by start frame.
        max_gap: Maximum number of frames between two linked fragments.
        max_distance: Maximum centre distance, in box sizes.

    Returns:
        Chains, in order of creation.
    """
    chains: list[Chain] = []
    active: list[Chain] = []
    synthetic_ids: Counter[str] = Counter()
    for fragment in fragments:
        active = [c for c in active if fragment.start - c.end <= max_gap]
        first = fragment.observations[0]
        best: Chain | None = None
        best_distance = math.inf
        for chain in active:
            gap = fragment.start - chain.end
            if chain.group != fragment.group or gap <= 0:
                continue
            tail = chain.observations[-VELOCITY_SAMPLES:]
            predicted = _shift(tail[-1].box, _velocity(tail), gap)
            size = max(width(predicted), height(predicted), width(first.box), height(first.box), 1.0)
            (px, py), (fx, fy) = center(predicted), center(first.box)
            distance = math.hypot(px - fx, py - fy) / size
            ratio = max(area(predicted), 1.0) / max(area(first.box), 1.0)
            if (
                distance <= max_distance * (1 + 0.1 * gap)
                and 1 / LINK_MAX_AREA_RATIO <= ratio <= LINK_MAX_AREA_RATIO
                and distance < best_distance
            ):
                best, best_distance = chain, distance
        if best is None:
            synthetic_ids[fragment.group] += 1
            chain_id = fragment.key or f"{fragment.group}:~{synthetic_ids[fragment.group]}"
            best = Chain(chain_id, fragment.group)
            chains.append(best)
            active.append(best)
        best.append(fragment)
    return chains


def _segments(observations: list[Observation], max_gap_frames: int) -> list[list[Observation]]:
    """Split a chain where two observations are further apart than the interpolation limit."""
    segments: list[list[Observation]] = [[observations[0]]]
    for obs in observations[1:]:
        if obs.frame - segments[-1][-1].frame > max_gap_frames + 1:
            segments.append([obs])
        else:
            segments[-1].append(obs)
    return segments


def _densify(segment: list[Observation]) -> tuple[list[int], list[Box], list[bool]]:
    """Every frame from the first to the last observation (union of same-frame boxes, gaps interpolated).

    Returns:
        Frames, boxes, and whether each box was observed (``False`` = interpolated).
    """
    per_frame: dict[int, Box] = {}
    for obs in segment:
        per_frame[obs.frame] = (
            union_box([per_frame[obs.frame], obs.box]) if obs.frame in per_frame else obs.box
        )
    observed = sorted(per_frame)
    frames: list[int] = []
    boxes: list[Box] = []
    flags: list[bool] = []
    for current, following in zip(observed, [*observed[1:], None], strict=True):
        frames.append(current)
        boxes.append(per_frame[current])
        flags.append(True)
        if following is None:
            continue
        gap = following - current
        for step in range(1, gap):
            frames.append(current + step)
            boxes.append(lerp(per_frame[current], per_frame[following], step / gap))
            flags.append(False)
    return frames, boxes, flags


def _smooth(boxes: list[Box]) -> list[Box]:
    """Envelope smoothing: moving average, then union with the original box."""
    half = SMOOTHING_WINDOW // 2
    result = []
    for i, box in enumerate(boxes):
        window = boxes[max(0, i - half) : i + half + 1]
        mean: Box = (
            sum(b[0] for b in window) / len(window),
            sum(b[1] for b in window) / len(window),
            sum(b[2] for b in window) / len(window),
            sum(b[3] for b in window) / len(window),
        )
        result.append(union_box([mean, box]))
    return result


def build_blur_plan(
    detections: Detections,
    settings: Settings,
    *,
    frame_size: tuple[int, int],
    fps: float,
) -> BlurPlan:
    """Compute the blur plan of a video from its detections.

    Args:
        detections: Content of ``detections.jsonl``.
        settings: Thresholds, linking, margins and padding.
        frame_size: Coded frame ``(width, height)``.
        fps: Average frame rate (converts durations to frames).

    Returns:
        The blur plan, with statistics.
    """
    blur_classes = settings.classes_with(ClassAction.BLUR)
    groups = {
        name: group for name, group in class_groups(settings.class_policy).items() if name in blur_classes
    }
    frame_count = len(detections.frames)
    max_gap_frames = max(0, round(settings.max_interpolation_gap_s * fps))
    padding, growth = settings.blur_temporal_padding_frames, settings.blur_padding_growth

    fragments = fragments_from_detections(detections, groups)
    chains = link_fragments(
        fragments,
        max_gap=max(1, round(settings.link_max_gap_s * fps)),
        max_distance=settings.link_max_distance,
    )
    stats: Counter[str] = Counter(fragments=len(fragments))
    chain_scores: dict[str, float] = {}
    regions: dict[tuple[int, str], tuple[Box, Source, str]] = {}

    def add(frame: int, key: str, box: Box, source: Source, cls: str) -> None:
        if not 0 <= frame < frame_count:
            return
        existing = regions.get((frame, key))
        if existing is None:
            regions[(frame, key)] = (box, source, cls)
            return
        best = source if _SOURCE_PRIORITY[source] > _SOURCE_PRIORITY[existing[1]] else existing[1]
        regions[(frame, key)] = (union_box([existing[0], box]), best, cls)

    for chain in chains:
        observations = chain.observations
        if max(o.score for o in observations) < settings.conf_blur:
            stats["chains_below_threshold"] += 1
            continue
        cls = chain.cls
        stats[f"chains_{cls}"] += 1
        chain_scores[chain.id] = max(o.score for o in observations)
        isolated = len(observations) == 1
        for segment in _segments(observations, max_gap_frames):
            frames, boxes, observed = _densify(segment)
            smoothed = _smooth(boxes)
            for frame, box, was_observed in zip(frames, smoothed, observed, strict=True):
                source: Source = ("orphan" if isolated else "detected") if was_observed else "interpolated"
                add(frame, chain.id, box, source, cls)
            v_head = _velocity(segment[:VELOCITY_SAMPLES])
            v_tail = _velocity(segment[-VELOCITY_SAMPLES:])
            for step in range(1, padding + 1):
                grow = growth * step / 2
                add(
                    frames[0] - step,
                    chain.id,
                    expand(_shift(smoothed[0], v_head, -step), grow),
                    "padded",
                    cls,
                )
                add(
                    frames[-1] + step,
                    chain.id,
                    expand(_shift(smoothed[-1], v_tail, step), grow),
                    "padded",
                    cls,
                )

    frame_width, frame_height = frame_size
    plan = BlurPlan(frame_count=frame_count, chain_scores=chain_scores)
    for (frame, key), (box, source, cls) in sorted(regions.items(), key=lambda item: item[0]):
        enlarged = expand(box, settings.blur_box_margin)
        if area(clip(enlarged, frame_width, frame_height)) <= 0:
            continue
        kind: Literal["ellipse", "rect"] = "ellipse" if cls in ELLIPSE_CLASSES else "rect"
        plan.frames.setdefault(frame, []).append(BlurShape(kind, enlarged, cls, source, key))
        stats[f"boxes_{source}"] += 1
    stats["chains"] = len(chains)
    stats["frames_with_blur"] = len(plan.frames)
    plan.stats = dict(stats)
    return plan
