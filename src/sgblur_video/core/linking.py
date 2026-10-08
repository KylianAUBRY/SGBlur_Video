"""Offline linking: group detections of one object over time into chains.

Used for **traffic signs** only (one Panoramax annotation per physical sign,
``semantics.annotations``) and to pre-annotate benchmark clips
(``bench.clips``). Blurring never uses it: every frame is blurred on its own
detections (``core.postprocess``).

Every tracker track is a fragment, and so is every detection the tracker left
alone. Fragments of the same class group are chained when one starts shortly
after another ends, near where the previous one was heading
(``docs/adr/0011-offline-linking.md``). On 360° video, distances are measured
around the 0°/360° seam, so an object crossing it stays one chain.
"""

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from sgblur_video.core.detections_io import Detections
from sgblur_video.core.geometry import Box, area, center, height, width
from sgblur_video.video360.wrap import unwrap_towards

#: Detections used to estimate where a chain is heading.
VELOCITY_SAMPLES = 5

#: Two fragments are linked only if their box areas differ by less than this factor.
LINK_MAX_AREA_RATIO = 4.0


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

    def append(self, fragment: Fragment, wrap_width: float | None = None) -> None:
        """Link a fragment that starts after the chain ends (unwrapped next to it on 360° video)."""
        observations = fragment.observations
        if wrap_width and self.observations:
            observations = _unwrap_sequence(observations, self.observations[-1].box, wrap_width)
        self.observations.extend(observations)
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


def _unwrap_sequence(
    observations: Sequence[Observation], reference: Box, wrap_width: float
) -> list[Observation]:
    """Shift each observation by whole turns so the sequence stays continuous across the 360° seam."""
    result = []
    previous = reference
    for obs in observations:
        box = unwrap_towards(obs.box, previous, wrap_width)
        result.append(Observation(obs.frame, box, obs.score, obs.cls) if box is not obs.box else obs)
        previous = box
    return result


def fragments_from_detections(
    detections: Detections, groups: Mapping[str, str], *, wrap_width: float | None = None
) -> list[Fragment]:
    """Group detections of the given classes into tracker fragments and orphan fragments.

    Args:
        detections: Content of ``detections.jsonl``.
        groups: Class name to group name, for the classes to keep.
        wrap_width: Frame width of a 360° video: tracks are kept continuous across the seam.

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
        if wrap_width:
            observations = _unwrap_sequence(observations, observations[0].box, wrap_width)
        fragments.append(Fragment(key, groups[observations[0].cls], observations))
    fragments.sort(key=lambda f: (f.start, f.end))
    return fragments


def link_fragments(
    fragments: Sequence[Fragment], *, max_gap: int, max_distance: float, wrap_width: float | None = None
) -> list[Chain]:
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
        wrap_width: Frame width of a 360° video: distances are measured across the 0°/360° seam,
            so an object leaving on the right and coming back on the left stays one chain.

    Returns:
        Chains, in order of creation.
    """
    chains: list[Chain] = []
    active: list[Chain] = []
    synthetic_ids: Counter[str] = Counter()
    for fragment in fragments:
        active = [c for c in active if fragment.start - c.end <= max_gap]
        first_observation = fragment.observations[0]
        best: Chain | None = None
        best_distance = math.inf
        for chain in active:
            gap = fragment.start - chain.end
            if chain.group != fragment.group or gap <= 0:
                continue
            tail = chain.observations[-VELOCITY_SAMPLES:]
            predicted = _shift(tail[-1].box, _velocity(tail), gap)
            first = (
                unwrap_towards(first_observation.box, predicted, wrap_width)
                if wrap_width
                else first_observation.box
            )
            size = max(width(predicted), height(predicted), width(first), height(first), 1.0)
            (px, py), (fx, fy) = center(predicted), center(first)
            distance = math.hypot(px - fx, py - fy) / size
            ratio = max(area(predicted), 1.0) / max(area(first), 1.0)
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
        best.append(fragment, wrap_width)
    return chains
