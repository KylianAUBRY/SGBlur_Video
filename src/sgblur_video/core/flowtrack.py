"""Optical-flow tracker for small, fast objects (``tracker_type: flow``).

On 8K 360° video filmed from a car, a plate parked a few metres away moves more
than its own width between two frames (about 70 px for a 50 px plate). Trackers
that associate boxes by overlap (ByteTrack, TrackTrack) see no overlap and start
a new track on every frame: on a parking-lot sample, 261 of 265 plate detections
were left untracked, one plate became a dozen fragments, and each fragment was
padded on both sides by post-processing.

Here every track's box is first moved by the optical flow of the image around it
(pyramidal Lucas-Kanade between the previous and the current tracking frames,
with a forward-backward check), then matched to the detections of the frame by
centre distance in box sizes (optimal assignment). The flow follows the parallax
of each object, which no single camera-motion model describes in an
equirectangular frame. Distances wrap around the 0°/360° seam.

Like the other trackers it only assigns ids: the boxes blurred are always the
detector's boxes.
"""

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import lap
import numpy as np
import numpy.typing as npt

from sgblur_video.core.geometry import Box

#: Points sampled on a ``_GRID`` x ``_GRID`` grid over each track's box, enlarged by ``context``
#: box sizes on each side (a plate alone is a few pixels at tracking resolution).
_GRID = 4
#: Fewer good points than this: the flow is not trusted and the box keeps its last motion.
_MIN_POINTS = 3
#: Two boxes whose areas differ by more than this factor are never the same object.
_MAX_AREA_RATIO = 4.0
_UNMATCHED = 1e6
_LK_PARAMS = {
    "winSize": (15, 15),
    "maxLevel": 3,
    "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
}


@dataclass
class _Track:
    id: int
    box: Box
    velocity: tuple[float, float] = (0.0, 0.0)
    missed: int = 0


class FlowTracker:
    """Track boxes of one class group at tracking resolution.

    Args:
        max_missed: Frames a track survives without detection (its box keeps following the flow).
        gate: Largest centre distance, in box sizes, between a track's predicted box and a
            detection it is matched with, for boxes up to ``gate_cap`` pixels (small, fast
            objects). Larger boxes get ``large_gate`` box sizes, at least ``gate * gate_cap``
            pixels: a whole box size would let a face track jump to the next face in a group.
            The gate grows by half of it per missed frame.
        gate_cap: Box size, in tracking pixels, up to which ``gate`` applies.
        large_gate: Gate of large boxes, in box sizes.
        frame_size: ``(width, height)`` of the tracking frames.
        wrap: The frames are 360° equirectangular: distances wrap around horizontally.
        min_size: Smallest box side considered, in tracking pixels (tiny boxes of distant objects).
        fb_error: Largest forward-backward flow error of a point that is kept, in tracking pixels.
        context: Margin around the box where the flow is measured, in box sizes on each side.
    """

    def __init__(
        self,
        *,
        max_missed: int,
        gate: float,
        frame_size: tuple[int, int],
        gate_cap: float = 32.0,
        large_gate: float = 0.5,
        wrap: bool = False,
        min_size: float = 4.0,
        fb_error: float = 2.0,
        context: float = 1.0,
    ) -> None:
        self._max_missed = max_missed
        self._gate = gate
        self._gate_cap = gate_cap
        self._large_gate = large_gate
        self._width, self._height = frame_size
        self._wrap = wrap
        self._min_size = min_size
        self._fb_error = fb_error
        self._context = context
        self._tracks: list[_Track] = []
        self._ids = itertools.count(1)

    def update(
        self,
        boxes: Sequence[Box],
        gray: npt.NDArray[np.uint8] | None,
        previous: npt.NDArray[np.uint8] | None,
    ) -> list[int]:
        """Track the boxes of one frame and return their track ids (one per box, in order).

        Args:
            boxes: Detections of the frame at tracking resolution.
            gray: The frame in grayscale (``None``: no flow, boxes keep their last motion).
            previous: The previous frame in grayscale, or ``None``.
        """
        if self._tracks:
            if gray is not None and previous is not None:
                self._propagate(previous, gray)
            else:
                for track in self._tracks:
                    track.box = _shift(track.box, *track.velocity)
        ids = self._match(boxes)
        self._tracks = [t for t in self._tracks if t.missed <= self._max_missed]
        return ids

    def _propagate(self, previous: npt.NDArray[np.uint8], gray: npt.NDArray[np.uint8]) -> None:
        """Move every track's box by the median optical flow of the image around it."""
        grids = [self._grid(track.box) for track in self._tracks]
        points = np.concatenate(grids).reshape(-1, 1, 2).astype(np.float32)
        moved, status, _ = cv2.calcOpticalFlowPyrLK(previous, gray, points, None, **_LK_PARAMS)  # type: ignore[call-overload]
        back, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, previous, moved, None, **_LK_PARAMS)  # type: ignore[call-overload]
        error = np.linalg.norm((back - points).reshape(-1, 2), axis=1)
        good = (status.ravel() == 1) & (back_status.ravel() == 1) & (error < self._fb_error)
        motion = (moved - points).reshape(-1, 2)
        start = 0
        for track, grid in zip(self._tracks, grids, strict=True):
            end = start + len(grid)
            kept = motion[start:end][good[start:end]]
            if len(kept) >= _MIN_POINTS:
                dx, dy = (float(v) for v in np.median(kept, axis=0))
                track.velocity = (dx, dy)
            track.box = _shift(track.box, *track.velocity)
            start = end

    def _grid(self, box: Box) -> npt.NDArray[np.float32]:
        """Sample points over the box and its surroundings, inside the frame."""
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        half_w = max(box[2] - box[0], self._min_size) * (0.5 + self._context)
        half_h = max(box[3] - box[1], self._min_size) * (0.5 + self._context)
        if self._wrap:
            cx %= self._width
        xs = np.clip(np.linspace(cx - half_w, cx + half_w, _GRID), 0, self._width - 1)
        ys = np.clip(np.linspace(cy - half_h, cy + half_h, _GRID), 0, self._height - 1)
        return np.array([(x, y) for y in ys for x in xs], dtype=np.float32)

    def _distance(self, track: Box, box: Box) -> tuple[float, float]:
        """Centre distance in pixels (``inf`` when the areas are too different), and the box size."""
        tw, th = max(track[2] - track[0], self._min_size), max(track[3] - track[1], self._min_size)
        bw, bh = max(box[2] - box[0], self._min_size), max(box[3] - box[1], self._min_size)
        size = max(tw, th, bw, bh)
        ratio = (tw * th) / (bw * bh)
        if not 1 / _MAX_AREA_RATIO <= ratio <= _MAX_AREA_RATIO:
            return math.inf, size
        dx = (box[0] + box[2] - track[0] - track[2]) / 2
        dy = (box[1] + box[3] - track[1] - track[3]) / 2
        if self._wrap:
            dx = (dx + self._width / 2) % self._width - self._width / 2
        return math.hypot(dx, dy), size

    def _limit(self, size: float, missed: int) -> float:
        """Largest matching distance in pixels for a box size (see ``gate``)."""
        base = max(self._gate * min(size, self._gate_cap), self._large_gate * size)
        return base * (1 + 0.5 * missed)

    def _match(self, boxes: Sequence[Box]) -> list[int]:
        """Assign detections to tracks (optimal assignment), start tracks for the others."""
        ids = [0] * len(boxes)
        matched: set[int] = set()
        if self._tracks and boxes:
            cost = np.full((len(self._tracks), len(boxes)), _UNMATCHED)
            for i, track in enumerate(self._tracks):
                for j, box in enumerate(boxes):
                    distance, size = self._distance(track.box, box)
                    if distance <= self._limit(size, track.missed):
                        cost[i, j] = distance / size
            _total, rows, _cols = lap.lapjv(cost, extend_cost=True, cost_limit=_UNMATCHED / 2)
            for i, j in enumerate(rows):
                if j < 0:
                    continue
                track = self._tracks[i]
                track.box, track.missed = boxes[j], -1
                ids[j] = track.id
                matched.add(j)
        for track in self._tracks:
            track.missed += 1
        for j, box in enumerate(boxes):
            if j not in matched:
                track = _Track(next(self._ids), box)
                self._tracks.append(track)
                ids[j] = track.id
        return ids


def _shift(box: Box, dx: float, dy: float) -> Box:
    return (box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy)
