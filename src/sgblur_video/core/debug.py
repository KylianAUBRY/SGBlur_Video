"""Annotated debug video (CLI ``--debug``, HTTP API ``debug=1``).

The overlay is drawn on the **blurred** frames, downscaled to at most
``DEBUG_MAX_WIDTH``, so a debug video never shows more than the real output.
It shows what the detector found and what the blur covers:

* the **colour** is the class: magenta = face, yellow = plate (blurred);
  blue = sign, cyan = direction sign (never blurred);
* a **solid** outline is a region the model detected on that frame, labelled
  with its class, score and track number;
* a **dashed** outline, without label, is blurred without a detection on that
  frame, deduced from the object's track: interpolated between two
  detections, or padded before the first or after the last one (labelling
  them buried the detections under text on busy streets).
"""

import itertools
import math
from collections import defaultdict
from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

from sgblur_video.core.detections_io import Detections
from sgblur_video.core.geometry import Box, circumscribed_ellipse, iomin, iou
from sgblur_video.core.postprocess import BlurPlan, BlurShape
from sgblur_video.video360.wrap import copies, normalize, unwrap_towards

Colour = tuple[int, int, int]

#: BGR colour per class (CSS: face #ff00ff, plate #ffd700, sign #0080ff, direction #00ffff).
CLASS_COLOURS: dict[str, Colour] = {
    "face": (255, 0, 255),
    "plate": (0, 215, 255),
    "sign": (255, 128, 0),
    "direction": (255, 255, 0),
}
#: Colour of other blurred classes (custom class policy).
OTHER_BLUR_COLOUR: Colour = (0, 200, 0)
#: Colour of other annotated classes.
OTHER_SIGN_COLOUR: Colour = CLASS_COLOURS["sign"]
#: Blur sources drawn dashed: no detection on that frame.
DEDUCED_SOURCES = frozenset({"interpolated", "padded"})
DEBUG_MAX_WIDTH = 1920


@dataclass(frozen=True)
class _Seen:
    """A detection of one frame."""

    cls: str
    box: Box
    score: float
    track_id: str | None


def _number(track_id: str | None) -> str:
    """Short track label: ``face:12`` → ``#12``, ``face:~3`` → ``#~3``."""
    return f" #{track_id.rsplit(':', 1)[-1]}" if track_id else ""


class DebugOverlay:
    """Draw blur shapes and annotated objects on downscaled frames.

    Args:
        plan: Blur plan of the video.
        detections: Detections (scores of blurred regions, signs).
        annotate_classes: Classes drawn as annotations (signs).
    """

    def __init__(self, plan: BlurPlan, detections: Detections, annotate_classes: frozenset[str]) -> None:
        self._plan = plan
        self._seen: dict[int, list[_Seen]] = defaultdict(list)
        self._signs: dict[int, list[_Seen]] = defaultdict(list)
        for frame in detections.frames:
            for det in frame.detections:
                target = self._signs if det.class_ in annotate_classes else self._seen
                target[frame.index].append(_Seen(det.class_, det.box, det.score, det.track_id))

    def draw(self, index: int, image: npt.NDArray[np.uint8], factor: float) -> None:
        """Draw the overlay of frame ``index`` in place.

        Args:
            index: Frame index.
            image: BGR image (downscaled frame).
            factor: Scale from coded-frame pixels to ``image`` pixels.
        """
        thickness = max(1, round(image.shape[1] / 960))
        wrap = self._plan.wrap_width
        for shape in self._plan.shapes(index):
            colour = CLASS_COLOURS.get(shape.cls, OTHER_BLUR_COLOUR)
            dashed = shape.source in DEDUCED_SOURCES
            outline = _ellipse_points if shape.kind == "ellipse" else _rect_points
            for box in copies(shape.box, wrap) if wrap else [shape.box]:
                _outline(image, outline(box, factor), colour, thickness, dashed=dashed)
            if dashed:
                continue
            score = self._score(index, shape)
            text = shape.cls + (f" {score:.2f}" if score is not None else "") + _number(shape.track_id)
            _label(image, text, normalize(shape.box, wrap) if wrap else shape.box, factor, colour)
        for sign in self._signs.get(index, []):
            colour = CLASS_COLOURS.get(sign.cls, OTHER_SIGN_COLOUR)
            _outline(image, _rect_points(sign.box, factor), colour, thickness, dashed=False)
            _label(image, f"{sign.cls} {sign.score:.2f}{_number(sign.track_id)}", sign.box, factor, colour)

    def _score(self, index: int, shape: BlurShape) -> float | None:
        """Score of the detection behind a blurred region (the same-class box overlapping it best)."""
        wrap = self._plan.wrap_width
        best: tuple[float, float] | None = None
        for seen in self._seen.get(index, []):
            if seen.cls != shape.cls:
                continue
            box = unwrap_towards(seen.box, shape.box, wrap) if wrap else seen.box
            if iomin(box, shape.box) < 0.5:
                continue
            overlap = iou(box, shape.box)
            if best is None or overlap > best[0]:
                best = (overlap, seen.score)
        return best[1] if best is not None else None


def _rect_points(box: Box, factor: float) -> npt.NDArray[np.float64]:
    x1, y1, x2, y2 = (v * factor for v in box)
    return np.array([(x1, y1), (x2, y1), (x2, y2), (x1, y2)], dtype=np.float64)


def _ellipse_points(box: Box, factor: float) -> npt.NDArray[np.float64]:
    cx, cy, rx, ry = (v * factor for v in circumscribed_ellipse(box))
    points = cv2.ellipse2Poly((round(cx), round(cy)), (max(1, round(rx)), max(1, round(ry))), 0, 0, 360, 6)
    return np.asarray(points, dtype=np.float64)


def _point(value: npt.NDArray[np.float64]) -> tuple[int, int]:
    return round(float(value[0])), round(float(value[1]))


def _outline(
    image: npt.NDArray[np.uint8],
    points: npt.NDArray[np.float64],
    colour: Colour,
    thickness: int,
    *,
    dashed: bool,
) -> None:
    """Draw a closed polygon, solid or dashed (dashes follow the perimeter, ellipses included)."""
    if not dashed:
        cv2.polylines(image, [np.round(points).astype(np.int32)], True, colour, thickness, cv2.LINE_AA)
        return
    dash = 4 * thickness + 4
    closed = np.vstack([points, points[:1]])
    drawing, left = True, float(dash)
    for start, end in itertools.pairwise(closed):
        length = math.hypot(*(end - start))
        position = 0.0
        while position < length:
            step = min(left, length - position)
            if drawing:
                a = start + (end - start) * (position / length)
                b = start + (end - start) * ((position + step) / length)
                cv2.line(image, _point(a), _point(b), colour, thickness, cv2.LINE_AA)
            position += step
            left -= step
            if left <= 0:
                drawing, left = not drawing, float(dash)


def _label(image: npt.NDArray[np.uint8], text: str, box: Box, factor: float, colour: Colour) -> None:
    """Write ``text`` on a filled tag of ``colour`` above the box (inside it at the top edge)."""
    # Readable once the video is shown in a page (0.8 and bold at 1920 px).
    scale = max(0.5, image.shape[1] / 2400)
    weight = max(1, round(image.shape[1] / 1280))
    (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, weight)
    pad = 3
    x = min(max(0, math.floor(box[0] * factor)), max(0, image.shape[1] - width - 2 * pad))
    top = math.floor(box[1] * factor) - height - baseline - 2 * pad
    if top < 0:
        top = max(0, math.floor(box[1] * factor))
    bottom = top + height + baseline + 2 * pad
    cv2.rectangle(image, (x, top), (x + width + 2 * pad, bottom), colour, cv2.FILLED)
    luminance = (0.0722 * colour[0] + 0.7152 * colour[1] + 0.2126 * colour[2]) / 255
    ink = (0, 0, 0) if luminance > 0.5 else (255, 255, 255)
    origin = (x + pad, top + pad + height)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, ink, weight, cv2.LINE_AA)
