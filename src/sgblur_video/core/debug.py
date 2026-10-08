"""Annotated debug video (CLI ``--debug``, HTTP API ``debug=1``).

The overlay is drawn on the **blurred** frames, downscaled to at most
``DEBUG_MAX_WIDTH``, so a debug video never shows more than the real output.
It shows what the detector found on each frame:

* the **colour** is the class: magenta = face, yellow = plate (blurred);
  blue = sign, cyan = direction sign (never blurred);
* every region blurred on a frame is a detection of that frame, outlined and
  labelled with its class and score; signs also show their track number (one
  annotation per track).
"""

import math
from collections import defaultdict
from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

from sgblur_video.core.detections_io import Detections
from sgblur_video.core.geometry import Box
from sgblur_video.core.postprocess import BlurPlan
from sgblur_video.video360.wrap import copies, normalize

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
DEBUG_MAX_WIDTH = 1920


@dataclass(frozen=True)
class _Seen:
    """A sign detection of one frame."""

    cls: str
    box: Box
    score: float
    track_id: str | None


def _number(track_id: str | None) -> str:
    """Short track label: ``signage:12`` → ``#12``."""
    return f" #{track_id.rsplit(':', 1)[-1]}" if track_id else ""


class DebugOverlay:
    """Draw blurred regions and annotated objects on downscaled frames.

    Args:
        plan: Blur plan of the video.
        detections: Detections (signs).
        annotate_classes: Classes drawn as annotations (signs).
    """

    def __init__(self, plan: BlurPlan, detections: Detections, annotate_classes: frozenset[str]) -> None:
        self._plan = plan
        self._signs: dict[int, list[_Seen]] = defaultdict(list)
        for frame in detections.frames:
            for det in frame.detections:
                if det.class_ in annotate_classes:
                    self._signs[frame.index].append(_Seen(det.class_, det.box, det.score, det.track_id))

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
            for box in copies(shape.box, wrap) if wrap else [shape.box]:
                _outline(image, box, factor, colour, thickness)
            text = f"{shape.cls} {shape.score:.2f}"
            _label(image, text, normalize(shape.box, wrap) if wrap else shape.box, factor, colour)
        for sign in self._signs.get(index, []):
            colour = CLASS_COLOURS.get(sign.cls, OTHER_SIGN_COLOUR)
            _outline(image, sign.box, factor, colour, thickness)
            _label(image, f"{sign.cls} {sign.score:.2f}{_number(sign.track_id)}", sign.box, factor, colour)


def _outline(image: npt.NDArray[np.uint8], box: Box, factor: float, colour: Colour, thickness: int) -> None:
    """Draw the outline of a box."""
    x1, y1, x2, y2 = (round(v * factor) for v in box)
    cv2.rectangle(image, (x1, y1), (x2, y2), colour, thickness, cv2.LINE_AA)


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
