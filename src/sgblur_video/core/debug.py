"""Annotated debug video (CLI only, never exposed through the HTTP API).

The overlay is drawn on the **blurred** frames, so a debug video never shows
more than the real output. Colours:

* green — detected (blurred) region; yellow — interpolated; orange — padded;
  magenta — orphan;
* blue — sign / direction sign (never blurred), with its track id.
"""

import math
from collections import defaultdict
from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

from sgblur_video.core.detections_io import Detections
from sgblur_video.core.geometry import Box, circumscribed_ellipse
from sgblur_video.core.postprocess import BlurPlan
from sgblur_video.video360.wrap import copies, normalize

#: BGR colours per blur source.
SOURCE_COLOURS: dict[str, tuple[int, int, int]] = {
    "detected": (0, 200, 0),
    "interpolated": (0, 220, 255),
    "padded": (0, 140, 255),
    "orphan": (255, 0, 255),
}
SIGN_COLOUR = (255, 120, 0)
DEBUG_MAX_WIDTH = 1920


@dataclass(frozen=True)
class _Label:
    box: Box
    text: str


class DebugOverlay:
    """Draw blur shapes and annotated objects on downscaled frames.

    Args:
        plan: Blur plan of the video.
        detections: Detections (for signs and their track ids).
        annotate_classes: Classes drawn as annotations (signs).
    """

    def __init__(self, plan: BlurPlan, detections: Detections, annotate_classes: frozenset[str]) -> None:
        self._plan = plan
        self._signs: dict[int, list[_Label]] = defaultdict(list)
        for frame in detections.frames:
            for det in frame.detections:
                if det.class_ in annotate_classes:
                    label = f"{det.class_} {det.track_id or '-'} {det.score:.2f}"
                    self._signs[frame.index].append(_Label(det.box, label))

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
            colour = SOURCE_COLOURS[shape.source]
            for box in copies(shape.box, wrap) if wrap else [shape.box]:
                self._draw_shape(image, shape.kind, box, factor, colour, thickness)
            text = f"{shape.cls} {shape.track_id} {shape.source[0]}"
            self._text(image, text, normalize(shape.box, wrap) if wrap else shape.box, factor, colour)
        for label in self._signs.get(index, []):
            x1, y1, x2, y2 = (round(v * factor) for v in label.box)
            cv2.rectangle(image, (x1, y1), (x2, y2), SIGN_COLOUR, thickness)
            self._text(image, label.text, label.box, factor, SIGN_COLOUR)

    @staticmethod
    def _draw_shape(
        image: npt.NDArray[np.uint8],
        kind: str,
        box: Box,
        factor: float,
        colour: tuple[int, int, int],
        thickness: int,
    ) -> None:
        if kind == "ellipse":
            cx, cy, rx, ry = circumscribed_ellipse(box)
            cv2.ellipse(
                image,
                (round(cx * factor), round(cy * factor)),
                (max(1, round(rx * factor)), max(1, round(ry * factor))),
                0,
                0,
                360,
                colour,
                thickness,
            )
        else:
            x1, y1, x2, y2 = (round(v * factor) for v in box)
            cv2.rectangle(image, (x1, y1), (x2, y2), colour, thickness)

    @staticmethod
    def _text(
        image: npt.NDArray[np.uint8], text: str, box: Box, factor: float, colour: tuple[int, int, int]
    ) -> None:
        scale = max(0.4, image.shape[1] / 3200)
        origin = (max(0, math.floor(box[0] * factor)), max(12, math.floor(box[1] * factor) - 4))
        cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)
