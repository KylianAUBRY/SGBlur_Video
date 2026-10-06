"""Wrap-around geometry for equirectangular frames.

In an equirectangular frame, the left edge (x = 0) and the right edge (x = w)
are the same meridian. An object crossing it appears cut in two, and a box can
be described with ``x2 > w`` (it continues on the left side). These helpers
keep boxes comparable across the seam (``docs/adr/0007-360-video.md``).

Example:
    >>> normalize((7950.0, 10.0, 8050.0, 60.0), 8000)
    (7950.0, 10.0, 8050.0, 60.0)
    >>> split((7950.0, 10.0, 8050.0, 60.0), 8000)
    [(7950.0, 10.0, 8000.0, 60.0), (0.0, 10.0, 50.0, 60.0)]
"""

import math

from sgblur_video.core.geometry import Box, iou, translate


def shift_x(box: Box, dx: float) -> Box:
    """Move a box horizontally."""
    return translate(box, dx, 0.0)


def normalize(box: Box, width: float) -> Box:
    """Shift a box by a multiple of ``width`` so that ``0 <= x1 < width`` (``x2`` may exceed ``width``)."""
    turns = math.floor(box[0] / width)
    return shift_x(box, -turns * width) if turns else box


def unwrap_towards(box: Box, reference: Box, width: float) -> Box:
    """Shift ``box`` by a multiple of ``width`` to put it as close as possible to ``reference``.

    Used to keep a track continuous when an object crosses the seam: its
    coordinates may then leave ``[0, width)``, which is fine for interpolation.
    """
    turns = round(((reference[0] + reference[2]) - (box[0] + box[2])) / 2 / width)
    return shift_x(box, turns * width) if turns else box


def circular_dx(a: float, b: float, width: float) -> float:
    """Signed shortest horizontal difference ``b - a`` on a circle of circumference ``width``."""
    return (b - a + width / 2) % width - width / 2


def wrapped_iou(a: Box, b: Box, width: float) -> float:
    """IoU after bringing ``b`` next to ``a`` across the seam if that is shorter."""
    return iou(a, unwrap_towards(b, a, width))


def split(box: Box, width: float) -> list[Box]:
    """Visible parts of a box inside ``[0, width)``: one part, or two when it crosses the seam."""
    x1, y1, x2, y2 = normalize(box, width)
    if x2 <= width:
        return [(x1, y1, x2, y2)]
    return [(x1, y1, width, y2), (0.0, y1, min(x2 - width, x1), y2)]


def copies(box: Box, width: float) -> list[Box]:
    """The box and its copies one turn to the left and right (for drawing or masking near the seam)."""
    base = normalize(box, width)
    return [shift_x(base, -width), base, shift_x(base, width)]
