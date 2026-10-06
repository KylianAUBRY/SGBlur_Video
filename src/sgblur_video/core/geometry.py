"""Axis-aligned box geometry.

Boxes are ``(x1, y1, x2, y2)`` tuples of floats in pixels, origin top-left,
``x2 > x1`` and ``y2 > y1``. Plain tuples keep the code simple and fast enough
for the few hundred boxes a frame can hold.

Example:
    >>> iou((0, 0, 10, 10), (5, 0, 15, 10))
    0.3333333333333333
"""

import math

Box = tuple[float, float, float, float]


def width(box: Box) -> float:
    """Width of a box (0 for degenerate boxes)."""
    return max(0.0, box[2] - box[0])


def height(box: Box) -> float:
    """Height of a box (0 for degenerate boxes)."""
    return max(0.0, box[3] - box[1])


def area(box: Box) -> float:
    """Area of a box."""
    return width(box) * height(box)


def intersection(a: Box, b: Box) -> float:
    """Area of the intersection of two boxes."""
    return area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def iou(a: Box, b: Box) -> float:
    """Intersection over union, 0 when both boxes are empty."""
    inter = intersection(a, b)
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def iomin(a: Box, b: Box) -> float:
    """Intersection over the area of the smaller box (1.0 when one contains the other)."""
    smallest = min(area(a), area(b))
    return intersection(a, b) / smallest if smallest > 0 else 0.0


def union_box(boxes: list[Box]) -> Box:
    """Smallest box containing every box of the list.

    Raises:
        ValueError: If the list is empty.
    """
    if not boxes:
        msg = "union of an empty list of boxes"
        raise ValueError(msg)
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def expand(box: Box, ratio: float) -> Box:
    """Enlarge a box by ``ratio`` of its width/height on each side.

    Example:
        >>> expand((10, 10, 20, 30), 0.5)
        (5.0, 0.0, 25.0, 40.0)
    """
    dx, dy = width(box) * ratio, height(box) * ratio
    return (box[0] - dx, box[1] - dy, box[2] + dx, box[3] + dy)


def translate(box: Box, dx: float, dy: float) -> Box:
    """Move a box by ``(dx, dy)``."""
    return (box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy)


def scale(box: Box, sx: float, sy: float) -> Box:
    """Scale box coordinates (change of resolution)."""
    return (box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy)


def clip(box: Box, frame_width: float, frame_height: float) -> Box:
    """Clip a box to the frame; the result may be degenerate if the box is outside."""
    return (
        min(max(box[0], 0.0), frame_width),
        min(max(box[1], 0.0), frame_height),
        min(max(box[2], 0.0), frame_width),
        min(max(box[3], 0.0), frame_height),
    )


def lerp(a: Box, b: Box, t: float) -> Box:
    """Linear interpolation between two boxes (``t=0`` gives ``a``, ``t=1`` gives ``b``)."""
    return (
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
        a[3] + (b[3] - a[3]) * t,
    )


def center(box: Box) -> tuple[float, float]:
    """Centre of a box."""
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def circumscribed_ellipse(box: Box) -> tuple[float, float, float, float]:
    """Ellipse with the box's aspect ratio passing through its four corners.

    An ellipse *inscribed* in a box leaves its corners (21.5 % of the area)
    uncovered; the circumscribed one, with semi-axes √2 × the half-sides,
    contains the whole box.

    Returns:
        ``(cx, cy, rx, ry)``: centre and semi-axes.
    """
    cx, cy = center(box)
    return (cx, cy, width(box) / 2 * math.sqrt(2), height(box) / 2 * math.sqrt(2))


def round_out(box: Box) -> tuple[int, int, int, int]:
    """Integer pixel box covering ``box`` entirely (floor of the start, ceil of the end)."""
    return (math.floor(box[0]), math.floor(box[1]), math.ceil(box[2]), math.ceil(box[3]))
