"""Tests of sgblur_video.core.geometry."""

import math

import pytest

from sgblur_video.core.geometry import (
    area,
    circumscribed_ellipse,
    clip,
    expand,
    iomin,
    iou,
    lerp,
    round_out,
    union_box,
)


def test_iou_and_iomin() -> None:
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)
    assert iomin((0, 0, 100, 100), (10, 10, 20, 20)) == pytest.approx(1.0)
    assert iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0.0


def test_union_expand_clip_lerp() -> None:
    assert union_box([(0, 0, 1, 1), (5, -2, 6, 3)]) == (0, -2, 6, 3)
    with pytest.raises(ValueError, match="empty"):
        union_box([])
    assert expand((10, 10, 20, 30), 0.5) == (5.0, 0.0, 25.0, 40.0)
    assert clip((-5, -5, 50, 50), 40, 30) == (0, 0, 40, 30)
    assert lerp((0, 0, 10, 10), (10, 0, 20, 10), 0.5) == (5, 0, 15, 10)
    assert round_out((0.2, 0.7, 9.1, 9.9)) == (0, 0, 10, 10)
    assert area((0, 0, -1, 5)) == 0.0


def test_circumscribed_ellipse_contains_box_corners() -> None:
    box = (10.0, 20.0, 50.0, 40.0)
    cx, cy, rx, ry = circumscribed_ellipse(box)
    for x, y in ((box[0], box[1]), (box[2], box[1]), (box[0], box[3]), (box[2], box[3])):
        assert ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 == pytest.approx(1.0)
    assert rx == pytest.approx(20 * math.sqrt(2))
