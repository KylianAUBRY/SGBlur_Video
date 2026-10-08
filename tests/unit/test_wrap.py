"""Tests of 360° wrap-around geometry and seam-aware detection helpers."""

import numpy as np
import pytest

from sgblur_video.config import DEFAULT_CLASS_POLICY, DetectProfile
from sgblur_video.core.detect import Detection, build_plan, crop_wrapped, merge_detections, pad_circular
from sgblur_video.video360.wrap import circular_dx, copies, normalize, split, unwrap_towards, wrapped_iou

W = 1000


def test_normalize_split_and_copies() -> None:
    assert normalize((1010, 0, 1050, 10), W) == (10, 0, 50, 10)
    assert normalize((-30, 0, 20, 10), W) == (970, 0, 1020, 10)
    assert split((970, 0, 1020, 10), W) == [(970, 0, W, 10), (0.0, 0, 20, 10)]
    assert split((100, 0, 200, 10), W) == [(100, 0, 200, 10)]
    assert [b[0] for b in copies((970, 0, 1020, 10), W)] == [-30, 970, 1970]


def test_distances_across_the_seam() -> None:
    assert circular_dx(990, 10, W) == pytest.approx(20)
    assert circular_dx(10, 990, W) == pytest.approx(-20)
    assert unwrap_towards((5, 0, 25, 10), (980, 0, 1000, 10), W) == (1005, 0, 1025, 10)
    assert wrapped_iou((990, 0, 1010, 10), (-10, 0, 10, 10), W) == pytest.approx(1.0)


def test_circular_padding_and_wrapped_crops() -> None:
    image = np.arange(10, dtype=np.uint8).reshape(1, 10).repeat(2, axis=0)[..., None].repeat(3, axis=2)
    padded = pad_circular(image, 2)
    assert padded[0, :, 0].tolist() == [8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1]
    assert crop_wrapped(image, (-2, 0, 3, 2))[0, :, 0].tolist() == [8, 9, 0, 1, 2]
    assert crop_wrapped(image, (8, 0, 12, 2))[0, :, 0].tolist() == [8, 9, 0, 1]


def test_crop_wrapped_matches_modular_indexing() -> None:
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (6, 10, 3), dtype=np.uint8)
    for region in [(-4, 1, 6, 5), (3, 0, 7, 6), (7, 2, 15, 4), (-12, 0, 3, 6), (0, 0, 25, 2)]:
        x1, y1, x2, y2 = region
        expected = image[y1:y2][:, np.arange(x1, x2) % 10]
        cropped = crop_wrapped(image, region)
        assert cropped.flags["C_CONTIGUOUS"]
        assert np.array_equal(cropped, expected), region


def test_equirect_plan_pads_and_tiles_cross_the_seam() -> None:
    plan = build_plan(
        7680, 3840, projection="equirectangular", profile=DetectProfile.STANDARD,
        tile_trigger_width=5760, equirect_pad_ratio=0.0625,
    )  # fmt: skip
    assert all(p.pad == 480 for p in plan if p.kind == "global")
    tiles = {p.id: p.region for p in plan if p.kind == "tile"}
    assert tiles["tL"] == (-480, 960, 4320, 2880)
    assert tiles["tR"] == (3360, 960, 8160, 2880)
    flat = build_plan(1920, 1080, projection="flat", profile=DetectProfile.STANDARD, tile_trigger_width=5760)
    assert all(p.pad == 0 for p in flat)


def test_merge_across_the_seam() -> None:
    raw = [
        Detection("face", 0.8, (-20, 100, 30, 150), ["g1024"]),  # seen in the left padding
        Detection("face", 0.6, (980, 100, 1030, 150), ["g2048"]),  # same face, right side
    ]
    merged = merge_detections(raw, DEFAULT_CLASS_POLICY, wrap_width=W)
    assert len(merged) == 1
    assert merged[0].box == (980, 100, 1030, 150)
