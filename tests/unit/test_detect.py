"""Tests of the detection plan, cross-pass merge and rotation helpers."""

import numpy as np
import pytest

from sgblur_video.config import DEFAULT_CLASS_POLICY, ClassAction, DetectProfile
from sgblur_video.core.decode import rotate_upright
from sgblur_video.core.detect import (
    Detection,
    YoloDetector,
    build_plan,
    class_groups,
    merge_detections,
    rotate_box,
    unrotate_box,
)


def _ids(
    width: int, height: int, profile: DetectProfile = DetectProfile.STANDARD, projection: str = "flat"
) -> list[str]:
    plan = build_plan(width, height, projection=projection, profile=profile, tile_trigger_width=5760)  # type: ignore[arg-type]
    return [p.id for p in plan]


def test_plan_small_and_hd() -> None:
    assert _ids(640, 360) == ["g1024"]
    plan = build_plan(640, 360, projection="flat", profile=DetectProfile.STANDARD, tile_trigger_width=5760)
    assert plan[0].imgsz == 640  # never upscale beyond the frame
    assert _ids(1920, 1080) == ["g1024", "g1920"]
    assert _ids(3840, 1920, projection="equirectangular") == ["g1024", "g2048"]


def test_plan_8k_equirect_uses_middle_band_halves() -> None:
    plan = build_plan(
        7680, 3840, projection="equirectangular", profile=DetectProfile.STANDARD, tile_trigger_width=5760
    )
    tiles = {p.id: p for p in plan if p.kind == "tile"}
    assert set(tiles) == {"tL", "tR"}
    assert tiles["tL"].region == (0, 960, 4320, 2880)
    assert tiles["tR"].region == (3360, 960, 7680, 2880)
    assert all(p.imgsz <= 4096 for p in plan)
    thorough = build_plan(
        7680, 3840, projection="equirectangular", profile=DetectProfile.THOROUGH, tile_trigger_width=5760
    )
    assert next(p for p in thorough if p.id == "tL").region == (0, 0, 4320, 3840)
    assert _ids(7680, 3840, DetectProfile.FAST, "equirectangular") == ["g1024", "g2048"]


def test_plan_8k_flat_grid_covers_frame() -> None:
    plan = build_plan(7680, 4320, projection="flat", profile=DetectProfile.STANDARD, tile_trigger_width=5760)
    tiles = [p for p in plan if p.kind == "tile"]
    assert len(tiles) == 4
    covered = np.zeros((4320, 7680), bool)
    for tile in tiles:
        assert tile.region is not None
        x1, y1, x2, y2 = tile.region
        covered[y1:y2, x1:x2] = True
    assert covered.all()


def test_class_groups() -> None:
    assert class_groups(DEFAULT_CLASS_POLICY) == {
        "face": "face",
        "plate": "plate",
        "sign": "signage",
        "direction": "signage",
    }


def test_merge_unions_blur_classes_and_keeps_best_sign() -> None:
    raw = [
        Detection("face", 0.4, (10, 10, 30, 30), ["g1024"]),
        Detection("face", 0.6, (12, 12, 34, 34), ["g2048"]),
        Detection("sign", 0.9, (100, 100, 140, 140), ["g2048"]),
        Detection("direction", 0.5, (98, 98, 150, 150), ["tL"]),
        Detection("plate", 0.3, (12, 12, 30, 30), ["g1024"]),  # overlaps the face but another group
        Detection("bicycle", 0.9, (0, 0, 5, 5), ["g1024"]),  # no policy: dropped
    ]
    merged = {d.cls: d for d in merge_detections(raw, DEFAULT_CLASS_POLICY)}
    assert set(merged) == {"face", "sign", "plate"}
    assert merged["face"].box == (10, 10, 34, 34)
    assert merged["face"].score == 0.6
    assert sorted(merged["face"].passes) == ["g1024", "g2048"]
    assert merged["sign"].box == (100, 100, 140, 140)  # best member, not the union


def test_merge_containment_counts_as_duplicate() -> None:
    raw = [
        Detection("plate", 0.5, (0, 0, 100, 40), ["g1024"]),
        Detection("plate", 0.4, (10, 10, 40, 30), ["tL"]),
    ]
    assert len(merge_detections(raw, {"plate": ClassAction.BLUR})) == 1


@pytest.mark.parametrize("rotation", [0, 90, 180, 270, -90])
def test_rotation_round_trip_matches_numpy(rotation: int) -> None:
    coded = np.zeros((60, 100), np.uint8)
    coded[10:20, 30:50] = 255  # box (30, 10, 50, 20) in coded pixels
    upright = rotate_upright(coded, rotation)
    ys, xs = np.nonzero(upright)
    upright_box = (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
    assert rotate_box((30, 10, 50, 20), rotation, 100, 60) == upright_box
    assert unrotate_box(upright_box, rotation, 100, 60) == (30, 10, 50, 20)


@pytest.mark.parametrize(("device", "tile_batches"), [("cpu", [1, 1]), ("mps", [2])])
def test_tiles_are_batched_on_accelerators_only(device: str, tile_batches: list[int]) -> None:
    plan = build_plan(
        2048, 1024, projection="equirectangular", profile=DetectProfile.STANDARD, tile_trigger_width=2048
    )
    calls: list[int] = []
    detector = object.__new__(YoloDetector)  # no checkpoint: inference is replaced below
    detector._device, detector._names = device, ("face",)
    detector._predict = lambda images, imgsz: calls.append(len(images)) or [[] for _ in images]  # type: ignore[method-assign]
    detector.infer(detector.prepare(np.zeros((1024, 2048, 3), dtype=np.uint8), plan))
    globals_count = sum(p.kind == "global" for p in plan)
    assert calls == [1] * globals_count + tile_batches
