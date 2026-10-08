"""Tests of the irreversible blur on planes and frames."""

import av
import numpy as np
import pytest

from sgblur_video.config import BlurMethod
from sgblur_video.privacy.blur import blur_frame, blur_plane, fallback_format


def _checkerboard(height: int, width: int, dtype: type = np.uint8, high: int = 255) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    return (((yy // 4 + xx // 4) % 2) * high).astype(dtype)


def _energy(patch: np.ndarray) -> float:
    return float(np.abs(np.diff(patch.astype(np.float64), axis=1)).mean())


@pytest.mark.parametrize("method", list(BlurMethod))
def test_blur_destroys_texture_inside_and_keeps_outside(method: BlurMethod) -> None:
    plane = _checkerboard(120, 160)
    original = plane.copy()
    blur_plane(plane, [(40, 30, 120, 90)], method, cells=6, neutral=128)
    assert _energy(plane[30:90, 40:120]) < 0.15 * _energy(original[30:90, 40:120])
    assert np.array_equal(plane[:, :40], original[:, :40])
    assert np.array_equal(plane[95:, :], original[95:, :])


def test_pixelate_keeps_at_most_cells_degrees_of_freedom() -> None:
    rng = np.random.default_rng(0)
    plane = rng.integers(0, 256, (90, 120), dtype=np.uint8)
    blur_plane(plane, [(0, 0, 120, 90)], BlurMethod.PIXELATE_BLUR, cells=6, neutral=128)
    # Smooth result: neighbouring pixels differ by little (no recoverable detail).
    assert np.abs(np.diff(plane.astype(int), axis=1)).max() < 40


def test_ten_bit_planes() -> None:
    plane = _checkerboard(64, 64, np.uint16, high=1023)
    blur_plane(plane, [(8, 8, 56, 56)], BlurMethod.PIXELATE_BLUR, cells=6, neutral=512)
    assert plane.dtype == np.uint16
    assert plane.max() <= 1023
    assert _energy(plane[8:56, 8:56]) < 100


def test_shapes_outside_the_plane_are_ignored() -> None:
    plane = _checkerboard(32, 32)
    original = plane.copy()
    blur_plane(plane, [(100, 100, 120, 120)], BlurMethod.SOLID, cells=6, neutral=0)
    assert np.array_equal(plane, original)


@pytest.mark.parametrize("pix_fmt", ["yuv420p", "yuvj420p", "yuv420p10le", "nv12"])
def test_blur_frame_returns_a_blurred_copy(pix_fmt: str) -> None:
    source = av.VideoFrame.from_ndarray(np.dstack([_checkerboard(64, 96)] * 3), format="rgb24").reformat(
        format=pix_fmt
    )
    source.pts = 42
    before = source.to_ndarray().copy()
    result = blur_frame(source, [(16, 16, 80, 48)], BlurMethod.SOLID, cells=6)
    assert np.array_equal(source.to_ndarray(), before)  # decoded frame untouched
    assert result.pts == 42
    assert result.format.name == (pix_fmt if pix_fmt != "nv12" else "yuv420p")
    luma = np.asarray(result.planes[0]).reshape(result.planes[0].height, -1)
    assert luma.size > 0
    blurred = result.reformat(format="gray").to_ndarray()
    assert _energy(blurred[18:46, 18:78]) < 1.0


def test_fallback_format() -> None:
    assert fallback_format("nv12") == "yuv420p"
    assert fallback_format("p010le") == "yuv420p10le"
    assert fallback_format("rgb24") == "yuv420p"
