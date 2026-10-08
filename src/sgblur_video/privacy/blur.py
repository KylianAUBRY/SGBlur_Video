"""Irreversible blur applied directly on YUV planes.

Working on the decoder's planar YUV data (8 or 10 bit) avoids an RGB round
trip: no colour shift, half the memory traffic on 8K 10-bit video. Each plane
is processed at its own resolution (chroma planes are subsampled), with the
shape mask computed analytically for that plane.

Methods (``docs/adr/0004-irreversible-blur.md``):

* ``pixelate_blur`` — averages down to at most ``cells`` cells on the long
  side, smoothed and interpolated back (no visible blocks);
* ``gaussian_strong`` — Gaussian blur of σ = long side / 4 plus low-amplitude noise;
* ``solid`` — constant neutral grey.

Example:
    >>> import numpy as np
    >>> y = np.zeros((64, 64), np.uint8); y[::2] = 255  # stripes
    >>> shape = PlaneShape("rect", (16, 16, 48, 48))
    >>> blur_plane(y, [shape], BlurMethod.PIXELATE_BLUR, cells=6, neutral=128)
    >>> int(y[32, 16:48].std()) < 20
    True
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import av
import av.video.plane
import cv2
import numpy as np
import numpy.typing as npt

from sgblur_video.config import BlurMethod
from sgblur_video.core.geometry import Box, inscribed_ellipse

#: 2-D uint8 (8-bit) or uint16 (10-bit) plane.
PlaneArray = npt.NDArray[Any]


@dataclass(frozen=True)
class PlaneShape:
    """A shape expressed in the pixel coordinates of one plane.

    Attributes:
        kind: ``ellipse`` (inscribed in ``box``) or ``rect``.
        box: Box in plane pixels (may extend past the plane edges).
    """

    kind: Literal["ellipse", "rect"]
    box: Box


def _roi(shape: PlaneShape, plane_width: int, plane_height: int) -> tuple[int, int, int, int] | None:
    """Integer region covered by the shape, clipped to the plane; None if empty."""
    if shape.kind == "ellipse":
        cx, cy, rx, ry = inscribed_ellipse(shape.box)
        x1, y1, x2, y2 = cx - rx, cy - ry, cx + rx, cy + ry
    else:
        x1, y1, x2, y2 = shape.box
    left, top = max(0, math.floor(x1)), max(0, math.floor(y1))
    right, bottom = min(plane_width, math.ceil(x2)), min(plane_height, math.ceil(y2))
    if right - left < 1 or bottom - top < 1:
        return None
    return left, top, right, bottom


def _mask(shape: PlaneShape, roi: tuple[int, int, int, int]) -> npt.NDArray[np.bool_] | None:
    """Boolean mask of the shape inside the ROI (None means the whole ROI)."""
    if shape.kind == "rect":
        return None
    cx, cy, rx, ry = inscribed_ellipse(shape.box)
    left, top, right, bottom = roi
    # Pixel centres; a pixel is blurred when its centre is inside the ellipse,
    # and the ellipse is grown by half a pixel so that edge pixels are included.
    xs = (np.arange(left, right) + 0.5 - cx) / max(rx + 0.5, 0.5)
    ys = (np.arange(top, bottom) + 0.5 - cy) / max(ry + 0.5, 0.5)
    result: npt.NDArray[np.bool_] = (xs[np.newaxis, :] ** 2 + ys[:, np.newaxis] ** 2) <= 1.0
    return result


def _pixelate_blur(patch: PlaneArray, cells: int) -> PlaneArray:
    """Average the patch down to at most ``cells`` cells on its long side, then interpolate smoothly.

    Only ``cells × cells`` averages survive, which is what makes the result
    irreversible; smoothing them at low resolution and upscaling with bilinear
    interpolation removes the block edges of a plain mosaic (which help
    super-resolution attacks) at a fraction of the cost of a full-size blur.
    """
    patch_height, patch_width = patch.shape
    cell = max(2, math.ceil(max(patch_width, patch_height) / cells))
    small_size = (max(1, math.ceil(patch_width / cell)), max(1, math.ceil(patch_height / cell)))
    small = cv2.resize(patch, small_size, interpolation=cv2.INTER_AREA)
    if min(small_size) > 1:
        small = cv2.GaussianBlur(small, (3, 3), sigmaX=0.5, borderType=cv2.BORDER_REPLICATE)
    result: PlaneArray = cv2.resize(small, (patch_width, patch_height), interpolation=cv2.INTER_LINEAR)
    return result


def _gaussian_strong(patch: PlaneArray, rng: np.random.Generator) -> PlaneArray:
    """Strong Gaussian blur (σ = long side / 4) computed at reduced resolution, plus noise."""
    patch_height, patch_width = patch.shape
    sigma = max(patch_width, patch_height) / 4
    factor = max(1.0, sigma / 4)
    small_size = (max(1, round(patch_width / factor)), max(1, round(patch_height / factor)))
    small = cv2.resize(patch, small_size, interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigmaX=sigma / factor, borderType=cv2.BORDER_REPLICATE)
    blurred = cv2.resize(small, (patch_width, patch_height), interpolation=cv2.INTER_LINEAR)
    maximum = float(np.iinfo(patch.dtype).max)
    noise = rng.normal(0.0, maximum * 0.01, size=patch.shape)
    result: PlaneArray = np.clip(blurred.astype(np.float32) + noise, 0, maximum).astype(patch.dtype)
    return result


def blur_plane(
    plane: PlaneArray,
    shapes: Sequence[PlaneShape],
    method: BlurMethod,
    *,
    cells: int,
    neutral: int,
    rng: np.random.Generator | None = None,
) -> None:
    """Blur shapes of one plane **in place**.

    Args:
        plane: 2-D plane (uint8 or uint16), writable, not owned by a decoder.
        shapes: Shapes in this plane's pixel coordinates.
        method: Blur operation.
        cells: Maximum mosaic cells on the long side (``pixelate_blur``).
        neutral: Value used by ``solid`` (neutral grey for this plane).
        rng: Random generator for ``gaussian_strong`` noise.
    """
    plane_height, plane_width = plane.shape
    generator = rng if rng is not None else np.random.default_rng()
    for shape in shapes:
        roi = _roi(shape, plane_width, plane_height)
        if roi is None:
            continue
        left, top, right, bottom = roi
        patch = plane[top:bottom, left:right]
        if method is BlurMethod.SOLID:
            blurred = np.full_like(patch, neutral)
        elif method is BlurMethod.GAUSSIAN_STRONG:
            blurred = _gaussian_strong(np.ascontiguousarray(patch), generator)
        else:
            blurred = _pixelate_blur(np.ascontiguousarray(patch), cells)
        mask = _mask(shape, roi)
        if mask is None:
            patch[...] = blurred
        else:
            patch[mask] = blurred[mask]


def neutral_values(bit_depth: int) -> tuple[int, int]:
    """Neutral grey for luma and chroma planes at a bit depth (mid-range)."""
    mid = 1 << (bit_depth - 1)
    return mid, mid


#: Planar YUV formats blurred natively: name -> (bit depth, log2 chroma width, log2 chroma height).
PLANAR_FORMATS: dict[str, tuple[int, int, int]] = {
    "yuv420p": (8, 1, 1),
    "yuvj420p": (8, 1, 1),
    "yuv422p": (8, 1, 0),
    "yuvj422p": (8, 1, 0),
    "yuv444p": (8, 0, 0),
    "yuvj444p": (8, 0, 0),
    "yuv420p10le": (10, 1, 1),
    "yuv422p10le": (10, 1, 0),
    "yuv444p10le": (10, 0, 0),
}


def fallback_format(pix_fmt: str) -> str:
    """Planar format used when the decoder outputs a format we do not blur natively.

    Args:
        pix_fmt: Decoder output format (e.g. ``nv12``, ``p010le``, ``rgb24``).

    Returns:
        ``yuv420p10le`` for formats with more than 8 bits per component, ``yuv420p`` otherwise.
    """
    bits = max(component.bits for component in av.VideoFormat(pix_fmt).components)
    return "yuv420p10le" if bits > 8 else "yuv420p"


def _plane_view(plane: av.video.plane.VideoPlane, dtype: type[np.uint8] | type[np.uint16]) -> PlaneArray:
    """NumPy view of a plane without its line padding."""
    itemsize = np.dtype(dtype).itemsize
    array: PlaneArray = np.frombuffer(plane, dtype).reshape(plane.height, plane.line_size // itemsize)
    return array[:, : plane.width]


def blur_frame(
    frame: av.VideoFrame,
    shapes: Sequence[tuple[Literal["ellipse", "rect"], Box]],
    method: BlurMethod,
    *,
    cells: int,
    rng: np.random.Generator | None = None,
) -> av.VideoFrame:
    """Return a blurred **copy** of a decoded frame.

    The decoded frame is never modified (decoders may reuse its buffers as
    reference frames). Frames in a format without native support are first
    converted to a planar YUV format (see :func:`fallback_format`).

    Args:
        frame: Decoded frame.
        shapes: ``(kind, box)`` pairs in coded-frame pixels.
        method: Blur operation.
        cells: Maximum mosaic cells on the long side.
        rng: Random generator for ``gaussian_strong``.

    Returns:
        A new frame with the same timestamps and colour properties.
    """
    name = frame.format.name
    source = frame if name in PLANAR_FORMATS else frame.reformat(format=fallback_format(name))
    bit_depth, log2_w, log2_h = PLANAR_FORMATS[source.format.name]
    dtype = np.uint16 if bit_depth > 8 else np.uint8
    luma_neutral, chroma_neutral = neutral_values(bit_depth)
    result = av.VideoFrame(source.width, source.height, source.format.name)
    for index, (src_plane, dst_plane) in enumerate(zip(source.planes, result.planes, strict=True)):
        destination = _plane_view(dst_plane, dtype)
        destination[...] = _plane_view(src_plane, dtype)
        sx, sy = (1.0, 1.0) if index == 0 else (1.0 / (1 << log2_w), 1.0 / (1 << log2_h))
        plane_shapes = [
            PlaneShape(kind, (box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy)) for kind, box in shapes
        ]
        blur_plane(
            destination,
            plane_shapes,
            method,
            cells=cells,
            neutral=luma_neutral if index == 0 else chroma_neutral,
            rng=rng,
        )
    result.pts = frame.pts
    if frame.time_base is not None:
        result.time_base = frame.time_base
    for attribute in ("color_range", "colorspace", "color_primaries", "color_trc"):
        value = getattr(frame, attribute, None)
        if value is not None:
            setattr(result, attribute, value)
    return result
