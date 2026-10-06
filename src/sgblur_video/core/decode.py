"""Sequential frame decoding with exact timestamps.

Timestamps always come from the stream (``pts × time_base``): they are never
recomputed from a nominal frame rate, so variable-frame-rate videos stay
aligned with their audio and GPS tracks.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import numpy.typing as npt


def to_fraction(rational: object) -> Fraction:
    """Convert a PyAV rational (``AVRational``/``Fraction``) into a :class:`fractions.Fraction`."""
    return Fraction(int(getattr(rational, "numerator", 0)), int(getattr(rational, "denominator", 1)) or 1)


@dataclass(frozen=True)
class DecodedFrame:
    """One decoded video frame.

    Attributes:
        index: 0-based index in decoding (= presentation) order.
        pts: Presentation timestamp in ``time_base`` units.
        time: Seconds since the first frame.
        frame: The PyAV frame. **Never modify its buffers**: decoders may keep
            them as reference frames.
    """

    index: int
    pts: int
    time: float
    frame: av.VideoFrame


def iter_frames(path: Path, *, max_frames: int | None = None) -> Iterator[DecodedFrame]:
    """Decode the first video stream of a file, frame by frame.

    Args:
        path: Video file.
        max_frames: Stop after this many frames (development and tests).

    Yields:
        Decoded frames in presentation order.
    """
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        time_base = to_fraction(stream.time_base) if stream.time_base else Fraction(1, 90000)
        first_pts: int | None = None
        last_pts = -1
        for index, frame in enumerate(container.decode(stream)):
            if max_frames is not None and index >= max_frames:
                break
            # Frames without pts are rare (broken streams); keep timestamps strictly increasing.
            pts = frame.pts if frame.pts is not None else last_pts + 1
            if first_pts is None:
                first_pts = pts
            last_pts = pts
            yield DecodedFrame(index=index, pts=pts, time=float((pts - first_pts) * time_base), frame=frame)


def to_bgr(
    frame: av.VideoFrame, width: int | None = None, height: int | None = None
) -> npt.NDArray[np.uint8]:
    """Convert a frame to an 8-bit BGR array, optionally resized (swscale).

    Args:
        frame: Decoded frame (any pixel format, 8 or 10 bit).
        width: Output width; ``None`` keeps the frame width.
        height: Output height; ``None`` keeps the aspect ratio when ``width`` is given.

    Returns:
        ``(height, width, 3)`` uint8 array.
    """
    if width is not None and height is None:
        height = max(2, round(frame.height * width / frame.width))
    return np.asarray(frame.reformat(width=width, height=height, format="bgr24").to_ndarray(), dtype=np.uint8)


def rotate_upright(image: npt.NDArray[np.uint8], rotation: int) -> npt.NDArray[np.uint8]:
    """Rotate an image so that it is displayed upright.

    Args:
        image: Image in coded orientation.
        rotation: Display rotation in degrees counter-clockwise (multiple of 90).

    Returns:
        The rotated image (a view when possible).
    """
    turns = (round(rotation / 90) % 4) if rotation else 0
    return np.ascontiguousarray(np.rot90(image, k=turns)) if turns else image
