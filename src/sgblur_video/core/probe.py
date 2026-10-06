"""Inspect a video before processing: streams, timing, projection, rotation, copyability.

:func:`probe` opens the file with PyAV, decodes a single frame (to read the
display rotation) and returns a :class:`VideoInfo`. It also decides which
non-video streams can be copied unchanged into an MP4 output: FFmpeg refuses
streams whose codec is unknown (GoPro timecode ``tmcd``, CAMM, DJI…), so each
stream is tried in a throw-away in-memory MP4 header.

Unsupported inputs raise :class:`UnsupportedVideoError` with a machine code
matching the HTTP API errors (``unsupported_media_type``,
``unsupported_projection``, ``video_too_long``).
"""

import io
import logging
import struct
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Literal

import av
import av.logging

from sgblur_video.config import Projection, Settings
from sgblur_video.core.decode import to_fraction

logger = logging.getLogger(__name__)

ProjectionName = Literal["flat", "equirectangular"]

# AVSphericalProjection values (libavutil/spherical.h).
_AV_SPHERICAL_EQUIRECTANGULAR = 0
_AV_SPHERICAL_EQUIRECTANGULAR_TILE = 2


class UnsupportedVideoError(ValueError):
    """The input cannot be processed.

    Attributes:
        code: Machine-readable reason, identical to the HTTP API error codes.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class StreamInfo:
    """A non-video stream of the input.

    Attributes:
        index: Stream index in the container.
        kind: ``audio``, ``data``, ``subtitle``…
        handler_name: Handler name from the container (e.g. ``GoPro MET``), may be empty.
        copyable: Whether FFmpeg can copy it unchanged into an MP4 file.
    """

    index: int
    kind: str
    handler_name: str
    copyable: bool


@dataclass(frozen=True)
class VideoInfo:
    """Everything the pipeline needs to know about the input video.

    Attributes:
        path: Input file.
        container: Container format name reported by FFmpeg.
        width: Coded frame width in pixels.
        height: Coded frame height in pixels.
        rotation: Display rotation in degrees (counter-clockwise), from the display matrix.
        time_base: Time base of the video stream.
        start_pts: First presentation timestamp of the video stream.
        avg_frame_rate: Average frame rate.
        frame_count: Number of frames from the container index, if known.
        duration_s: Duration in seconds.
        codec: Video codec name (``h264``, ``hevc``…).
        codec_tag: Codec tag in the container (``avc1``, ``hvc1``…).
        pix_fmt: Pixel format of decoded frames.
        bit_rate: Video bit rate in bit/s, if known.
        color: Colour properties of the video codec context.
        projection: ``flat`` or ``equirectangular``.
        projection_source: Where the projection comes from (metadata, heuristic, setting).
        streams: Non-video streams.
        metadata: Container-level metadata (``creation_time``…).
        video_metadata: Video stream metadata (``timecode``, ``handler_name``…).
    """

    path: Path
    container: str
    width: int
    height: int
    rotation: int
    time_base: Fraction
    start_pts: int
    avg_frame_rate: Fraction
    frame_count: int | None
    duration_s: float
    codec: str
    codec_tag: str
    pix_fmt: str
    bit_rate: int | None
    color: dict[str, int]
    projection: ProjectionName
    projection_source: str
    streams: tuple[StreamInfo, ...] = field(default_factory=tuple)
    metadata: dict[str, str] = field(default_factory=dict)
    video_metadata: dict[str, str] = field(default_factory=dict)

    @property
    def fps(self) -> float:
        """Average frame rate as a float (30.0 when unknown)."""
        return float(self.avg_frame_rate) if self.avg_frame_rate else 30.0


def _spherical_projection(stream: av.video.stream.VideoStream) -> int | None:
    """Return the AVSphericalProjection of the stream, or None without spherical metadata."""
    try:
        side_data = stream.codec_context.coded_side_data
    except AttributeError:  # pragma: no cover - PyAV < 19
        return None
    raw = side_data.get("spherical") if hasattr(side_data, "get") else None
    if not raw or len(raw) < 4:
        return None
    return int(struct.unpack_from("<i", bytes(raw))[0])


def _is_copyable(stream: av.stream.Stream) -> bool:
    """Try to write an MP4 header holding only this stream (FFmpeg rejects unknown codecs)."""
    previous = av.logging.get_level()
    av.logging.set_level(av.logging.PANIC)
    try:
        with av.open(io.BytesIO(), "w", format="mp4") as probe_container:
            probe_container.add_stream_from_template(stream)
            probe_container.start_encoding()
    except av.FFmpegError, ValueError, TypeError:
        return False
    finally:
        av.logging.set_level(previous)
    return True


def _check_raw_360(path: Path, container: av.container.InputContainer | None) -> None:
    """Reject raw 360° camera files that are not equirectangular (by extension, then by stream layout)."""
    if path.suffix.lower() == ".insv":
        raise UnsupportedVideoError(
            "unsupported_projection",
            "Insta360 .insv files (dual fisheye) are not supported; export an equirectangular MP4.",
        )
    if path.suffix.lower() == ".360" or (container is not None and len(container.streams.video) > 1):
        raise UnsupportedVideoError(
            "unsupported_projection",
            "Files with several video streams (e.g. GoPro .360, EAC layout) are not supported; "
            "export an equirectangular MP4.",
        )


def _projection(
    stream: av.video.stream.VideoStream, width: int, height: int, setting: Projection
) -> tuple[ProjectionName, str]:
    """Decide the projection from settings, spherical metadata, then the 2:1 heuristic."""
    if setting is Projection.FLAT:
        return "flat", "setting"
    if setting is Projection.EQUIRECTANGULAR:
        return "equirectangular", "setting"
    spherical = _spherical_projection(stream)
    if spherical is not None:
        if spherical in (_AV_SPHERICAL_EQUIRECTANGULAR, _AV_SPHERICAL_EQUIRECTANGULAR_TILE):
            return "equirectangular", "spherical-metadata"
        raise UnsupportedVideoError(
            "unsupported_projection",
            "Only equirectangular 360° videos are supported (cubemap/mesh projections are not).",
        )
    if width == 2 * height:
        return "equirectangular", "aspect-ratio-heuristic"
    return "flat", "default"


def probe(path: Path, settings: Settings) -> VideoInfo:
    """Inspect a video file.

    Args:
        path: Input video.
        settings: Settings (projection override, duration limit, accepted containers).

    Returns:
        The video description.

    Raises:
        UnsupportedVideoError: If the file cannot or must not be processed.
    """
    _check_raw_360(path, None)
    if path.suffix.lower().lstrip(".") not in settings.accepted_containers:
        raise UnsupportedVideoError(
            "unsupported_media_type",
            f"Accepted containers: {', '.join(settings.accepted_containers)}.",
        )
    try:
        container = av.open(str(path))
    except (av.FFmpegError, OSError) as exc:
        raise UnsupportedVideoError("unsupported_media_type", "The file is not a readable video.") from exc
    with container:
        if not container.streams.video:
            raise UnsupportedVideoError("unsupported_media_type", "The file has no video stream.")
        _check_raw_360(path, container)
        stream = container.streams.video[0]
        codec_context = stream.codec_context
        duration_s = (
            float(stream.duration * stream.time_base)
            if stream.duration
            else (container.duration or 0) / av.time_base
        )
        if duration_s > settings.max_video_duration_s:
            raise UnsupportedVideoError(
                "video_too_long", f"Video longer than {settings.max_video_duration_s} s ({duration_s:.0f} s)."
            )
        try:
            first = next(container.decode(stream))
        except (StopIteration, av.FFmpegError) as exc:
            raise UnsupportedVideoError(
                "unsupported_media_type", "The video stream cannot be decoded."
            ) from exc
        width, height = first.width, first.height
        projection, source = _projection(stream, width, height, settings.projection)
        streams = tuple(
            StreamInfo(
                index=s.index,
                kind=s.type,
                handler_name=s.metadata.get("handler_name", "").strip(),
                copyable=_is_copyable(s),
            )
            for s in container.streams
            if s.type != "video"
        )
        color = {
            name: int(getattr(codec_context, name))
            for name in ("color_range", "colorspace", "color_primaries", "color_trc")
            if getattr(codec_context, name, None) is not None
        }
        info = VideoInfo(
            path=path,
            container=container.format.name,
            width=width,
            height=height,
            rotation=round(first.rotation or 0),
            time_base=to_fraction(stream.time_base) if stream.time_base else Fraction(1, 90000),
            start_pts=int(stream.start_time or 0),
            avg_frame_rate=to_fraction(stream.average_rate) if stream.average_rate else Fraction(30),
            frame_count=int(stream.frames) or None,
            duration_s=duration_s,
            codec=codec_context.name,
            codec_tag=str(codec_context.codec_tag or ""),
            pix_fmt=first.format.name,
            bit_rate=int(codec_context.bit_rate or stream.bit_rate or 0) or None,
            color=color,
            projection=projection,
            projection_source=source,
            streams=streams,
            metadata=dict(container.metadata),
            video_metadata=dict(stream.metadata),
        )
    logger.info(
        "probed video: %dx%d %s %s, %.1f s, projection=%s (%s), %d other streams",
        info.width,
        info.height,
        info.codec,
        info.pix_fmt,
        info.duration_s,
        info.projection,
        info.projection_source,
        len(info.streams),
    )
    return info
