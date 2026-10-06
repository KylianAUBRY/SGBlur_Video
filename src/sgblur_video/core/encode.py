"""Encoder selection: keep the source codec, prefer hardware encoders.

The output keeps the source codec family (H.264 → H.264, HEVC → HEVC; other
codecs → H.264), resolution, bit depth, colour properties and timestamps. The
first encoder of the family that can actually be opened is used:
VideoToolbox (macOS), then NVENC (NVIDIA), then ``libx264``/``libx265``.
"""

import logging
from dataclasses import dataclass, field
from typing import cast

import av
from av.video.codeccontext import VideoCodecContext

from sgblur_video.config import Settings
from sgblur_video.core.probe import VideoInfo
from sgblur_video.privacy.blur import PLANAR_FORMATS, fallback_format

logger = logging.getLogger(__name__)

_FAMILY_ENCODERS = {
    "hevc": ("hevc_videotoolbox", "hevc_nvenc", "libx265"),
    "h264": ("h264_videotoolbox", "h264_nvenc", "libx264"),
}
_FAMILY_TAGS = {"hevc": "hvc1", "h264": "avc1"}
_JPEG_RANGE = 2


@dataclass(frozen=True)
class EncoderChoice:
    """How the output video stream is encoded.

    Attributes:
        codec: FFmpeg encoder name.
        family: ``hevc`` or ``h264``.
        pix_fmt: Pixel format given to the encoder.
        bit_rate: Target bit rate, if any.
        options: Encoder options.
        codec_tag: Container tag of the output stream.
        color_range: Colour range to signal (1 = limited, 2 = full/JPEG).
    """

    codec: str
    family: str
    pix_fmt: str
    bit_rate: int | None
    options: dict[str, str] = field(default_factory=dict)
    codec_tag: str = ""
    color_range: int = 1


def _bit_depth(pix_fmt: str) -> int:
    if pix_fmt in PLANAR_FORMATS:
        return PLANAR_FORMATS[pix_fmt][0]
    return 10 if fallback_format(pix_fmt) == "yuv420p10le" else 8


def _encoder_pix_fmt(codec: str, source_fmt: str) -> str:
    """Pick the encoder input format closest to the (blurred) frames."""
    depth = _bit_depth(source_fmt)
    if codec.endswith("_videotoolbox"):
        return "p010le" if depth > 8 else "nv12"
    supported = {f.name for f in (av.Codec(codec, "w").video_formats or ())}
    planar = source_fmt if source_fmt in PLANAR_FORMATS else fallback_format(source_fmt)
    # yuvj* formats are the same data with a full-range flag: encode as yuv* and signal the range.
    candidate = planar.replace("yuvj", "yuv")
    if candidate in supported:
        return candidate
    for fallback in ("p010le", "yuv420p10le") if depth > 8 else ("yuv420p", "nv12"):
        if fallback in supported:
            return fallback
    return "yuv420p"


def _options(codec: str, bit_rate: int | None, depth: int) -> dict[str, str]:
    if codec.endswith("_videotoolbox"):
        options = {"allow_sw": "0", "realtime": "0"}
        if not bit_rate:
            options["q:v"] = "65"
        if codec.startswith("hevc") and depth > 8:
            options["profile"] = "main10"
        return options
    if codec.endswith("_nvenc"):
        return {"preset": "p5", "rc": "vbr"} if bit_rate else {"preset": "p5", "rc": "vbr", "cq": "23"}
    options = {"preset": "medium", "crf": "20"}
    if codec == "libx265":
        params = ["log-level=error"]
        if bit_rate:
            params += [f"vbv-maxrate={bit_rate * 3 // 2000}", f"vbv-bufsize={bit_rate * 2 // 1000}"]
        options["x265-params"] = ":".join(params)
    elif bit_rate:
        options |= {"maxrate": str(bit_rate * 3 // 2), "bufsize": str(bit_rate * 2)}
    return options


def _can_open(codec: str, info: VideoInfo, pix_fmt: str) -> bool:
    """Whether the encoder exists and accepts this frame size and format here."""
    if codec not in av.codecs_available:
        return False
    try:
        context = cast(VideoCodecContext, av.CodecContext.create(codec, "w"))
        context.width, context.height, context.pix_fmt = info.width, info.height, pix_fmt
        context.time_base = info.time_base
        context.framerate = info.avg_frame_rate
        context.open()
        del context  # released by garbage collection; PyAV has no explicit close
    except av.FFmpegError, ValueError, OSError:
        return False
    return True


def choose_encoder(
    info: VideoInfo, settings: Settings, *, source_pix_fmt: str | None = None
) -> EncoderChoice:
    """Choose the encoder and its settings for a video.

    Args:
        info: Probed input.
        settings: ``ENCODER`` and ``ENCODE_BITRATE_FACTOR``.
        source_pix_fmt: Format of the frames given to the encoder (defaults to the decoder format).

    Returns:
        The encoder choice.

    Raises:
        RuntimeError: If no candidate encoder can be opened.
    """
    family = info.codec if info.codec in _FAMILY_ENCODERS else "h264"
    candidates = _FAMILY_ENCODERS[family] if settings.encoder == "auto" else (settings.encoder,)
    if settings.encoder != "auto":
        family = "hevc" if settings.encoder.startswith(("hevc", "libx265")) else "h264"
    source_fmt = source_pix_fmt or info.pix_fmt
    bit_rate = int(info.bit_rate * settings.encode_bitrate_factor) if info.bit_rate else None
    color_range = _JPEG_RANGE if source_fmt.startswith("yuvj") else info.color.get("color_range", 1) or 1
    for codec in candidates:
        pix_fmt = _encoder_pix_fmt(codec, source_fmt) if codec in av.codecs_available else "yuv420p"
        if _can_open(codec, info, pix_fmt):
            choice = EncoderChoice(
                codec=codec,
                family=family,
                pix_fmt=pix_fmt,
                bit_rate=bit_rate,
                options=_options(codec, bit_rate, _bit_depth(source_fmt)),
                codec_tag=_FAMILY_TAGS[family],
                color_range=color_range,
            )
            logger.info(
                "encoder: %s (%s, %s bit/s)", choice.codec, choice.pix_fmt, choice.bit_rate or "quality"
            )
            return choice
    msg = f"no usable encoder among {list(candidates)}"
    raise RuntimeError(msg)
