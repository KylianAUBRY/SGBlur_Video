"""Pass 2: decode again, blur, encode and mux, in a single loop.

The original file is demuxed once: video packets are decoded, blurred and
encoded; audio, subtitle and supported data packets (GoPro GPMF ``gpmd``) are
copied unchanged with their timestamps. Streams FFmpeg cannot write to MP4
(codec "none": GoPro timecode, CAMM, DJI…) are dropped and reported; the GoPro
timecode track is recreated by the muxer from the video stream's ``timecode``
metadata.

Box-level metadata that FFmpeg does not write (Spherical V1/V2, GoPro ``udta``)
is transplanted afterwards by ``mp4boxes``.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import cast

import av
import numpy as np
from av.video.stream import VideoStream

from sgblur_video.config import Settings
from sgblur_video.core.debug import DEBUG_MAX_WIDTH, DebugOverlay
from sgblur_video.core.decode import configure_decoder, to_bgr, to_fraction
from sgblur_video.core.encode import EncoderChoice, choose_encoder
from sgblur_video.core.geometry import Box
from sgblur_video.core.postprocess import BlurPlan, BlurShape
from sgblur_video.core.probe import VideoInfo
from sgblur_video.privacy.blur import blur_frame
from sgblur_video.video360.wrap import copies

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int | None], None]
#: Receives every output frame (after blurring) with its index, e.g. to save best frames.
FrameSink = Callable[[int, av.VideoFrame], None]
#: Receives the original (unblurred) frame and the shapes about to be blurred (``keep=1``).
RegionSink = Callable[[int, av.VideoFrame, list[BlurShape]], None]

# Container-level keys that the muxer writes itself.
_SKIPPED_METADATA = {"major_brand", "minor_version", "compatible_brands", "encoder"}
_KEPT_STREAM_METADATA = ("handler_name", "language", "timecode", "creation_time")


class RenderError(RuntimeError):
    """Rendering produced an inconsistent output (e.g. a frame count mismatch)."""


@dataclass
class RenderStats:
    """What pass 2 did.

    Attributes:
        frames: Frames decoded and encoded.
        blurred_frames: Frames that had at least one region blurred.
        encoder: Encoder used.
        copied_streams: Handler names (or kinds) of streams copied unchanged.
        dropped_streams: Handler names (or kinds) of streams that could not be copied.
        elapsed_s: Wall-clock duration.
    """

    frames: int = 0
    blurred_frames: int = 0
    encoder: str = ""
    copied_streams: list[str] = field(default_factory=list)
    dropped_streams: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0


def _wrapped(box: Box, wrap_width: int | None) -> list[Box]:
    """The box, plus its copies one turn left and right on 360° video (the seam is blurred on both sides)."""
    return copies(box, wrap_width) if wrap_width else [box]


#: After the last frame of a range, copied packets are read this much further (interleaving).
_DRAIN_MARGIN_S = 2


def _packet_seconds(packet: av.Packet[av.stream.Stream]) -> Fraction:
    """Presentation time of a packet, in seconds."""
    timestamp = packet.pts if packet.pts is not None else packet.dts
    return (timestamp or 0) * to_fraction(packet.time_base)


class _RangeCopier:
    """Audio and telemetry packets of a frame range (``render(first_frame=…, allow_short=…)``).

    A packet is kept when its time falls between the first rendered frame and the end of the
    last one. Both are only known once the decoder outputs those frames, several packets after
    the demuxer delivered the matching audio, so packets wait in a queue until they can be
    decided. With ``shift``, kept packets are moved to start at 0 like the video.
    """

    def __init__(
        self,
        target: av.container.OutputContainer,
        copies: dict[int, av.stream.Stream],
        *,
        shift: bool,
        info: VideoInfo,
    ) -> None:
        self._target = target
        self._copies = copies
        self._shift = shift
        rate = to_fraction(info.avg_frame_rate)
        self._frame_duration = 1 / rate if rate > 0 else Fraction(1, 30)
        self._pending: list[av.Packet[av.stream.Stream]] = []
        self._start: Fraction | None = None
        self.end: Fraction | None = None

    def add(self, packet: av.Packet[av.stream.Stream]) -> None:
        """A copied packet from the demuxer."""
        if self.end is not None:
            self._decide(packet)
        else:
            self._pending.append(packet)

    def skip_to(self, frame_s: Fraction) -> None:
        """A frame before the range was skipped: earlier packets are before the range too."""
        self._pending = [p for p in self._pending if _packet_seconds(p) >= frame_s]

    def rendered(self, frame_s: Fraction) -> None:
        """A frame of the range is rendered: packets before it are decided."""
        if self._start is None:
            self._start = frame_s
        ready = [p for p in self._pending if _packet_seconds(p) < frame_s]
        self._pending = [p for p in self._pending if _packet_seconds(p) >= frame_s]
        for packet in ready:
            self._decide(packet)

    def finish(self, last_frame_s: Fraction) -> None:
        """The last frame of the range is rendered: decide every waiting packet."""
        self.end = last_frame_s + self._frame_duration
        waiting, self._pending = self._pending, []
        for packet in waiting:
            self._decide(packet)

    def past_end(self, packet: av.Packet[av.stream.Stream]) -> bool:
        """Whether a packet is far enough after the range to stop reading the source."""
        return self.end is not None and _packet_seconds(packet) >= self.end + _DRAIN_MARGIN_S

    def _decide(self, packet: av.Packet[av.stream.Stream]) -> None:
        seconds = _packet_seconds(packet)
        if packet.time_base is None or (self.end is not None and seconds >= self.end):
            return
        if self._shift:
            if self._start is None or seconds < self._start:
                return
            shift = round(self._start / to_fraction(packet.time_base))
            if packet.dts is not None:
                packet.dts -= shift
            if packet.pts is not None:
                packet.pts -= shift
        packet.stream = self._copies[packet.stream.index]
        self._target.mux(packet)


def _stream_label(stream: av.stream.Stream) -> str:
    return stream.metadata.get("handler_name", "").strip() or stream.type


def _open_video_stream(
    container: av.container.OutputContainer, info: VideoInfo, choice: EncoderChoice
) -> VideoStream:
    stream = cast(
        VideoStream, container.add_stream(choice.codec, rate=info.avg_frame_rate, options=choice.options)
    )
    stream.width, stream.height, stream.pix_fmt = info.width, info.height, choice.pix_fmt
    stream.time_base = info.time_base
    context = stream.codec_context
    context.time_base = info.time_base
    if choice.bit_rate:
        context.bit_rate = choice.bit_rate
    context.color_range = choice.color_range
    for name in ("colorspace", "color_primaries", "color_trc"):
        if name in info.color and info.color[name] not in (0, 2):  # 0 = reserved, 2 = unspecified
            setattr(context, name, info.color[name])
    if choice.codec_tag:
        context.codec_tag = choice.codec_tag
    stream.metadata.update({k: v for k, v in info.video_metadata.items() if k in _KEPT_STREAM_METADATA})
    return stream


def _debug_writer(path: Path, info: VideoInfo) -> tuple[av.container.OutputContainer, VideoStream, float]:
    factor = min(1.0, DEBUG_MAX_WIDTH / info.width)
    width, height = max(2, round(info.width * factor) // 2 * 2), max(2, round(info.height * factor) // 2 * 2)
    container = av.open(str(path), "w")
    codec = "h264_videotoolbox" if "h264_videotoolbox" in av.codecs_available else "libx264"
    stream = cast(VideoStream, container.add_stream(codec, rate=info.avg_frame_rate))
    stream.width, stream.height, stream.pix_fmt = (
        width,
        height,
        "nv12" if codec.endswith("toolbox") else "yuv420p",
    )
    stream.time_base = info.time_base
    stream.codec_context.time_base = info.time_base
    stream.codec_context.bit_rate = 8_000_000
    return container, stream, width / info.width


def render(
    info: VideoInfo,
    plan: BlurPlan,
    output: Path,
    settings: Settings,
    *,
    max_frames: int | None = None,
    first_frame: int = 0,
    allow_short: bool = False,
    debug: tuple[Path, DebugOverlay] | None = None,
    frame_sink: FrameSink | None = None,
    region_sink: RegionSink | None = None,
    progress: ProgressCallback | None = None,
) -> RenderStats:
    """Blur the video according to ``plan`` and write ``output``.

    Args:
        info: Probed input video.
        plan: Blur plan from post-processing.
        output: Output file (MP4 or MOV, by extension).
        settings: Blur method, cells, encoder settings.
        max_frames: Stop after this many frames (must match the analysis).
        first_frame: Skip the source frames before this one (frame range of a job). The output
            then starts at 0: video, audio and telemetry are shifted by the time of that frame, and
            copied packets from before it are dropped. The plan is indexed from that frame.
        allow_short: Accept a source that ends before ``plan.frame_count`` frames (at least one).
        debug: Optional ``(path, overlay)`` to also write an annotated debug video.
        frame_sink: Optional callback receiving each blurred frame (best-frame pictures).
        region_sink: Optional callback receiving original frames with their shapes (``keep=1``).
        progress: Called with ``(frames_done, frames_total)``.

    Returns:
        Statistics of the rendering.

    Raises:
        RenderError: If the number of rendered frames differs from the plan (with ``allow_short``:
            if no frame was rendered).
    """
    started = time.monotonic()
    stats = RenderStats()
    rng = np.random.default_rng()
    total = min(plan.frame_count, max_frames) if max_frames is not None else plan.frame_count
    with av.open(str(info.path)) as source, av.open(str(output), "w") as target:
        in_video = source.streams.video[0]
        configure_decoder(in_video)
        skipped = 0
        offset_pts: int | None = None  # pts of the first rendered frame, with first_frame
        last_s = Fraction(0)
        choice = choose_encoder(info, settings)
        stats.encoder = choice.codec
        out_video = _open_video_stream(target, info, choice)
        copies: dict[int, av.stream.Stream] = {}
        copyable = {s.index for s in info.streams if s.copyable}
        for stream in source.streams:
            if stream.type == "video":
                continue
            if stream.index not in copyable:
                stats.dropped_streams.append(_stream_label(stream))
                continue
            copy = target.add_stream_from_template(stream)
            copy.metadata.update({k: v for k, v in stream.metadata.items() if k in _KEPT_STREAM_METADATA})
            copies[stream.index] = copy
            stats.copied_streams.append(_stream_label(stream))
        target.metadata.update({k: v for k, v in info.metadata.items() if k not in _SKIPPED_METADATA})
        ranged = first_frame > 0 or allow_short
        copier = _RangeCopier(target, copies, shift=first_frame > 0, info=info) if ranged else None

        debug_container, debug_stream, debug_factor = (None, None, 1.0)
        if debug is not None:
            debug_container, debug_stream, debug_factor = _debug_writer(debug[0], info)

        def encode(frame: av.VideoFrame) -> None:
            shapes = plan.shapes(stats.frames)
            if shapes and region_sink is not None:
                region_sink(stats.frames, frame, shapes)
            if shapes:
                frame = blur_frame(
                    frame,
                    [box for s in shapes for box in _wrapped(s.box, plan.wrap_width)],
                    settings.blur_method,
                    cells=settings.pixelate_cells,
                    rng=rng,
                )
                stats.blurred_frames += 1
            if frame_sink is not None:
                frame_sink(stats.frames, frame)
            encoded = frame.reformat(
                format=choice.pix_fmt, src_color_range=choice.color_range, dst_color_range=choice.color_range
            )
            encoded.pts, encoded.time_base = frame.pts, frame.time_base
            target.mux(out_video.encode(encoded))
            if debug is not None and debug_container is not None and debug_stream is not None:
                image = to_bgr(frame, debug_stream.width, debug_stream.height)
                debug[1].draw(stats.frames, image, debug_factor)
                debug_frame = av.VideoFrame.from_ndarray(image, format="bgr24").reformat(
                    format=debug_stream.pix_fmt
                )
                debug_frame.pts, debug_frame.time_base = frame.pts, frame.time_base
                debug_container.mux(debug_stream.encode(debug_frame))
            stats.frames += 1
            if progress is not None:
                progress(stats.frames, total)

        try:
            for packet in source.demux():
                is_copied = packet.stream.index in copies and packet.dts is not None
                if stats.frames >= total:
                    # A range keeps audio and telemetry up to the end of its last frame, which the
                    # demuxer may deliver later (packets are interleaved by time).
                    if copier is None or copier.end is None:
                        break
                    if is_copied:
                        if copier.past_end(packet):
                            break
                        copier.add(packet)
                    continue
                if packet.stream.index == in_video.index:
                    for frame in packet.decode():
                        if not isinstance(frame, av.VideoFrame):
                            continue
                        frame_s = (frame.pts or 0) * to_fraction(frame.time_base or in_video.time_base)
                        if skipped < first_frame:
                            skipped += 1
                            if copier is not None:
                                copier.skip_to(frame_s)
                            continue
                        if stats.frames >= total:
                            break
                        if first_frame and frame.pts is not None:
                            offset_pts = frame.pts if offset_pts is None else offset_pts
                            frame.pts -= offset_pts
                        if copier is not None:
                            copier.rendered(frame_s)
                        encode(frame)
                        last_s = frame_s
                        if copier is not None and stats.frames >= total:
                            copier.finish(frame_s)
                elif is_copied:
                    if copier is not None:
                        copier.add(packet)
                    else:
                        packet.stream = copies[packet.stream.index]
                        target.mux(packet)
            if copier is not None and copier.end is None and stats.frames:
                copier.finish(last_s)  # the source ended before the end of the range
            target.mux(out_video.encode(None))
            if debug_container is not None and debug_stream is not None:
                debug_container.mux(debug_stream.encode(None))
        finally:
            if debug_container is not None:
                debug_container.close()
    stats.elapsed_s = round(time.monotonic() - started, 2)
    if allow_short and stats.frames == 0:
        msg = f"no frame from frame {first_frame}: the video has {skipped} frames"
        raise RenderError(msg)
    if stats.frames != total and not allow_short:
        msg = f"rendered {stats.frames} frames, expected {total}"
        raise RenderError(msg)
    logger.info(
        "rendered %d frames (%d blurred) with %s in %.1f s; copied %s; dropped %s",
        stats.frames,
        stats.blurred_frames,
        stats.encoder,
        stats.elapsed_s,
        stats.copied_streams,
        stats.dropped_streams,
    )
    return stats
