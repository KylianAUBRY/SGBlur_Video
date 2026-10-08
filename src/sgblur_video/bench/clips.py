"""``annotate export``: cut a clip, encode an annotation proxy, pre-annotate it with the model.

The clip is re-encoded at full resolution (same codec family, source bit rate)
rather than stream-copied: a stream copy can only start on a key frame, and the
clip and its proxy must hold exactly the same frames. Audio and telemetry are
dropped; projection metadata (360°) is transplanted from the source.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import cast

import av
from av.video.stream import VideoStream

from sgblur_video.bench.cvat import KeyBox, PreTrack
from sgblur_video.config import ClassAction, Settings
from sgblur_video.core.decode import configure_decoder, to_fraction
from sgblur_video.core.detect import class_groups
from sgblur_video.core.detections_io import Detections
from sgblur_video.core.encode import choose_encoder
from sgblur_video.core.geometry import Box, scale, union_box, width
from sgblur_video.core.linking import fragments_from_detections, link_fragments
from sgblur_video.core.mp4boxes import Mp4BoxError, transplant
from sgblur_video.core.probe import VideoInfo
from sgblur_video.video360.wrap import normalize, split

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int | None], None]
#: Default proxy width: 8K is too heavy for annotation tools.
PROXY_MAX_WIDTH = 3840
_PROXY_BIT_RATE = 25_000_000


class ClipError(ValueError):
    """The clip cannot be cut from this video."""


@dataclass(frozen=True)
class CutResult:
    """What ``cut_clip`` wrote."""

    frames: int
    proxy_width: int
    proxy_height: int


def _even(value: float) -> int:
    return max(2, round(value / 2) * 2)


def _proxy_stream(
    container: av.container.OutputContainer,
    size: tuple[int, int],
    rate: Fraction,
    time_base: Fraction,
    bit_rate: int,
) -> VideoStream:
    codec = "h264_videotoolbox" if "h264_videotoolbox" in av.codecs_available else "libx264"
    stream = cast(VideoStream, container.add_stream(codec, rate=rate))
    stream.width, stream.height = size
    stream.pix_fmt = "nv12" if codec.endswith("toolbox") else "yuv420p"
    stream.time_base = time_base
    stream.codec_context.time_base = time_base
    stream.codec_context.bit_rate = bit_rate
    return stream


def cut_clip(
    info: VideoInfo,
    settings: Settings,
    *,
    start_s: float,
    duration_s: float,
    clip_path: Path,
    proxy_path: Path,
    proxy_max_width: int = PROXY_MAX_WIDTH,
    progress: ProgressCallback | None = None,
) -> CutResult:
    """Write ``clip.mp4`` (full resolution) and ``proxy.mp4`` (≤ ``proxy_max_width``) from the same frames.

    Args:
        info: Probed source video.
        settings: Encoder settings.
        start_s: Clip start in the source, in seconds.
        duration_s: Clip duration, in seconds.
        clip_path: Full-resolution clip to write.
        proxy_path: Annotation proxy to write.
        proxy_max_width: Maximum proxy width.
        progress: Called with ``(frames_done, frames_expected)``.

    Returns:
        Frame count and proxy size.

    Raises:
        ClipError: For rotated videos or an empty clip.
    """
    if info.rotation:
        msg = "rotated videos are not supported in the dataset yet (annotation coordinates would not match)"
        raise ClipError(msg)
    factor = min(1.0, proxy_max_width / info.width)
    proxy_size = (_even(info.width * factor), _even(info.height * factor))
    expected = max(1, round(duration_s * info.fps))
    choice = choose_encoder(info, settings)
    frames = 0
    with (
        av.open(str(info.path)) as source,
        av.open(str(clip_path), "w") as clip_out,
        av.open(str(proxy_path), "w") as proxy_out,
    ):
        stream = source.streams.video[0]
        configure_decoder(stream)
        time_base = to_fraction(stream.time_base) if stream.time_base else Fraction(1, 90000)
        origin = stream.start_time or 0
        if start_s > 0:
            source.seek(origin + int(start_s / time_base), stream=stream, backward=True)
        clip_stream = cast(
            VideoStream, clip_out.add_stream(choice.codec, rate=info.avg_frame_rate, options=choice.options)
        )
        clip_stream.width, clip_stream.height, clip_stream.pix_fmt = info.width, info.height, choice.pix_fmt
        clip_stream.time_base = time_base
        clip_stream.codec_context.time_base = time_base
        clip_stream.codec_context.color_range = choice.color_range
        if choice.bit_rate:
            clip_stream.codec_context.bit_rate = choice.bit_rate
        if choice.codec_tag:
            clip_stream.codec_context.codec_tag = choice.codec_tag
        # H.264 proxy: at most 25 Mbit/s, and not more than 1.5 × the source bit rate.
        bit_rate = min(_PROXY_BIT_RATE, int(info.bit_rate * 1.5)) if info.bit_rate else _PROXY_BIT_RATE
        proxy_stream = _proxy_stream(
            proxy_out, proxy_size, to_fraction(info.avg_frame_rate), time_base, bit_rate
        )
        first_pts: int | None = None
        for frame in source.decode(stream):
            if frame.pts is None or (frame.pts - origin) * time_base < start_s - float(time_base) / 2:
                continue
            if frames >= expected:
                break
            first_pts = frame.pts if first_pts is None else first_pts
            pts = frame.pts - first_pts
            full = frame.reformat(
                format=choice.pix_fmt, src_color_range=choice.color_range, dst_color_range=choice.color_range
            )
            full.pts, full.time_base = pts, time_base
            clip_out.mux(clip_stream.encode(full))
            small = frame.reformat(width=proxy_size[0], height=proxy_size[1], format=proxy_stream.pix_fmt)
            small.pts, small.time_base = pts, time_base
            proxy_out.mux(proxy_stream.encode(small))
            frames += 1
            if progress is not None:
                progress(frames, expected)
        clip_out.mux(clip_stream.encode(None))
        proxy_out.mux(proxy_stream.encode(None))
    if frames == 0:
        msg = f"no frame between {start_s} s and {start_s + duration_s} s"
        raise ClipError(msg)
    try:
        transplant(info.path, clip_path)
    except Mp4BoxError as exc:
        logger.warning("projection metadata not copied to the clip: %s", exc)
    return CutResult(frames=frames, proxy_width=proxy_size[0], proxy_height=proxy_size[1])


def _visible(box: Box, wrap_width: int | None) -> Box:
    """Box to show in CVAT: on 360° video, the larger part of a box crossing the seam."""
    if not wrap_width:
        return box
    return max(split(normalize(box, wrap_width), wrap_width), key=width)


#: Pre-annotation links fragments more loosely than signs (distance and gap factors):
#: one CVAT track per object saves the annotator from merging fragments by hand.
PREANNOTATION_LINK_DISTANCE_FACTOR = 3.0
PREANNOTATION_LINK_GAP_FACTOR = 2.0
#: A pre-annotated track is split where the object is not detected for longer than this.
PREANNOTATION_MAX_GAP_S = 2.0


def preannotation_tracks(
    detections: Detections,
    settings: Settings,
    *,
    conf: float,
    fps: float,
    proxy_factor: float,
    wrap_width: int | None,
    keyframe_step: int,
    min_frames: int = 1,
) -> list[PreTrack]:
    """Face and plate tracks for CVAT, from the model's detections of the clip.

    Detections are linked like signs (``core.linking``), with looser limits
    (``PREANNOTATION_LINK_*_FACTOR``). A chain is pre-annotated when one of its
    detections reaches ``conf`` and it was detected on at least ``min_frames``
    frames: on 8K 360° video the model reports hundreds of flickering
    fragments, and deleting false tracks costs the annotator more than drawing
    the missed ones. A chain is split where the object is not detected for
    longer than ``PREANNOTATION_MAX_GAP_S``, and keeps one key box every
    ``keyframe_step`` frames (CVAT interpolates in between).

    Args:
        detections: Detections of the clip.
        settings: Class policy and linking settings.
        conf: Minimum best score of a pre-annotated chain.
        fps: Clip frame rate.
        proxy_factor: Scale from clip pixels to proxy pixels.
        wrap_width: Clip width if it is a 360° video.
        keyframe_step: Frames between two key boxes.
        min_frames: Minimum number of frames on which a chain was detected.

    Returns:
        Tracks in proxy pixels.
    """
    blur_classes = settings.classes_with(ClassAction.BLUR)
    groups = {n: g for n, g in class_groups(settings.class_policy).items() if n in blur_classes}
    fragments = fragments_from_detections(detections, groups, wrap_width=wrap_width)
    chains = link_fragments(
        fragments,
        max_gap=max(1, round(settings.link_max_gap_s * PREANNOTATION_LINK_GAP_FACTOR * fps)),
        max_distance=settings.link_max_distance * PREANNOTATION_LINK_DISTANCE_FACTOR,
        wrap_width=wrap_width,
    )
    max_gap = max(1, round(PREANNOTATION_MAX_GAP_S * fps))
    tracks = []
    for chain in chains:
        if max(o.score for o in chain.observations) < conf:
            continue
        if len({o.frame for o in chain.observations}) < min_frames:
            continue
        per_frame: dict[int, Box] = {}
        for obs in chain.observations:
            box = _visible(obs.box, wrap_width)
            per_frame[obs.frame] = union_box([per_frame[obs.frame], box]) if obs.frame in per_frame else box
        boxes = {f: scale(b, proxy_factor, proxy_factor) for f, b in per_frame.items()}
        segment: list[KeyBox] = []
        last_frame = -1
        for frame in sorted(boxes):
            if segment and frame - last_frame > max_gap:
                tracks.append(_close(chain.cls, segment, boxes, last_frame))
                segment = []
            if not segment or frame - segment[-1].frame >= keyframe_step:
                segment.append(KeyBox(frame, boxes[frame]))
            last_frame = frame
        tracks.append(_close(chain.cls, segment, boxes, last_frame))
    return tracks


def _close(label: str, segment: list[KeyBox], boxes: dict[int, Box], last: int) -> PreTrack:
    """A pre-annotation track ending with a key box on its last detected frame."""
    if segment[-1].frame != last:
        segment.append(KeyBox(last, boxes[last]))
    return PreTrack(label, tuple(segment), last + 1)
