"""Frame range of a job: cut the source to frames ``[start, end)`` before processing.

A stream copy can only start on a key frame, so the range is re-encoded with
the same encoder settings as the blurred output (same codec family and bit
rate as the source; above 4K, x265 holds few frames ahead to fit in memory, see ``encode``). Audio and
telemetry are copied for the same time span, shifted to start at 0, and the
box-level metadata (360°, rotation, camera boxes) is transplanted. The rest of
the pipeline then runs unchanged on the cut file, locally or on a Detect API.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from sgblur_video.config import Settings
from sgblur_video.core.mp4boxes import Mp4BoxError, transplant
from sgblur_video.core.postprocess import BlurPlan
from sgblur_video.core.probe import VideoInfo
from sgblur_video.core.render import render

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FrameRange:
    """Frames ``[start, end)`` of a video (``end`` ``None``: until the end)."""

    start: int = 0
    end: int | None = None

    def frames(self, frame_count: int | None) -> int | None:
        """Number of frames in the range for a video of ``frame_count`` frames (``None`` if unknown)."""
        if frame_count is None:
            return None if self.end is None else self.end - self.start
        last = frame_count if self.end is None else min(self.end, frame_count)
        return max(0, last - self.start)

    @property
    def is_whole(self) -> bool:
        """Whether the range is the whole video."""
        return self.start == 0 and self.end is None


def trim_video(info: VideoInfo, settings: Settings, frame_range: FrameRange, output: Path) -> int:
    """Write frames ``[start, end)`` of a video to ``output`` (MP4), with audio and telemetry.

    Args:
        info: Probed source video.
        settings: Encoder settings.
        frame_range: Frames to keep.
        output: File to write.

    Returns:
        The number of frames written (fewer than requested if the video ends first).

    Raises:
        sgblur_video.core.render.RenderError: If the range starts after the last frame.
    """
    wanted = frame_range.frames(info.frame_count)
    # Unknown length: render until the video ends (frame counts from the index can be estimates).
    count = wanted if wanted is not None else 10**9
    stats = render(
        info,
        BlurPlan(frame_count=count),
        output,
        settings,
        first_frame=frame_range.start,
        allow_short=True,
    )
    try:
        transplant(info.path, output)
    except Mp4BoxError as exc:
        logger.warning("metadata not transplanted to the cut video: %s", exc)
    logger.info("frame range %d-%s: %d frames", frame_range.start, frame_range.end, stats.frames)
    return stats.frames
