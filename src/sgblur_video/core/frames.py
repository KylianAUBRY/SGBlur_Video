"""Best-frame JPEGs: one blurred picture per frame that holds a sign's best view.

These pictures bridge to today's Panoramax, which accepts pictures but not
videos: they can be uploaded with ``isBlurred=true`` and their annotations
(backend ≥ 2.16). Every JPEG is taken from the **blurred** frame, in display
orientation, so faces and plates on it are blurred like in the video.

EXIF carries the capture date (container ``creation_time`` + frame timestamp);
GPS is added when telemetry provides positions (step 7).
"""

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import av
import numpy as np
from PIL import Image

from sgblur_video.config import Settings
from sgblur_video.core.decode import iter_frames, rotate_upright
from sgblur_video.core.postprocess import BlurPlan
from sgblur_video.core.probe import VideoInfo
from sgblur_video.privacy.blur import blur_frame
from sgblur_video.semantics.annotations import Annotation

logger = logging.getLogger(__name__)

JPEG_QUALITY = 92
INDEX_FILE = "frames.json"

# EXIF tag numbers (TIFF/EXIF 2.32).
_EXIF_IFD = 0x8769
_DATETIME = 0x0132
_DATETIME_ORIGINAL = 0x9003
_SUBSEC_ORIGINAL = 0x9291
_OFFSET_ORIGINAL = 0x9011
_SOFTWARE = 0x0131


def _utc_offset(moment: datetime) -> str:
    """EXIF ``OffsetTimeOriginal`` value (``+HH:MM``)."""
    minutes = int((moment.utcoffset() or timedelta(0)).total_seconds() // 60)
    hours, rest = divmod(abs(minutes), 60)
    return f"{'+' if minutes >= 0 else '-'}{hours:02d}:{rest:02d}"


def _creation_time(metadata: Mapping[str, str]) -> datetime | None:
    raw = metadata.get("creation_time")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


@dataclass
class BestFrameWriter:
    """Collects the frames holding best sign views and writes them as JPEG.

    Args:
        out_dir: Destination folder (created if needed).
        annotations: Sign annotations; each one's ``video.best_frame`` is extracted.
        info: Probed video (rotation, creation time).
        software: Value of the EXIF ``Software`` tag.
    """

    out_dir: Path
    annotations: Sequence[Annotation]
    info: VideoInfo
    software: str = "SGBlur-Video"
    written: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._wanted: dict[int, list[int]] = {}
        for index, annotation in enumerate(self.annotations):
            self._wanted.setdefault(annotation.video.best_frame, []).append(index)
        self._start = _creation_time(self.info.metadata)

    @property
    def wanted_frames(self) -> frozenset[int]:
        """Frame indices that must be written."""
        return frozenset(self._wanted)

    def __call__(self, index: int, frame: av.VideoFrame) -> None:
        """Write the JPEG of frame ``index`` if a sign's best view is on it (render sink)."""
        indices = self._wanted.get(index)
        if not indices:
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        image = rotate_upright(
            np.asarray(frame.to_ndarray(format="rgb24"), dtype=np.uint8), self.info.rotation
        )
        number = len(self.written)
        name = f"{number}.jpg"
        timestamp = self.annotations[indices[0]].video.best_timestamp
        picture = Image.fromarray(image)
        exif = picture.getexif()
        exif[_SOFTWARE] = self.software
        if self._start is not None:
            taken = self._start + timedelta(seconds=timestamp)
            exif[_DATETIME] = taken.strftime("%Y:%m:%d %H:%M:%S")
            details = exif.get_ifd(_EXIF_IFD)
            details[_DATETIME_ORIGINAL] = taken.strftime("%Y:%m:%d %H:%M:%S")
            details[_SUBSEC_ORIGINAL] = f"{taken.microsecond // 1000:03d}"
            details[_OFFSET_ORIGINAL] = _utc_offset(taken)
        picture.save(self.out_dir / name, format="JPEG", quality=JPEG_QUALITY, exif=exif)
        self.written.append(
            {
                "n": number,
                "file": name,
                "frame": index,
                "timestamp": timestamp,
                "width": image.shape[1],
                "height": image.shape[0],
                "annotation_indices": indices,
                "shapes": [list(self.annotations[i].shape) for i in indices],
            }
        )

    def write_index(self) -> Path:
        """Write ``frames.json`` listing the pictures and their annotations."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / INDEX_FILE
        path.write_text(json.dumps({"frames": self.written}, indent=2), encoding="utf-8")
        return path


def extract_best_frames(
    info: VideoInfo,
    plan: BlurPlan,
    writer: BestFrameWriter,
    settings: Settings,
    *,
    max_frames: int | None = None,
) -> list[dict[str, Any]]:
    """Decode the video and write the blurred best frames (when no full render is needed).

    Args:
        info: Probed video.
        plan: Blur plan (faces and plates on the extracted frames are blurred).
        writer: Best-frame writer.
        settings: Blur settings.
        max_frames: Stop after this many frames.

    Returns:
        The index entries of the written pictures.
    """
    wanted = writer.wanted_frames
    last = max(wanted, default=-1)
    rng = np.random.default_rng()
    for decoded in iter_frames(info.path, max_frames=max_frames):
        if decoded.index > last:
            break
        if decoded.index not in wanted:
            continue
        shapes = [(s.kind, s.box) for s in plan.shapes(decoded.index)]
        frame = (
            blur_frame(decoded.frame, shapes, settings.blur_method, cells=settings.pixelate_cells, rng=rng)
            if shapes
            else decoded.frame
        )
        writer(decoded.index, frame)
    writer.write_index()
    logger.info("wrote %d best-frame pictures to %s", len(writer.written), writer.out_dir)
    return writer.written
