"""Cached analysis results for benchmarks, and re-tracking of cached detections.

Detection is the expensive part (about 0.6 s per 8K frame). Benchmarks keep
every ``detections.jsonl`` they compute, keyed by what influences it:

* the **detection key** (model checksum, profile, ``CONF_DETECT``, tiling and
  360° padding, class policy, frame limit) names the detections themselves;
* the **tracking key** (tracker YAML content, ``TRACK_WIDTH``) names the track ids.

Comparing trackers then only re-runs tracking on the cached detections
(:func:`retrack`): the merged boxes are identical, only the ids change, exactly
as if the analysis had run with the other tracker. Comparing post-processing
settings (``CONF_BLUR``, padding, linking…) reuses the file as is.
"""

import hashlib
import json
import logging
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from sgblur_video.config import Settings
from sgblur_video.core.analyze import ProgressCallback, analyze
from sgblur_video.core.decode import iter_frames, to_bgr
from sgblur_video.core.detect import Detection, class_groups
from sgblur_video.core.detections_io import (
    DetectionRecord,
    Detections,
    DetectionsFormatError,
    DetectionsWriter,
    Footer,
    FrameRecord,
    read_detections,
)
from sgblur_video.core.pipeline import LoadedModel
from sgblur_video.core.probe import VideoInfo
from sgblur_video.core.track import GroupTrackers, load_tracker_config

logger = logging.getLogger(__name__)

ModelLoader = Callable[[], LoadedModel]


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:12]


def video_key(path: Path) -> str:
    """Identifier of a video file for the cache (path, size and modification time; no content read)."""
    stat = path.stat()
    return _digest([str(path.resolve()), stat.st_size, stat.st_mtime_ns])


def detection_key(settings: Settings, model_sha256: str, max_frames: int | None) -> str:
    """Key of everything that changes the detections themselves."""
    return _digest(
        {
            "model": model_sha256,
            "profile": settings.detect_profile.value,
            "conf": settings.conf_detect,
            "tile_trigger_width": settings.tile_trigger_width,
            "equirect_pad_ratio": settings.equirect_pad_ratio,
            "projection": settings.projection,
            "policy": {k: v.value for k, v in settings.class_policy.items()},
            "max_frames": max_frames,
        }
    )


def tracking_key(settings: Settings) -> str:
    """Key of everything that changes the track ids."""
    return _digest(
        {"tracker": settings.tracker_config.read_text(encoding="utf-8"), "width": settings.track_width}
    )


def _finished(path: Path) -> Detections | None:
    """A cached file, unless it is missing or was cut short by an interrupted run."""
    if not path.exists():
        return None
    try:
        detections = read_detections(path)
    except DetectionsFormatError:
        return None
    return detections if detections.footer is not None else None


class DetectionCache:
    """``detections.jsonl`` files of one video, under ``root``.

    Args:
        root: Cache folder of this video.
        model_sha256: Checksum of the model the detections come from (registry entry).
        load_model: Loads the detector on a cache miss (models are only loaded when needed).
    """

    def __init__(self, root: Path, model_sha256: str, load_model: ModelLoader) -> None:
        self.root = root
        self.model_sha256 = model_sha256
        self._load_model = load_model

    def get(
        self,
        info: VideoInfo,
        settings: Settings,
        *,
        max_frames: int | None = None,
        progress: ProgressCallback | None = None,
    ) -> tuple[Detections, dict[str, float]]:
        """Detections of the video with these settings: cached, re-tracked, or computed.

        Returns:
            The detections and timings (``analysis_s`` or ``tracking_s`` when computed now).
        """
        det_key = detection_key(settings, self.model_sha256, max_frames)
        path = self.root / f"{det_key}-{tracking_key(settings)}.jsonl"
        if (detections := _finished(path)) is not None:
            return detections, {}
        self.root.mkdir(parents=True, exist_ok=True)
        for sibling in sorted(self.root.glob(f"{det_key}-*.jsonl")):
            if sibling != path and (base := _finished(sibling)) is not None:
                started = time.monotonic()
                retrack(info, base, settings, path, progress=progress)
                return read_detections(path), {"tracking_s": time.monotonic() - started}
        model = self._load_model()
        footer = analyze(
            info,
            model.detector,
            settings,
            path,
            model=model.header(settings),
            device=model.device,
            max_frames=max_frames,
            progress=progress,
        )
        return read_detections(path), {"analysis_s": footer.elapsed_s}


def retrack(
    info: VideoInfo,
    base: Detections,
    settings: Settings,
    output: Path,
    *,
    progress: ProgressCallback | None = None,
) -> None:
    """Assign new track ids to cached detections with another tracker configuration.

    The video is decoded again because trackers with camera-motion compensation
    need the frames (at ``TRACK_WIDTH``, like during analysis).

    Args:
        info: Probed video.
        base: Detections computed with any tracker.
        settings: Settings holding the tracker to use.
        output: ``detections.jsonl`` to write.
        progress: Called with ``(frames_done, frames_total)``.
    """
    tracker_config = load_tracker_config(settings.tracker_config, info.fps)
    factor = min(1.0, settings.track_width / info.width)
    track_size = (max(2, round(info.width * factor)), max(2, round(info.height * factor)))
    trackers = GroupTrackers(
        tracker_config,
        class_groups(settings.class_policy),
        track_size,
        wrap=info.projection == "equirectangular",
    )
    header = base.header.model_copy(
        update={
            "tracking": {
                "tracker": tracker_config["tracker_type"],
                "config": str(settings.tracker_config),
                "track_buffer": tracker_config.get("track_buffer"),
                "track_width": track_size[0],
                "retracked": True,
            }
        }
    )
    started = time.monotonic()
    counts: Counter[str] = Counter()
    with DetectionsWriter(output, header) as writer:
        frames = iter(base.frames)
        for decoded in iter_frames(info.path, max_frames=len(base.frames)):
            record = next(frames)
            detections = [Detection(d.class_, d.score, d.box, list(d.passes)) for d in record.detections]
            small = np.asarray(
                cv2.resize(to_bgr(decoded.frame), track_size, interpolation=cv2.INTER_AREA)
                if factor < 1
                else to_bgr(decoded.frame),
                dtype=np.uint8,
            )
            trackers.update(detections, small, factor)
            counts.update(d.cls for d in detections)
            counts["orphans"] += sum(1 for d in detections if d.track_id is None)
            writer.write_frame(
                FrameRecord(
                    index=record.index,
                    pts=record.pts,
                    time=record.time,
                    detections=[
                        DetectionRecord.model_validate(
                            {
                                "class": d.cls,
                                "score": d.score,
                                "box": d.box,
                                "track_id": d.track_id,
                                "passes": d.passes,
                            }
                        )
                        for d in detections
                    ],
                )
            )
            if progress is not None:
                progress(record.index + 1, len(base.frames))
        complete = base.footer is not None and base.footer.complete
        writer.close_with(
            Footer(
                frames=len(base.frames),
                complete=complete,
                elapsed_s=round(time.monotonic() - started, 2),
                counts=dict(counts),
            )
        )
