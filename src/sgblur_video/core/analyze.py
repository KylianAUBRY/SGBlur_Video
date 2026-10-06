"""Pass 1: decode, detect (multi-scale), merge, track, write ``detections.jsonl``."""

import logging
import time
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from sgblur_video import __version__
from sgblur_video.config import Settings
from sgblur_video.core.decode import iter_frames, rotate_upright, to_bgr
from sgblur_video.core.detect import FrameDetector, build_plan, class_groups, merge_detections, unrotate_box
from sgblur_video.core.detections_io import DetectionRecord, DetectionsWriter, Footer, FrameRecord, Header
from sgblur_video.core.probe import VideoInfo
from sgblur_video.core.track import GroupTrackers, load_tracker_config

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int | None], None]


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def _round_box(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return (round(box[0], 1), round(box[1], 1), round(box[2], 1), round(box[3], 1))


def analyze(
    info: VideoInfo,
    detector: FrameDetector,
    settings: Settings,
    output: Path,
    *,
    model: dict[str, Any],
    device: str = "cpu",
    max_frames: int | None = None,
    progress: ProgressCallback | None = None,
) -> Footer:
    """Run pass 1 on a video and write ``detections.jsonl``.

    Args:
        info: Probed input video.
        detector: YOLO detector (or a fake detector in tests).
        settings: Detection, tracking and policy settings.
        output: ``detections.jsonl`` to write.
        model: Model description for the header (name, version, sha256, classes, policy).
        device: Device used, recorded in the header.
        max_frames: Stop after this many frames (the footer then says ``complete: false``).
        progress: Called with ``(frames_done, frames_total)``.

    Returns:
        The footer written at the end of the file.
    """
    started = time.monotonic()
    rotated = (round(info.rotation / 90) % 2) == 1 if info.rotation else False
    upright_w, upright_h = (info.height, info.width) if rotated else (info.width, info.height)
    plan = build_plan(
        upright_w,
        upright_h,
        projection=info.projection,
        profile=settings.detect_profile,
        tile_trigger_width=settings.tile_trigger_width,
    )
    groups = class_groups(settings.class_policy)
    tracker_config = load_tracker_config(settings.tracker_config, info.fps)
    factor = min(1.0, settings.track_width / info.width)
    track_size = (max(2, round(info.width * factor)), max(2, round(info.height * factor)))
    trackers = GroupTrackers(tracker_config, groups, track_size)
    header = Header(
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        video={
            "width": info.width,
            "height": info.height,
            "rotation": info.rotation,
            "time_base": str(info.time_base),
            "avg_frame_rate": str(info.avg_frame_rate),
            "frame_count": info.frame_count,
            "duration_s": round(info.duration_s, 3),
            "codec": info.codec,
            "pix_fmt": info.pix_fmt,
            "projection": info.projection,
            "projection_source": info.projection_source,
        },
        model=model,
        detection={
            "profile": settings.detect_profile.value,
            "conf": settings.conf_detect,
            "passes": [
                {
                    "id": p.id,
                    "kind": p.kind,
                    "imgsz": p.imgsz,
                    **({"region": list(p.region)} if p.region else {}),
                }
                for p in plan
            ],
        },
        tracking={
            "tracker": tracker_config["tracker_type"],
            "config": str(settings.tracker_config),
            "track_buffer": tracker_config.get("track_buffer"),
            "track_width": track_size[0],
            "groups": {
                group: sorted(c for c, g in groups.items() if g == group) for group in set(groups.values())
            },
        },
        software={
            "sgblur_video": __version__,
            "ultralytics": _package_version("ultralytics"),
            "torch": _package_version("torch"),
            "av": _package_version("av"),
            "device": device,
        },
    )
    counts: Counter[str] = Counter()
    frames_done = 0
    logger.info("analysis: %d passes per frame (%s)", len(plan), ", ".join(p.id for p in plan))
    with DetectionsWriter(output, header) as writer:
        for decoded in iter_frames(info.path, max_frames=max_frames):
            full = to_bgr(decoded.frame)
            upright = rotate_upright(full, info.rotation)
            raw = detector.detect(upright, plan, decoded.index)
            for det in raw:
                det.box = unrotate_box(det.box, info.rotation, info.width, info.height)
            merged = merge_detections(raw, settings.class_policy)
            small = (
                np.asarray(cv2.resize(full, track_size, interpolation=cv2.INTER_AREA), dtype=np.uint8)
                if factor < 1
                else full
            )
            trackers.update(merged, small, factor)
            records = []
            for det in merged:
                counts[det.cls] += 1
                if det.track_id is None:
                    counts["orphans"] += 1
                records.append(
                    DetectionRecord.model_validate(
                        {
                            "class": det.cls,
                            "score": round(det.score, 3),
                            "box": _round_box(det.box),
                            "track_id": det.track_id,
                            "passes": det.passes,
                        }
                    )
                )
            writer.write_frame(
                FrameRecord(index=decoded.index, pts=decoded.pts, time=decoded.time, detections=records)
            )
            frames_done += 1
            if progress is not None:
                progress(frames_done, info.frame_count)
        footer = Footer(
            frames=frames_done,
            complete=max_frames is None or frames_done < max_frames,
            elapsed_s=round(time.monotonic() - started, 2),
            counts=dict(counts),
        )
        writer.close_with(footer)
    logger.info("analysis done: %d frames in %.1f s, %s", frames_done, footer.elapsed_s, dict(counts))
    return footer
