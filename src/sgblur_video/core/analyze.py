"""Pass 1: decode, detect (multi-scale), merge, track, write ``detections.jsonl``.

The work of each frame runs in three overlapping stages, so that the accelerator
does not wait for the CPU:

1. a thread decodes the next frames, converts them to BGR and prepares the
   detector inputs (360° padding, tiles) and the small tracking image;
2. the calling thread runs inference and merges the detections;
3. a thread tracks them and writes ``detections.jsonl``, in frame order.

Bounded queues keep at most ``_READY_FRAMES`` prepared frames in memory (about
150 MB each for 8K; one on CPU, where inference and preparation share the cores
anyway). An error in any stage stops the others and is re-raised.
Results are identical to running the stages one after the other.
"""

import logging
import queue
import threading
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import numpy.typing as npt

from sgblur_video import __version__
from sgblur_video.config import Settings
from sgblur_video.core.decode import iter_frames, rotate_upright, to_bgr
from sgblur_video.core.detect import (
    Detection,
    FrameDetector,
    PreparedFrame,
    build_plan,
    class_groups,
    infer_frame,
    merge_detections,
    prepare_frame,
    unrotate_box,
)
from sgblur_video.core.detections_io import DetectionRecord, DetectionsWriter, Footer, FrameRecord, Header
from sgblur_video.core.probe import VideoInfo
from sgblur_video.core.track import GroupTrackers, load_tracker_config

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int | None], None]

#: Prepared frames waiting for inference, and inferred frames waiting for tracking.
_READY_FRAMES = 2
_INFERRED_FRAMES = 4
_POLL_S = 0.2
_JOIN_TIMEOUT_S = 10.0
_END = object()  # end of a stage's output


@dataclass(frozen=True)
class _Frame:
    """Timing and tracking image of a frame, passed from stage to stage."""

    index: int
    pts: int
    time: float
    small: npt.NDArray[np.uint8]


@dataclass(frozen=True)
class _Prepared:
    """A frame ready for inference: its detector inputs (about 150 MB for 8K, dropped after it)."""

    frame: _Frame
    inputs: PreparedFrame


class _Stage:
    """A pipeline stage in a thread; its error is kept and re-raised by :meth:`check`."""

    def __init__(self, target: Callable[[], None], name: str) -> None:
        self.error: BaseException | None = None
        self._target = target
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)

    def _run(self) -> None:
        try:
            self._target()
        except BaseException as exc:
            self.error = exc

    def start(self) -> None:
        """Start the thread."""
        self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        """Wait for the thread."""
        self._thread.join(timeout)

    def check(self) -> None:
        """Re-raise the error of the stage, if any."""
        if self.error is not None:
            raise self.error


def _put(target: queue.Queue[Any], item: object, abort: Callable[[], bool]) -> bool:
    """Put without blocking forever: give up (False) as soon as ``abort()`` is true."""
    while not abort():
        try:
            target.put(item, timeout=_POLL_S)
        except queue.Full:
            continue
        return True
    return False


def _get(source: queue.Queue[Any], stop: threading.Event) -> object:
    """Get the next item, or ``_END`` once the pipeline is stopped."""
    while True:
        try:
            return source.get(timeout=_POLL_S)
        except queue.Empty:
            if stop.is_set():
                return _END


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
        equirect_pad_ratio=settings.equirect_pad_ratio,
    )
    wrap_width = info.width if info.projection == "equirectangular" else None
    groups = class_groups(settings.class_policy)
    tracker_config = load_tracker_config(settings.tracker_config, info.fps)
    factor = min(1.0, settings.track_width / info.width)
    track_size = (max(2, round(info.width * factor)), max(2, round(info.height * factor)))
    trackers = GroupTrackers(tracker_config, groups, track_size, wrap=wrap_width is not None)
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
            "equirect_pad_px": plan[0].pad,
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
    written = 0
    logger.info("analysis: %d passes per frame (%s)", len(plan), ", ".join(p.id for p in plan))

    # Three stages overlap so that the accelerator never waits for the CPU (see the module
    # docstring): preparation and tracking run in threads, inference in this thread.
    stop = threading.Event()
    ready: queue.Queue[_Prepared | object] = queue.Queue(maxsize=1 if device == "cpu" else _READY_FRAMES)
    inferred: queue.Queue[tuple[_Frame, list[Detection]] | object] = queue.Queue(maxsize=_INFERRED_FRAMES)

    def prepare_frames() -> None:
        try:
            for decoded in iter_frames(info.path, max_frames=max_frames):
                full = to_bgr(decoded.frame)
                prepared = prepare_frame(detector, rotate_upright(full, info.rotation), plan)
                small = (
                    np.asarray(cv2.resize(full, track_size, interpolation=cv2.INTER_AREA), dtype=np.uint8)
                    if factor < 1
                    else full
                )
                item = _Prepared(_Frame(decoded.index, decoded.pts, decoded.time, small), prepared)
                if not _put(ready, item, stop.is_set):
                    return
        finally:
            _put(ready, _END, stop.is_set)

    def track_and_write() -> None:
        nonlocal written
        while (item := _get(inferred, stop)) is not _END:
            frame, merged = cast(tuple[_Frame, list[Detection]], item)
            trackers.update(merged, frame.small, factor)
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
                FrameRecord(index=frame.index, pts=frame.pts, time=frame.time, detections=records)
            )
            written += 1

    with DetectionsWriter(output, header) as writer:
        preparer = _Stage(prepare_frames, "analysis-prepare")
        tracker = _Stage(track_and_write, "analysis-track")
        preparer.start()
        tracker.start()

        def tracking_failed() -> bool:
            return tracker.error is not None

        try:
            inferences = 0
            while (item := _get(ready, stop)) is not _END:
                ready_frame = cast(_Prepared, item)
                frame = ready_frame.frame
                raw = infer_frame(detector, ready_frame.inputs, plan, frame.index)
                del item, ready_frame  # release the detector inputs before waiting for the next frame
                for det in raw:
                    det.box = unrotate_box(det.box, info.rotation, info.width, info.height)
                merged = merge_detections(raw, settings.class_policy, wrap_width=wrap_width)
                if not _put(inferred, (frame, merged), tracking_failed):
                    tracker.check()  # the tracking stage failed: raise its error
                inferences += 1
                if progress is not None:
                    progress(inferences, info.frame_count)
            preparer.join()
            preparer.check()
            if not _put(inferred, _END, tracking_failed):
                tracker.check()
            tracker.join()
            tracker.check()
        finally:
            stop.set()
            preparer.join(timeout=_JOIN_TIMEOUT_S)
            tracker.join(timeout=_JOIN_TIMEOUT_S)
        footer = Footer(
            frames=written,
            complete=max_frames is None or written < max_frames,
            elapsed_s=round(time.monotonic() - started, 2),
            counts=dict(counts),
        )
        writer.close_with(footer)
    logger.info("analysis done: %d frames in %.1f s, %s", written, footer.elapsed_s, dict(counts))
    return footer
