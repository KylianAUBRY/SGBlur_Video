"""Adapter around Ultralytics tracker classes, or the optical-flow tracker (one per class group).

Why not ``model.track()``: it drops detections that no track claims (orphans),
cannot merge several passes or tiles, and keeps state on a shared predictor
(``docs/adr/0002``). Here the tracker classes are driven directly with our
merged detections. Each tracker only *assigns ids*: the boxes we blur are
always the detector's boxes, never the Kalman predictions.

Ultralytics track ids come from a class attribute shared by every tracker of
the process, so ids are unique across groups inside one job; jobs run in
separate processes. ``tracker_type: flow`` selects :class:`~sgblur_video.core.flowtrack.FlowTracker`
(keys ``gate`` and ``track_buffer_s``).
"""

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import yaml

from sgblur_video.core.detect import Detection
from sgblur_video.core.flowtrack import FlowTracker
from sgblur_video.core.geometry import scale

logger = logging.getLogger(__name__)

#: sgblur-video extension key, converted to Ultralytics' ``track_buffer`` (frames).
TRACK_BUFFER_SECONDS_KEY = "track_buffer_s"

#: ``tracker_type`` of :class:`~sgblur_video.core.flowtrack.FlowTracker`, and the keys it reads
#: (``track_buffer`` comes from ``track_buffer_s``). Unknown keys are refused: a misspelt key
#: would otherwise silently keep its default.
FLOW_TRACKER = "flow"
FLOW_KEYS = frozenset(
    {"tracker_type", "gate", "gate_cap", "large_gate", "context", "fb_error", "min_size", "track_buffer"}
)


def load_tracker_config(path: Path, fps: float) -> dict[str, Any]:
    """Load a tracker YAML and convert ``track_buffer_s`` into a frame count.

    Ultralytics ≥ 8.4.38 counts ``track_buffer`` in frames whatever the frame
    rate, so a buffer expressed in seconds keeps the same behaviour at 24, 30
    or 60 fps.

    Args:
        path: Tracker YAML (``configs/trackers/*.yaml``).
        fps: Video frame rate.

    Returns:
        The configuration as Ultralytics expects it.

    Raises:
        ValueError: If the file has no ``tracker_type``.
    """
    config: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    if "tracker_type" not in config:
        msg = f"{path}: missing tracker_type"
        raise ValueError(msg)
    if TRACK_BUFFER_SECONDS_KEY in config:
        config["track_buffer"] = max(1, round(float(config.pop(TRACK_BUFFER_SECONDS_KEY)) * fps))
    return config


class GroupTrackers:
    """One Ultralytics tracker per class group, fed with merged detections.

    Args:
        config: Tracker configuration from :func:`load_tracker_config`.
        groups: Mapping of class name to group name (see :func:`~sgblur_video.core.detect.class_groups`).
        frame_size: ``(width, height)`` of the frames given to :meth:`update` (tracking resolution).
        wrap: The frames are 360° equirectangular (used by the flow tracker).
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        groups: Mapping[str, str],
        frame_size: tuple[int, int],
        *,
        wrap: bool = False,
    ) -> None:
        from ultralytics.trackers.track import TRACKER_MAP
        from ultralytics.utils import IterableSimpleNamespace

        tracker_type = str(config["tracker_type"])
        if tracker_type not in {*TRACKER_MAP, FLOW_TRACKER}:
            msg = f"unknown tracker_type {tracker_type!r}; available: {sorted({*TRACKER_MAP, FLOW_TRACKER})}"
            raise ValueError(msg)
        self._groups = dict(groups)
        self._class_index = {name: i for i, name in enumerate(sorted(groups))}
        self._frame_size = frame_size
        group_names = sorted(set(groups.values()))
        self._flow: dict[str, FlowTracker] = {}
        self._trackers: dict[str, Any] = {}
        self._previous: npt.NDArray[np.uint8] | None = None
        if tracker_type == FLOW_TRACKER:
            if unknown := set(config) - FLOW_KEYS:
                msg = f"unknown keys for the flow tracker: {sorted(unknown)}"
                raise ValueError(msg)
            self._flow = {
                group: FlowTracker(
                    max_missed=int(config.get("track_buffer", 30)),
                    gate=float(config.get("gate", 2.0)),
                    gate_cap=float(config.get("gate_cap", 32.0)),
                    large_gate=float(config.get("large_gate", 0.5)),
                    frame_size=frame_size,
                    wrap=wrap,
                    min_size=float(config.get("min_size", 4.0)),
                    fb_error=float(config.get("fb_error", 2.0)),
                    context=float(config.get("context", 1.0)),
                )
                for group in group_names
            }
        else:
            self._trackers = {
                group: TRACKER_MAP[tracker_type](IterableSimpleNamespace(**config)) for group in group_names
            }

    def update(
        self, detections: Sequence[Detection], image: npt.NDArray[np.uint8] | None, factor: float
    ) -> None:
        """Assign ``track_id`` to the detections of one frame (in place).

        Every tracker is updated on every frame, even without detections, so
        that lost tracks age correctly.

        Args:
            detections: Merged detections of the frame, boxes in coded-frame pixels.
            image: BGR frame at tracking resolution (for camera-motion compensation), or ``None``.
            factor: Scale from coded-frame pixels to tracking resolution.
        """
        if self._flow:
            self._update_flow(detections, image, factor)
            return
        from ultralytics.engine.results import Boxes

        height, width = self._frame_size[1], self._frame_size[0]
        for group, tracker in self._trackers.items():
            members = [d for d in detections if self._groups.get(d.cls) == group]
            rows = np.array(
                [[*scale(d.box, factor, factor), d.score, self._class_index[d.cls]] for d in members],
                dtype=np.float32,
            ).reshape(-1, 6)
            output = tracker.update(Boxes(rows, (height, width)), image)
            for row in output:
                index = int(row[7])
                if 0 <= index < len(members):
                    members[index].track_id = f"{group}:{int(row[4])}"

    def _update_flow(
        self, detections: Sequence[Detection], image: npt.NDArray[np.uint8] | None, factor: float
    ) -> None:
        gray = (
            np.asarray(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), dtype=np.uint8) if image is not None else None
        )
        for group, tracker in self._flow.items():
            members = [d for d in detections if self._groups.get(d.cls) == group]
            ids = tracker.update([scale(d.box, factor, factor) for d in members], gray, self._previous)
            for member, track_id in zip(members, ids, strict=True):
                member.track_id = f"{group}:{track_id}"
        self._previous = gray
