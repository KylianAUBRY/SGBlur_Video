"""Adapter around Ultralytics tracker classes (one tracker per class group).

Why not ``model.track()``: it drops detections that no track claims (orphans),
cannot merge several passes or tiles, and keeps state on a shared predictor
(``docs/adr/0002``). Here the tracker classes are driven directly with our
merged detections. Each tracker only *assigns ids*: the boxes we blur are
always the detector's boxes, never the Kalman predictions.

Ultralytics track ids come from a class attribute shared by every tracker of
the process, so ids are unique across groups inside one job; jobs run in
separate processes.
"""

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import yaml

from sgblur_video.core.detect import Detection
from sgblur_video.core.geometry import scale

logger = logging.getLogger(__name__)

#: sgblur-video extension key, converted to Ultralytics' ``track_buffer`` (frames).
TRACK_BUFFER_SECONDS_KEY = "track_buffer_s"


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
    """

    def __init__(
        self, config: Mapping[str, Any], groups: Mapping[str, str], frame_size: tuple[int, int]
    ) -> None:
        from ultralytics.trackers.track import TRACKER_MAP
        from ultralytics.utils import IterableSimpleNamespace

        tracker_type = str(config["tracker_type"])
        if tracker_type not in TRACKER_MAP:
            msg = f"unknown tracker_type {tracker_type!r}; available: {sorted(TRACKER_MAP)}"
            raise ValueError(msg)
        self._groups = dict(groups)
        self._class_index = {name: i for i, name in enumerate(sorted(groups))}
        self._frame_size = frame_size
        self._trackers = {
            group: TRACKER_MAP[tracker_type](IterableSimpleNamespace(**config))
            for group in sorted(set(groups.values()))
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
