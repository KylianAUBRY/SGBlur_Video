"""Optical-flow tracker: small objects moving more than their width per frame keep their id."""

from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import pytest

from sgblur_video.config import DEFAULT_CLASS_POLICY
from sgblur_video.core.detect import Detection, class_groups
from sgblur_video.core.flowtrack import FlowTracker
from sgblur_video.core.track import GroupTrackers, load_tracker_config

WIDTH, HEIGHT = 320, 160


def _texture(seed: int = 0) -> npt.NDArray[np.uint8]:
    noise = np.random.default_rng(seed).integers(0, 255, (HEIGHT, WIDTH), dtype=np.uint8)
    return np.asarray(cv2.GaussianBlur(noise, (5, 5), 1.5), dtype=np.uint8)


def test_small_fast_object_keeps_its_id() -> None:
    # An 8 px box moving 20 px (2.5 box sizes) per frame with the image: no overlap between frames.
    tracker = FlowTracker(max_missed=5, gate=1.0, frame_size=(WIDTH, HEIGHT))
    base, previous, ids = _texture(), None, []
    for frame in range(6):
        gray = np.roll(base, 20 * frame, axis=1)
        x = 60 + 20 * frame
        ids += tracker.update([(x, 70, x + 8, 78)], gray, previous)
        previous = gray
    assert len(set(ids)) == 1


def test_objects_with_different_motions_keep_separate_ids() -> None:
    # The moving object passes 40 px from a static one: the flow keeps them apart.
    tracker = FlowTracker(max_missed=5, gate=1.0, frame_size=(WIDTH, HEIGHT))
    base, previous = _texture(1), None
    moving, static = [], []
    for frame in range(5):
        gray = base.copy()
        gray[:, :160] = np.roll(base, 12 * frame, axis=1)[:, :160]  # the left half moves right
        x = 40 + 12 * frame
        first, second = tracker.update([(x, 40, x + 8, 48), (200, 40, 208, 48)], gray, previous)
        moving.append(first)
        static.append(second)
        previous = gray
    assert len(set(moving)) == 1
    assert len(set(static)) == 1
    assert moving[0] != static[0]


def test_distances_wrap_around_the_360_seam() -> None:
    tracker = FlowTracker(max_missed=5, gate=1.0, frame_size=(WIDTH, HEIGHT), wrap=True)
    ids = []
    for x in (300, 306, 312, 318, 4, 10):  # 6 px per frame, crosses x = 320 → 0
        ids += tracker.update([(x, 70, x + 8, 78)], None, None)
    assert len(set(ids)) == 1


def test_lost_tracks_end_after_max_missed() -> None:
    tracker = FlowTracker(max_missed=2, gate=1.0, frame_size=(WIDTH, HEIGHT))
    first = tracker.update([(100, 70, 108, 78)], None, None)
    for _ in range(3):
        tracker.update([], None, None)
    assert tracker.update([(100, 70, 108, 78)], None, None) != first


def test_group_trackers_use_the_flow_tracker(repo_root: Path) -> None:
    config = load_tracker_config(repo_root / "configs" / "trackers" / "flow.yaml", fps=30.0)
    assert config["track_buffer"] == 30
    trackers = GroupTrackers(config, class_groups(DEFAULT_CLASS_POLICY), (WIDTH, HEIGHT))
    base = cv2.cvtColor(_texture(), cv2.COLOR_GRAY2BGR)
    plate_ids, face_ids = [], []
    for frame in range(4):
        image = np.roll(base, 20 * frame, axis=1)
        x = 60 + 20 * frame
        face_x = 150 + 20 * frame
        detections = [
            Detection("plate", 0.3, (x, 70, x + 8, 78)),
            Detection("face", 0.9, (face_x, 20, face_x + 30, 50)),
        ]
        trackers.update(detections, image, 1.0)
        plate_ids.append(detections[0].track_id)
        face_ids.append(detections[1].track_id)
    assert len(set(plate_ids)) == 1
    assert plate_ids[0] is not None
    assert plate_ids[0].startswith("plate:")
    assert len(set(face_ids)) == 1


def test_unknown_flow_keys_are_refused() -> None:
    config = {"tracker_type": "flow", "gate": 2.0, "gaet_cap": 32, "track_buffer": 30}
    with pytest.raises(ValueError, match="gaet_cap"):
        GroupTrackers(config, class_groups(DEFAULT_CLASS_POLICY), (WIDTH, HEIGHT))
