"""Tests of the Ultralytics tracker adapter (contract tests for the pinned version)."""

from pathlib import Path

import numpy as np

from sgblur_video.config import DEFAULT_CLASS_POLICY
from sgblur_video.core.detect import Detection, class_groups
from sgblur_video.core.track import GroupTrackers, load_tracker_config


def test_track_buffer_seconds_are_converted(repo_root: Path) -> None:
    config = load_tracker_config(repo_root / "configs" / "trackers" / "tracktrack-recall.yaml", fps=25.0)
    assert config["track_buffer"] == 50
    assert "track_buffer_s" not in config


def test_ids_survive_a_missed_frame_and_groups_are_separate(repo_root: Path) -> None:
    config = load_tracker_config(repo_root / "configs" / "trackers" / "tracktrack-recall.yaml", fps=30.0)
    trackers = GroupTrackers(config, class_groups(DEFAULT_CLASS_POLICY), (960, 540))
    image = np.zeros((540, 960, 3), np.uint8)
    face_ids, plate_ids = [], []
    for frame in range(8):
        detections = [Detection("plate", 0.9, (500, 300, 560, 320))]
        if frame != 4:  # the face is missed once
            detections.append(Detection("face", 0.9, (100 + 5 * frame, 100, 150 + 5 * frame, 160)))
        trackers.update(detections, image, 1.0)
        face_ids += [d.track_id for d in detections if d.cls == "face" and d.track_id]
        plate_ids += [d.track_id for d in detections if d.cls == "plate" and d.track_id]
    assert face_ids
    assert len(set(face_ids)) == 1
    assert face_ids[0].startswith("face:")
    assert len(set(plate_ids)) == 1
    assert plate_ids[0].startswith("plate:")


def test_low_score_detection_is_left_as_orphan(repo_root: Path) -> None:
    config = load_tracker_config(repo_root / "configs" / "trackers" / "tracktrack-recall.yaml", fps=30.0)
    trackers = GroupTrackers(config, class_groups(DEFAULT_CLASS_POLICY), (960, 540))
    detection = Detection("face", 0.12, (10, 10, 30, 30))
    trackers.update([detection], None, 1.0)
    assert detection.track_id is None
