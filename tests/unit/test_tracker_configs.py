"""The shipped tracker YAMLs must match the keys of the pinned Ultralytics version.

Ultralytics silently ignores unknown keys and falls back to defaults for
missing ones, so a renamed key after an upgrade would change tracking without
any error. This test catches it.
"""

import importlib.util
from pathlib import Path

import pytest
import yaml

# sgblur-video extension, converted to Ultralytics' `track_buffer` (frames) at runtime.
EXTENSION_KEYS = {"track_buffer_s"}
REPLACED_KEYS = {"track_buffer"}


def _ultralytics_tracker_dir() -> Path:
    # Locate the installed package without importing it (importing creates settings files).
    spec = importlib.util.find_spec("ultralytics")
    assert spec is not None
    assert spec.submodule_search_locations
    return Path(next(iter(spec.submodule_search_locations))) / "cfg" / "trackers"


def _configs(repo_root: Path) -> list[Path]:
    return sorted((repo_root / "configs" / "trackers").glob("*.yaml"))


def test_tracker_configs_exist(repo_root: Path) -> None:
    names = {path.name for path in _configs(repo_root)}
    assert {"flow.yaml", "tracktrack-recall.yaml", "botsort-recall.yaml", "bytetrack-recall.yaml"} <= names


@pytest.mark.parametrize("name", ["tracktrack-recall.yaml", "botsort-recall.yaml", "bytetrack-recall.yaml"])
def test_keys_match_upstream(repo_root: Path, name: str) -> None:
    ours = yaml.safe_load((repo_root / "configs" / "trackers" / name).read_text(encoding="utf-8"))
    upstream_file = _ultralytics_tracker_dir() / f"{ours['tracker_type']}.yaml"
    upstream = yaml.safe_load(upstream_file.read_text(encoding="utf-8"))
    expected = (set(upstream) - REPLACED_KEYS) | EXTENSION_KEYS
    assert set(ours) == expected, f"{name}: missing {expected - set(ours)}, unknown {set(ours) - expected}"
    assert ours["track_buffer_s"] > 0
    assert 0 < ours["track_low_thresh"] <= ours["track_high_thresh"] <= ours["new_track_thresh"]


def test_tracktrack_compensates_camera_motion(repo_root: Path) -> None:
    config = yaml.safe_load(
        (repo_root / "configs" / "trackers" / "tracktrack-recall.yaml").read_text(encoding="utf-8")
    )
    assert config["gmc_method"] != "none"


def test_default_tracker_is_precision_oriented_botsort(repo_root: Path) -> None:
    from sgblur_video.config import Settings
    from sgblur_video.core.track import load_tracker_config

    settings = Settings()
    assert settings.tracker_config == Path("configs/trackers/botsort.yaml")
    config = load_tracker_config(repo_root / settings.tracker_config, fps=30.0)
    assert config["tracker_type"] == "botsort"
    assert config["track_buffer"] == 15  # half a second: a lost track is not kept long
    # Low-score detections only extend tracks; they never start one.
    assert config["track_low_thresh"] == settings.conf_detect
    assert config["new_track_thresh"] > config["track_low_thresh"]
