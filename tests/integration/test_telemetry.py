"""GoPro telemetry end to end on a free sample (gopro/gpmf-parser, Apache-2.0 OR MIT)."""

from pathlib import Path

import av
import pytest
from PIL import Image

from sgblur_video.config import Settings
from sgblur_video.core.analyze import analyze
from sgblur_video.core.decode import iter_frames
from sgblur_video.core.pipeline import load_detections, render_detections
from sgblur_video.core.probe import probe
from sgblur_video.core.trim import FrameRange, trim_video
from sgblur_video.telemetry.gps import read_gps
from tests.data.fetch import fixture_path
from tests.privacy.synthetic import FakeDetector, scenario

pytestmark = [pytest.mark.integration, pytest.mark.slow]

GPS_IFD = 0x8825


def test_gps_is_preserved_and_used_for_signs(tmp_path: Path, repo_root: Path) -> None:
    source = fixture_path("gopro-hero6-gps")
    settings = Settings(
        tracker_config=repo_root / "configs" / "trackers" / "bytetrack-recall.yaml", encoder="libx264"
    )
    info = probe(source, settings)
    track = read_gps(info)
    assert track is not None
    assert len(track) > 100
    position = track.position_at(5.0)
    assert position is not None
    assert 32 < position.lat < 34  # Southern California
    assert -118 < position.lon < -116

    detections_path = tmp_path / "detections.jsonl"
    analyze(info, FakeDetector(scenario()), settings, detections_path, model={"name": "fake", "version": "0"})
    detections = load_detections(detections_path, info, allow_partial=False)
    output = tmp_path / "blurred.mp4"
    result = render_detections(info, detections, output, settings, frames_dir=tmp_path / "frames")

    # GPMF telemetry is copied, so the blurred video still carries its GPS track.
    output_info = probe(output, settings)
    assert any(s.handler_name == "GoPro MET" for s in output_info.streams)
    output_track = read_gps(output_info)
    assert output_track is not None
    assert len(output_track) == len(track)
    # GoPro camera settings (udta) are restored, the timecode track is recreated.
    assert "FIRM" in result.metadata.stats["restored_metadata"]["udta"]
    with av.open(str(output)) as container:
        assert any(s.type == "data" for s in container.streams)

    assert result.metadata.video["gps"] is True
    assert result.metadata.annotations
    assert all(a.video.position is not None for a in result.metadata.annotations)
    assert result.frames
    with Image.open(tmp_path / "frames" / result.frames[0]["file"]) as picture:
        gps = picture.getexif().get_ifd(GPS_IFD)
    assert gps[1] == "N"
    assert gps[3] == "W"


def test_gps_of_a_frame_range_stays_aligned(tmp_path: Path) -> None:
    source = fixture_path("gopro-hero6-gps")
    settings = Settings(encoder="libx264")
    info = probe(source, settings)
    first = 460  # 15.3 s: GPS fix (from 14 s on), in the middle of a 1 s telemetry packet
    cut = tmp_path / "cut.mp4"
    assert trim_video(info, settings, FrameRange(first, first + 60), cut) == 60
    original, trimmed = read_gps(info), read_gps(probe(cut, settings))
    assert original is not None
    assert trimmed is not None
    offset = [frame.time for frame in iter_frames(source, max_frames=first + 1)][first]
    # Time t of the cut is time t + offset of the original: same interpolated position. The
    # telemetry packet straddling the start is dropped: the cut has samples from 0.67 s to 1.6 s.
    for t in (0.8, 1.1, 1.5):
        here, there = trimmed.position_at(t), original.position_at(t + offset)
        assert here is not None
        assert there is not None
        assert here.lat == pytest.approx(there.lat, abs=1e-7)
        assert here.lon == pytest.approx(there.lon, abs=1e-7)
