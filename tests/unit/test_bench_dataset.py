import hashlib
from pathlib import Path

import pytest

from sgblur_video.bench.clips import preannotation_tracks
from sgblur_video.bench.dataset import (
    ClipEntry,
    Dataset,
    DatasetError,
    GroundTruth,
    GtBox,
    GtTrack,
    check_clip_id,
    sha256_file,
)
from sgblur_video.bench.runs import override, parse_sweeps
from sgblur_video.config import Settings
from sgblur_video.core.detections_io import DetectionRecord, Detections, FrameRecord, Header


def _entry(clip_id: str) -> ClipEntry:
    return ClipEntry(
        id=clip_id,
        source_sha256="0" * 64,
        start_s=1.5,
        frames=30,
        fps=30.0,
        width=7680,
        height=3840,
        proxy_width=3840,
        proxy_height=1920,
        projection="equirectangular",
    )


def test_manifest_round_trip(tmp_path: Path) -> None:
    dataset = Dataset(tmp_path / "ds")
    manifest = dataset.load_manifest()
    assert manifest.clips == []
    manifest.put(_entry("b"))
    manifest.put(_entry("a"))
    manifest.put(_entry("b").model_copy(update={"annotator": "me"}))
    dataset.save_manifest(manifest)
    loaded = dataset.load_manifest()
    assert [c.id for c in loaded.clips] == ["a", "b"]
    assert loaded.get("b").annotator == "me"
    with pytest.raises(DatasetError, match="no clip 'c'"):
        loaded.get("c")


def test_invalid_manifest(tmp_path: Path) -> None:
    (tmp_path / "manifest.yaml").write_text("version: 2\n", encoding="utf-8")
    with pytest.raises(DatasetError, match=r"manifest\.yaml"):
        Dataset(tmp_path).load_manifest()


def test_ground_truth_round_trip(tmp_path: Path) -> None:
    dataset = Dataset(tmp_path)
    assert dataset.load_ground_truth("a") is None
    truth = GroundTruth(
        clip_id="a",
        frames=3,
        width=10,
        height=10,
        tracks=[GtTrack(id="face:0", cls="face", boxes=[GtBox(frame=1, box=(1, 2, 3, 4), readable=True)])],
    )
    dataset.save_ground_truth(truth)
    assert dataset.load_ground_truth("a") == truth
    assert dataset.cache_dir("a") == tmp_path / "cache" / "a"


@pytest.mark.parametrize("clip_id", ["", "A", "../x", "a b", "a" * 65])
def test_clip_ids_are_safe_folder_names(clip_id: str) -> None:
    with pytest.raises(DatasetError):
        check_clip_id(clip_id)


def test_sha256_file(tmp_path: Path) -> None:
    path = tmp_path / "f"
    path.write_bytes(b"x" * 100)
    assert sha256_file(path, chunk=7) == hashlib.sha256(b"x" * 100).hexdigest()


def test_sweeps_and_overrides() -> None:
    assert parse_sweeps([]) == [{}]
    assert parse_sweeps(["CONF_BLUR=0.1,0.2", "link_max_gap_s=1"]) == [
        {"CONF_BLUR": "0.1", "link_max_gap_s": "1"},
        {"CONF_BLUR": "0.2", "link_max_gap_s": "1"},
    ]
    with pytest.raises(ValueError, match="invalid --sweep"):
        parse_sweeps(["CONF_BLUR"])
    settings = override(Settings(), {"CONF_BLUR": "0.3", "detect_profile": "fast"})
    assert settings.conf_blur == 0.3
    assert settings.detect_profile.value == "fast"
    with pytest.raises(ValueError, match="unknown setting"):
        override(Settings(), {"NOPE": "1"})
    with pytest.raises(ValueError, match="conf_blur"):
        override(Settings(), {"CONF_BLUR": "high"})


def _detections(
    frames: dict[int, list[tuple[str, float, tuple[float, float, float, float], str | None]]],
) -> Detections:
    header = Header.model_validate(
        {"created_at": "now", "video": {}, "model": {}, "detection": {}, "tracking": {}, "software": {}}
    )
    records = [
        FrameRecord(
            index=i,
            pts=i,
            time=i / 30,
            detections=[
                DetectionRecord.model_validate({"class": c, "score": s, "box": b, "track_id": t})
                for c, s, b, t in frames.get(i, [])
            ],
        )
        for i in range(max(frames) + 1)
    ]
    return Detections(header=header, frames=records, footer=None)


def test_preannotation_tracks_split_long_gaps_and_keep_key_boxes() -> None:
    settings = Settings(max_interpolation_gap_s=0.1, link_max_gap_s=5.0, link_max_distance=10.0)
    face = ("face", 0.5, (100.0, 100.0, 120.0, 120.0), "face:1")
    frames = {f: [face] for f in [*range(0, 10), *range(20, 25)]}
    frames[30] = [("plate", 0.05, (0.0, 0.0, 10.0, 5.0), None)]  # below the pre-annotation threshold
    tracks = preannotation_tracks(
        _detections(frames), settings, conf=0.1, fps=30.0, proxy_factor=0.5, wrap_width=None, keyframe_step=4
    )
    assert [(t.label, [k.frame for k in t.boxes], t.end) for t in tracks] == [
        ("face", [0, 4, 8, 9], 10),
        ("face", [20, 24], 25),
    ]
    assert tracks[0].boxes[0].box == (50.0, 50.0, 60.0, 60.0)


def test_preannotation_shows_the_larger_part_of_a_box_on_the_seam() -> None:
    settings = Settings()
    frames = {0: [("plate", 0.5, (990.0, 0.0, 1030.0, 10.0), None)]}
    (track,) = preannotation_tracks(
        _detections(frames), settings, conf=0.1, fps=30.0, proxy_factor=1.0, wrap_width=1000, keyframe_step=1
    )
    assert track.boxes[0].box == (0.0, 0.0, 30.0, 10.0)


def test_preannotation_drops_short_and_low_score_tracks() -> None:
    settings = Settings()
    frames: dict[int, list[tuple[str, float, tuple[float, float, float, float], str | None]]] = {
        f: [("face", 0.5, (100.0, 100.0, 120.0, 120.0), "face:1")] for f in range(5)
    }
    frames[10] = [("plate", 0.9, (500.0, 0.0, 520.0, 10.0), None)]  # one frame only
    frames[20] = [("plate", 0.2, (900.0, 0.0, 920.0, 10.0), None)]  # low score
    tracks = preannotation_tracks(
        _detections(frames),
        settings,
        conf=0.25,
        fps=30.0,
        proxy_factor=1.0,
        wrap_width=None,
        keyframe_step=15,
        min_frames=3,
    )
    assert [(t.label, len(t.boxes)) for t in tracks] == [("face", 2)]
