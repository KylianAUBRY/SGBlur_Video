from pathlib import Path

from sgblur_video.bench.cache import _finished, detection_key, tracking_key, video_key
from sgblur_video.config import Settings
from sgblur_video.core.detections_io import DetectionsWriter, Footer, FrameRecord, Header

HEADER = Header.model_validate(
    {"created_at": "now", "video": {}, "model": {}, "detection": {}, "tracking": {}, "software": {}}
)


def test_only_finished_files_are_reused(tmp_path: Path) -> None:
    path = tmp_path / "d.jsonl"
    assert _finished(path) is None
    with DetectionsWriter(path, HEADER) as writer:
        writer.write_frame(FrameRecord(index=0, pts=0, time=0.0))
    assert _finished(path) is None  # interrupted run: no footer
    path.write_text(path.read_text(encoding="utf-8") + '{"type": "fra', encoding="utf-8")
    assert _finished(path) is None  # cut in the middle of a line
    with DetectionsWriter(path, HEADER) as writer:
        writer.close_with(Footer(frames=0, complete=True, elapsed_s=0.0))
    finished = _finished(path)
    assert finished is not None
    assert finished.complete


def test_cache_keys(tmp_path: Path, repo_root: Path) -> None:
    tracker = repo_root / "configs" / "trackers" / "tracktrack-recall.yaml"
    settings = Settings(tracker_config=tracker)
    key = detection_key(settings, "abc", None)
    # Post-processing settings do not change the detections; detection settings do.
    assert detection_key(settings.model_copy(update={"conf_blur": 0.9}), "abc", None) == key
    assert detection_key(settings.model_copy(update={"conf_detect": 0.2}), "abc", None) != key
    assert detection_key(settings, "def", None) != key
    assert detection_key(settings, "abc", 100) != key
    other = settings.model_copy(update={"tracker_config": tracker.with_name("bytetrack-recall.yaml")})
    assert tracking_key(other) != tracking_key(settings)
    video = tmp_path / "v.mp4"
    video.write_bytes(b"1")
    first = video_key(video)
    video.write_bytes(b"12")
    assert video_key(video) != first
