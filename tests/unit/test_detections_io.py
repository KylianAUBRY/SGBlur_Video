"""Tests of the detections.jsonl reader and writer."""

import json
from pathlib import Path

import pytest

from sgblur_video.core.detections_io import (
    DetectionRecord,
    DetectionsFormatError,
    DetectionsWriter,
    Footer,
    FrameRecord,
    Header,
    read_detections,
)


def _header() -> Header:
    return Header(
        created_at="2026-10-06T00:00:00Z",
        video={"width": 64, "height": 48},
        model={"name": "fake"},
        detection={},
        tracking={},
        software={},
    )


def _frame(index: int) -> FrameRecord:
    record = DetectionRecord.model_validate(
        {"class": "face", "score": 0.5, "box": (1, 2, 3, 4), "track_id": "face:1"}
    )
    return FrameRecord(index=index, pts=index * 100, time=index / 30, detections=[record])


def test_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "d.jsonl"
    with DetectionsWriter(path, _header()) as writer:
        for i in range(3):
            writer.write_frame(_frame(i))
        writer.close_with(Footer(frames=3, complete=True, elapsed_s=1.0))
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert lines[0]["schema"] == "sgblur-video/detections"
    assert lines[1]["detections"][0]["class"] == "face"
    loaded = read_detections(path)
    assert loaded.complete
    assert [f.index for f in loaded.frames] == [0, 1, 2]
    assert loaded.frames[2].detections[0].class_ == "face"


def test_truncated_file_is_incomplete(tmp_path: Path) -> None:
    path = tmp_path / "d.jsonl"
    with DetectionsWriter(path, _header()) as writer:
        writer.write_frame(_frame(0))
    loaded = read_detections(path)
    assert loaded.footer is None
    assert not loaded.complete


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("", "first line must be a header"),
        ('{"type": "frame", "index": 0}\n', "first line must be a header"),
        ('{"type": "header", "schema": "sgblur-video/detections", "version": 2}\n', "unsupported"),
        ("not json\n", "invalid JSON"),
    ],
)
def test_invalid_files(tmp_path: Path, content: str, message: str) -> None:
    path = tmp_path / "d.jsonl"
    path.write_text(content)
    with pytest.raises(DetectionsFormatError, match=message):
        read_detections(path)


def test_frames_out_of_order(tmp_path: Path) -> None:
    path = tmp_path / "d.jsonl"
    with DetectionsWriter(path, _header()) as writer:
        writer.write_frame(_frame(0))
        writer.write_frame(_frame(2))
    with pytest.raises(DetectionsFormatError, match="frame 2 found where 1 was expected"):
        read_detections(path)
