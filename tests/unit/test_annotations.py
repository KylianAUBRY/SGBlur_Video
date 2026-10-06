"""Tests of sign deduplication and Panoramax annotations."""

from collections.abc import Mapping

import pytest

from sgblur_video.config import Settings
from sgblur_video.core.detections_io import DetectionRecord, Detections, Footer, FrameRecord, Header
from sgblur_video.core.geometry import Box
from sgblur_video.semantics.annotations import (
    Metadata,
    build_annotations,
    display_shape,
    dump_metadata,
    find_sign_tracks,
    model_tag,
)

Det = tuple[str, float, Box, str | None]


def _detections(frames: Mapping[int, list[Det]], count: int = 60) -> Detections:
    header = Header(
        created_at="2026-10-06T00:00:00Z",
        video={"width": 1000, "height": 500},
        model={"name": "yolo26s", "version": "0.1.0"},
        detection={},
        tracking={"tracker": "tracktrack"},
        software={},
    )
    records = [
        FrameRecord(
            index=i,
            pts=i * 1000,
            time=i / 30,
            detections=[
                DetectionRecord.model_validate({"class": c, "score": s, "box": b, "track_id": t})
                for c, s, b, t in frames.get(i, [])
            ],
        )
        for i in range(count)
    ]
    return Detections(header, records, Footer(frames=count, complete=True, elapsed_s=0))


def test_one_annotation_per_physical_sign() -> None:
    frames: dict[int, list[Det]] = {}
    for i in range(0, 30):  # tracked sign growing as the camera approaches; best view at frame 25
        size = 20 + i
        score = 0.95 if i == 25 else 0.7
        frames.setdefault(i, []).append(("sign", score, (100, 100, 100 + size, 100 + size), "signage:1"))
    for i in range(0, 40, 2):  # flickering direction sign never tracked
        frames.setdefault(i, []).append(("direction", 0.65, (600, 50, 680, 90), None))
    for i in range(10, 13):  # short false positive
        frames.setdefault(i, []).append(("sign", 0.9, (900, 400, 920, 420), None))
    for i in range(40, 50):  # long but weak track
        frames.setdefault(i, []).append(("sign", 0.4, (300, 300, 340, 340), "signage:9"))
    frames.setdefault(5, []).append(("face", 0.9, (0, 0, 10, 10), "face:2"))  # not a sign
    tracks = find_sign_tracks(_detections(frames), Settings(), fps=30)
    assert len(tracks) == 2
    by_class = {t.cls: t for t in tracks}
    # Best view = highest score x area: frame 25 (0.95 x 45²) beats the larger frame 29 (0.7 x 49²).
    assert by_class["sign"].best.frame == 25
    assert len(by_class["sign"].observations) == 30
    # The flickering direction sign is one physical sign, not twenty.
    assert len(by_class["direction"].observations) == 20


def test_annotation_format_matches_sgblur() -> None:
    frames = {i: [("sign", 0.8, (100.4, 120.6, 160.2, 180.9), "signage:3")] for i in range(10)}
    detections = _detections(frames)
    settings = Settings()
    tracks = find_sign_tracks(detections, settings, fps=30)
    (annotation,) = build_annotations(tracks, detections, settings, frame_size=(1000, 500), rotation=0)
    assert annotation.shape == (100, 121, 160, 181)
    assert [(t.key, t.value) for t in annotation.semantics] == [
        ("osm|traffic_sign", "yes"),
        ("detection_model[osm|traffic_sign=yes]", "SGBlur-Video-yolo26s/0.1.0"),
        ("detection_confidence[osm|traffic_sign=yes]", "0.800"),
    ]
    assert annotation.video.first_frame == 0
    assert annotation.video.last_timestamp == pytest.approx(9 / 30, abs=1e-3)
    metadata = dump_metadata(Metadata(service_name=settings.api_name, annotations=[annotation]))
    assert metadata["service_name"] == "SGBlur-Video"
    assert metadata["annotations"][0]["video"]["class"] == "sign"
    assert "blurring_id" not in metadata  # absent rather than null, as in SGBlur
    assert "position" not in metadata["annotations"][0]["video"]
    assert all(isinstance(t["value"], str) for t in metadata["annotations"][0]["semantics"])


def test_model_tag_keeps_the_sgblur_prefix() -> None:
    assert model_tag(Settings(), {"name": "yolo26s", "version": "0.1.0"}).startswith("SGBlur-")


@pytest.mark.parametrize(
    ("rotation", "expected"),
    [(0, (10, 20, 50, 40)), (90, (20, 950, 40, 990)), (180, (950, 460, 990, 480)), (270, (460, 10, 480, 50))],
)
def test_shape_is_in_display_orientation(rotation: int, expected: tuple[int, int, int, int]) -> None:
    frames = {i: [("sign", 0.9, (10, 20, 50, 40), "signage:1")] for i in range(6)}
    (track,) = find_sign_tracks(_detections(frames), Settings(), fps=30)
    assert display_shape(track, frame_size=(1000, 500), rotation=rotation) == expected


def test_shape_is_clipped_to_the_frame() -> None:
    frames = {i: [("sign", 0.9, (-5, -3, 30, 30), "signage:1")] for i in range(6)}
    (track,) = find_sign_tracks(_detections(frames), Settings(), fps=30)
    assert display_shape(track, frame_size=(1000, 500), rotation=0) == (0, 0, 30, 30)
