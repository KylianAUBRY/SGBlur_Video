"""Debug overlay: one colour per class, solid outlines for detections, dashed ones for deduced regions."""

from collections.abc import Mapping

import numpy as np
import numpy.typing as npt

from sgblur_video.config import Settings
from sgblur_video.core.debug import CLASS_COLOURS, DebugOverlay
from sgblur_video.core.detections_io import DetectionRecord, Detections, Footer, FrameRecord, Header
from sgblur_video.core.geometry import Box
from sgblur_video.core.postprocess import BlurPlan, build_blur_plan

FACE: Box = (400, 400, 480, 480)
SIGNS = frozenset({"sign", "direction"})


def _detections(frames: Mapping[int, list[tuple[str, float, Box, str]]], count: int) -> Detections:
    header = Header(
        created_at="2026-10-07T00:00:00Z", video={}, model={}, detection={}, tracking={}, software={}
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
        for i in range(count)
    ]
    return Detections(header, records, Footer(frames=count, complete=True, elapsed_s=0))


def _overlay() -> tuple[BlurPlan, DebugOverlay]:
    detections = _detections(
        {
            0: [
                ("face", 0.9, FACE, "face:1"),
                ("plate", 0.8, (700, 700, 780, 730), "plate:1"),
                ("sign", 0.7, (100, 700, 160, 760), "signage:1"),
            ],
            10: [("face", 0.6, FACE, "face:1")],
        },
        20,
    )
    plan = build_blur_plan(
        detections, Settings(blur_temporal_padding_frames=0), frame_size=(1000, 1000), fps=30
    )
    return plan, DebugOverlay(plan, detections, SIGNS)


def _draw(overlay: DebugOverlay, index: int) -> npt.NDArray[np.uint8]:
    image = np.zeros((1000, 1000, 3), dtype=np.uint8)
    overlay.draw(index, image, 1.0)
    return image


def test_each_class_has_its_colour() -> None:
    _plan, overlay = _overlay()
    image = _draw(overlay, 0)
    for cls in ("face", "plate", "sign"):
        assert (image == CLASS_COLOURS[cls]).all(axis=2).any(), cls


def test_detected_regions_show_the_detection_score() -> None:
    plan, overlay = _overlay()
    assert overlay._score(0, plan.shapes(0)[0]) == 0.9
    face = next(s for s in plan.shapes(10) if s.cls == "face")
    assert overlay._score(10, face) == 0.6


def test_regions_without_detection_are_dashed() -> None:
    plan, overlay = _overlay()
    assert next(s for s in plan.shapes(5) if s.cls == "face").source == "interpolated"

    def outline_pixels(index: int) -> int:
        # Lower half of the face ellipse: no label there, no other object.
        return int(_draw(overlay, index)[440:540, 320:560].any(axis=2).sum())

    solid, dashed = outline_pixels(0), outline_pixels(5)
    assert solid > 0
    assert 0.3 * solid < dashed < 0.75 * solid
