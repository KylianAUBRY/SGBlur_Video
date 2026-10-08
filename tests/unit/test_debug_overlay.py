"""Debug overlay: one colour per class, each blurred region labelled with its detection score."""

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


def _detections(frames: Mapping[int, list[tuple[str, float, Box, str | None]]], count: int) -> Detections:
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
                ("face", 0.9, FACE, None),
                ("plate", 0.8, (700, 700, 780, 730), None),
                ("sign", 0.7, (100, 700, 160, 760), "signage:1"),
            ],
            10: [("face", 0.6, FACE, None)],
        },
        20,
    )
    plan = build_blur_plan(detections, Settings(), frame_size=(1000, 1000))
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


def test_only_frames_with_a_detection_are_drawn() -> None:
    _plan, overlay = _overlay()
    assert _draw(overlay, 0)[400:481, 400:481].any()
    assert _draw(overlay, 10)[400:481, 400:481].any()
    assert not _draw(overlay, 5).any()  # nothing carried between detections


def test_labels_show_the_detection_score() -> None:
    plan, _ = _overlay()
    assert [round(s.score, 2) for s in plan.shapes(10)] == [0.6]
