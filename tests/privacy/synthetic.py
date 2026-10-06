"""Synthetic videos with known object positions and a scripted fake detector.

This is the privacy oracle of ``docs/design/testing-strategy.md`` (option A).
Objects are high-frequency checkerboard patches on a smooth background, so a
blurred object is easy to measure: the texture energy inside its true box
collapses. The fake detector reports the objects with the defects real
detectors have (late onset, early release, holes, flicker, low scores,
too-small and jittered boxes), and the real pipeline must still blur every
frame of every object.
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import av
import numpy as np
import numpy.typing as npt

from sgblur_video.core.detect import Detection, DetectionPass
from sgblur_video.core.geometry import Box

WIDTH, HEIGHT, FRAMES, FPS = 640, 360, 60, 30


@dataclass
class SyntheticObject:
    """An object with a known position on every frame where it is visible.

    Attributes:
        cls: Class name reported by the fake detector.
        size: ``(width, height)`` in pixels.
        position: Function frame index → top-left corner.
        visible: Frames where the object is drawn.
        detected: Frames where the fake detector reports it.
        score: Function frame index → reported score.
    """

    cls: str
    size: tuple[int, int]
    position: Callable[[int], tuple[float, float]]
    visible: range
    detected: set[int]
    score: Callable[[int], float] = field(default=lambda _frame: 0.6)

    def box(self, frame: int) -> Box:
        """True box on ``frame``."""
        x, y = self.position(frame)
        return (x, y, x + self.size[0], y + self.size[1])


def scenario() -> list[SyntheticObject]:
    """Objects of the standard privacy scenario (60 frames, 640×360)."""
    walking_face = SyntheticObject(
        "face",
        (48, 48),
        lambda f: (40 + 4 * f, 60),
        visible=range(5, 45),
        # Late onset (5–7), a hole of 6 frames (17–22), early release (42–44), one isolated miss (10).
        detected=set(range(8, 42)) - {10, 17, 18, 19, 20, 21, 22},
        score=lambda f: 0.12 if f % 4 == 0 else 0.55,  # some detections below CONF_BLUR
    )
    flickering_plate = SyntheticObject(
        "plate",
        (64, 24),
        lambda f: (400 - 0.5 * f, 250),
        visible=range(FRAMES),
        detected=set(range(0, FRAMES, 2)),  # every other frame, never tracked reliably
        score=lambda _f: 0.3,
    )
    fast_small_face = SyntheticObject(
        "face",
        (24, 24),
        lambda f: (100 + 10 * f, 200),
        visible=range(10, 31),
        detected={12, 15, 18},  # sparse detections of a fast object
        score=lambda _f: 0.5,
    )
    sign = SyntheticObject(
        "sign",
        (40, 40),
        lambda _f: (560, 40),
        visible=range(FRAMES),
        detected=set(range(FRAMES)),
        score=lambda _f: 0.9,
    )
    flickering_sign = SyntheticObject(
        "sign",
        (36, 36),
        # Away from the extrapolated paths of the faces: padding after a face leaves
        # follows its trajectory and may blur whatever lies there (over-blur by design).
        lambda _f: (470, 110),
        visible=range(FRAMES),
        detected=set(range(0, FRAMES, 3)),  # one physical sign seen every third frame
        score=lambda _f: 0.7,
    )
    false_sign = SyntheticObject(
        "sign",
        (20, 20),
        lambda _f: (20, 320),
        visible=range(50, 54),
        detected={50, 51, 52},  # too short to be a real sign (SIGN_MIN_TRACK_LENGTH)
        score=lambda _f: 0.9,
    )
    return [walking_face, flickering_plate, fast_small_face, sign, flickering_sign, false_sign]


def _background() -> npt.NDArray[np.uint8]:
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
    gray = (60 + 80 * xx / WIDTH + 40 * yy / HEIGHT).astype(np.uint8)
    return np.dstack([gray, gray, gray])


def _texture(width: int, height: int) -> npt.NDArray[np.uint8]:
    yy, xx = np.mgrid[0:height, 0:width]
    pattern = (((yy // 4 + xx // 4) % 2) * 200 + 25).astype(np.uint8)
    return np.dstack([pattern, pattern, pattern])


def render_frame(objects: Sequence[SyntheticObject], frame: int) -> npt.NDArray[np.uint8]:
    """RGB image of ``frame``."""
    image = _background()
    for obj in objects:
        if frame not in obj.visible:
            continue
        x1, y1, x2, y2 = (round(v) for v in obj.box(frame))
        image[max(0, y1) : y2, max(0, x1) : x2] = _texture(x2 - max(0, x1), y2 - max(0, y1))
    return image


def write_video(
    path: Path, objects: Sequence[SyntheticObject], *, codec: str = "libx264", pix_fmt: str = "yuv420p"
) -> None:
    """Encode the scenario at high quality (the texture must survive compression)."""
    with av.open(str(path), "w") as container:
        stream = container.add_stream(codec, rate=FPS, options={"crf": "12", "preset": "fast"})
        stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, pix_fmt
        for frame_index in range(FRAMES):
            frame = av.VideoFrame.from_ndarray(render_frame(objects, frame_index), format="rgb24").reformat(
                format=pix_fmt
            )
            frame.pts = frame_index
            container.mux(stream.encode(frame))
        container.mux(stream.encode(None))


class FakeDetector:
    """Reports the scenario's objects with scripted defects (implements ``FrameDetector``).

    Reported boxes are 10 % smaller than the truth and jittered by up to 2 px,
    and each detection is also reported by a second pass (duplicates to merge).
    """

    def __init__(self, objects: Sequence[SyntheticObject]) -> None:
        self._objects = objects
        self._rng = np.random.default_rng(1234)

    @property
    def class_names(self) -> tuple[str, ...]:
        """Classes of the fake model."""
        return ("direction", "sign", "plate", "face")

    def detect(
        self, image: npt.NDArray[np.uint8], plan: Sequence[DetectionPass], frame_index: int
    ) -> list[Detection]:
        """Scripted detections for ``frame_index``."""
        detections = []
        for obj in self._objects:
            if frame_index not in obj.detected or frame_index not in obj.visible:
                continue
            x1, y1, x2, y2 = obj.box(frame_index)
            shrink_x, shrink_y = (x2 - x1) * 0.05, (y2 - y1) * 0.05
            jitter = self._rng.uniform(-2, 2, 4)
            box = (
                x1 + shrink_x + jitter[0],
                y1 + shrink_y + jitter[1],
                x2 - shrink_x + jitter[2],
                y2 - shrink_y + jitter[3],
            )
            score = obj.score(frame_index)
            detections.append(Detection(obj.cls, score, box, [plan[0].id]))
            detections.append(Detection(obj.cls, score * 0.9, box, [plan[-1].id]))
        return detections


def texture_energy(luma: npt.NDArray[np.uint8], box: Box) -> float:
    """Mean absolute horizontal gradient inside a box (high for the checkerboard, ~1 for the background)."""
    x1, y1, x2, y2 = (
        max(0, math.floor(box[0])),
        max(0, math.floor(box[1])),
        math.ceil(box[2]),
        math.ceil(box[3]),
    )
    patch = luma[y1:y2, x1:x2].astype(np.float64)
    if patch.shape[1] < 2 or patch.shape[0] < 1:
        return 0.0
    return float(np.abs(np.diff(patch, axis=1)).mean())
