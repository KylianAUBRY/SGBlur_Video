"""Turn ``detections.jsonl`` into a blur plan: what to blur on every frame.

Every frame is blurred on its own detections, like SGBlur blurs a picture
(``docs/design/pipeline.md`` §3): each face or plate detection of the frame
with a score ≥ ``CONF_DETECT`` is blurred with a rectangle exactly on its box.
YOLO already drops lower scores; the plan checks again, so a
``detections.jsonl`` produced with a lower threshold blurs the same.

Nothing is carried from one frame to the next: no tracking, interpolation or
padding, so an object the detector misses on a frame stays visible on that
frame, as on a picture blurred by SGBlur.

Like SGBlur, boxes smaller than ``MIN_BLUR_SIZE`` pixels on a side are not
blurred (nothing identifiable fits in them).

On 360° video, boxes may cross the 0°/360° seam (``x2 > width``); the renderer
blurs their part on each side.
"""

from collections import Counter
from dataclasses import dataclass, field

from sgblur_video.config import ClassAction, Settings
from sgblur_video.core.detections_io import Detections
from sgblur_video.core.geometry import Box, area, clip, height, width
from sgblur_video.video360.wrap import normalize

#: Boxes smaller than this on either side are not blurred (SGBlur: 12 px).
MIN_BLUR_SIZE = 12


@dataclass(frozen=True)
class BlurShape:
    """One rectangle to blur on one frame.

    Attributes:
        box: Detected box in coded-frame pixels (on 360° video, ``x2`` may exceed the width).
        cls: Class name.
        score: Detection score.
    """

    box: Box
    cls: str
    score: float


@dataclass
class BlurPlan:
    """Regions to blur, per frame index.

    Attributes:
        frame_count: Number of frames covered by the plan.
        frames: Shapes per frame index (frames without shapes are absent).
        stats: Counters for the job metadata.
        wrap_width: Frame width of a 360° video (shapes may cross the 0°/360° seam), else ``None``.
    """

    frame_count: int
    frames: dict[int, list[BlurShape]] = field(default_factory=dict)
    stats: dict[str, int] = field(default_factory=dict)
    wrap_width: int | None = None

    def shapes(self, index: int) -> list[BlurShape]:
        """Shapes to blur on frame ``index`` (empty list if none)."""
        return self.frames.get(index, [])


def build_blur_plan(
    detections: Detections,
    settings: Settings,
    *,
    frame_size: tuple[int, int],
    wrap_width: int | None = None,
) -> BlurPlan:
    """Compute the blur plan of a video from its detections, frame by frame.

    Args:
        detections: Content of ``detections.jsonl``.
        settings: Class policy and ``CONF_DETECT``.
        frame_size: Coded frame ``(width, height)``.
        wrap_width: Frame width of a 360° video, ``None`` for flat video.

    Returns:
        The blur plan, with statistics.
    """
    blur_classes = settings.classes_with(ClassAction.BLUR)
    frame_width, frame_height = frame_size
    plan = BlurPlan(frame_count=len(detections.frames), wrap_width=wrap_width)
    stats: Counter[str] = Counter()
    for frame in detections.frames:
        shapes = []
        for det in frame.detections:
            if det.class_ not in blur_classes:
                continue
            if det.score < settings.conf_detect:
                stats["below_conf"] += 1
                continue
            box = normalize(det.box, wrap_width) if wrap_width else det.box
            if width(box) < MIN_BLUR_SIZE or height(box) < MIN_BLUR_SIZE:
                stats["too_small"] += 1
                continue
            if wrap_width:
                # Horizontal position wraps around: only the vertical extent can leave the frame.
                if box[3] <= 0 or box[1] >= frame_height:
                    continue
            elif area(clip(box, frame_width, frame_height)) <= 0:
                continue
            shapes.append(BlurShape(box, det.class_, det.score))
            stats[f"boxes_{det.class_}"] += 1
        if shapes:
            plan.frames[frame.index] = shapes
    stats["frames_with_blur"] = len(plan.frames)
    plan.stats = dict(stats)
    return plan
