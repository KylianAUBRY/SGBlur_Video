"""Multi-scale detection: detection plan, YOLO inference and cross-pass merge.

The plan reproduces SGBlur's logic for video frames (``docs/design/pipeline.md``
§2.2): a full-frame pass at ``imgsz=1024`` for large, close objects, a pass at
``imgsz=2048``, and for very large frames (8K 360°) tiles at native
resolution. Boxes from every pass are merged per class group: blurred classes
keep the **union** of duplicates (never shrink a blur area), annotated classes
keep the best-scoring box.

The detector is behind the :class:`FrameDetector` protocol so that tests can
replace YOLO with a scripted fake detector (privacy oracle).
"""

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np
import numpy.typing as npt

from sgblur_video.config import ClassAction, DetectProfile
from sgblur_video.core.geometry import Box, iomin, iou, translate, union_box

logger = logging.getLogger(__name__)

#: Group of every ``annotate`` class (signs and direction signs share one tracker).
SIGNAGE_GROUP = "signage"


@dataclass(frozen=True)
class DetectionPass:
    """One inference of the detection plan.

    Attributes:
        id: Short identifier written in ``detections.jsonl`` (``g1024``, ``tL``…).
        kind: ``global`` (full frame) or ``tile`` (crop at native resolution).
        imgsz: Ultralytics inference size.
        region: ``(x1, y1, x2, y2)`` integer crop in frame pixels for tiles, ``None`` for global passes.
    """

    id: str
    kind: Literal["global", "tile"]
    imgsz: int
    region: tuple[int, int, int, int] | None = None


@dataclass
class Detection:
    """A detection after cross-pass merge.

    Attributes:
        cls: Class name.
        score: Best confidence among merged boxes.
        box: Merged box in coded-frame pixels.
        passes: Passes that produced the merged boxes.
        track_id: ``"<group>:<id>"`` once the tracker has matched it, else ``None`` (orphan).
    """

    cls: str
    score: float
    box: Box
    passes: list[str] = field(default_factory=list)
    track_id: str | None = None


class FrameDetector(Protocol):
    """Anything that turns a frame into detections (YOLO, or a fake detector in tests)."""

    @property
    def class_names(self) -> tuple[str, ...]:
        """Class names provided by the detector."""
        ...

    def detect(
        self, image: npt.NDArray[np.uint8], plan: Sequence[DetectionPass], frame_index: int
    ) -> list[Detection]:
        """Run every pass of ``plan`` on a BGR image and return raw (unmerged) detections.

        Args:
            image: Upright BGR frame at full resolution.
            plan: Detection passes.
            frame_index: Index of the frame (used by scripted detectors).

        Returns:
            One detection per raw box, each with a single pass id.
        """
        ...


def _round32(value: float) -> int:
    """Round up to a multiple of 32 (YOLO stride)."""
    return int(math.ceil(value / 32) * 32)


def build_plan(
    width: int,
    height: int,
    *,
    projection: Literal["flat", "equirectangular"],
    profile: DetectProfile,
    tile_trigger_width: int,
) -> list[DetectionPass]:
    """Build the detection plan for a video.

    Args:
        width: Frame width (upright orientation).
        height: Frame height (upright orientation).
        projection: ``flat`` or ``equirectangular``.
        profile: ``fast``, ``standard`` or ``thorough``.
        tile_trigger_width: Long side from which tiles are added.

    Returns:
        The passes, global ones first.

    Example:
        >>> profile = DetectProfile.STANDARD
        >>> plan = build_plan(1920, 1080, projection="flat", profile=profile, tile_trigger_width=5760)
        >>> " ".join(p.id for p in plan)
        'g1024 g1920'
    """
    long_side = max(width, height)
    plan = [DetectionPass("g1024", "global", min(1024, _round32(long_side)))]
    if long_side > 1024:
        size = min(2048, _round32(long_side))
        plan.append(DetectionPass(f"g{size}", "global", size))
    if profile is DetectProfile.FAST or long_side < tile_trigger_width:
        return plan
    if projection == "equirectangular":
        # SGBlur layout: two halves of the middle band (full height with `thorough`),
        # overlapping by 1/16 of the width around the middle split line.
        top, bottom = (0, height) if profile is DetectProfile.THOROUGH else (height // 4, height * 3 // 4)
        overlap = width // 16
        half = width // 2
        for tile_id, (x1, x2) in (("tL", (0, half + overlap)), ("tR", (half - overlap, width))):
            imgsz = min(4096, _round32(max(x2 - x1, bottom - top)))
            plan.append(DetectionPass(tile_id, "tile", imgsz, (x1, top, x2, bottom)))
        return plan
    # Flat frames: grid of ~3840 px tiles with 10 % overlap.
    tile = 3840
    cols, rows = math.ceil(width / tile), math.ceil(height / tile)
    tile_w, tile_h = math.ceil(width / cols), math.ceil(height / rows)
    margin_x, margin_y = tile_w // 10, tile_h // 10
    for row in range(rows):
        for col in range(cols):
            x1, y1 = max(0, col * tile_w - margin_x), max(0, row * tile_h - margin_y)
            x2, y2 = min(width, (col + 1) * tile_w + margin_x), min(height, (row + 1) * tile_h + margin_y)
            imgsz = min(4096, _round32(max(x2 - x1, y2 - y1)))
            plan.append(DetectionPass(f"t{row}{col}", "tile", imgsz, (x1, y1, x2, y2)))
    return plan


def class_groups(policy: Mapping[str, ClassAction]) -> dict[str, str]:
    """Map each class name to its tracking group.

    Each ``blur`` class is its own group (a face track must never continue as a
    plate); every ``annotate`` class shares the ``signage`` group (the model can
    hesitate between ``sign`` and ``direction`` on the same object).

    Example:
        >>> class_groups({"face": ClassAction.BLUR, "sign": ClassAction.ANNOTATE})
        {'face': 'face', 'sign': 'signage'}
    """
    return {name: (name if action is ClassAction.BLUR else SIGNAGE_GROUP) for name, action in policy.items()}


def merge_detections(
    raw: Sequence[Detection],
    policy: Mapping[str, ClassAction],
    *,
    iou_threshold: float = 0.5,
    iomin_threshold: float = 0.8,
) -> list[Detection]:
    """Merge duplicate boxes from several passes, per class group.

    Boxes are processed by decreasing score; a box joins the first cluster of
    the same group it overlaps (IoU ≥ ``iou_threshold`` or IoMin ≥
    ``iomin_threshold``). A cluster keeps the best score and class; its box is
    the union of its members for ``blur`` classes (privacy) and the best
    member's box for ``annotate`` classes.

    Args:
        raw: Detections of every pass for one frame.
        policy: Class policy (classes absent from it are dropped).
        iou_threshold: IoU above which two boxes are duplicates.
        iomin_threshold: Intersection over the smaller area above which two boxes are duplicates.

    Returns:
        Merged detections, best score first.
    """
    groups = class_groups(policy)
    clusters: list[tuple[str, Detection, list[Box]]] = []
    for det in sorted((d for d in raw if d.cls in groups), key=lambda d: d.score, reverse=True):
        group = groups[det.cls]
        for cluster_group, best, members in clusters:
            if cluster_group == group and any(
                iou(det.box, m) >= iou_threshold or iomin(det.box, m) >= iomin_threshold for m in members
            ):
                members.append(det.box)
                best.passes.extend(p for p in det.passes if p not in best.passes)
                break
        else:
            clusters.append((group, Detection(det.cls, det.score, det.box, list(det.passes)), [det.box]))
    merged = []
    for _group, best, members in clusters:
        if policy[best.cls] is ClassAction.BLUR:
            best.box = union_box(members)
        merged.append(best)
    return merged


class YoloDetector:
    """Ultralytics YOLO behind the :class:`FrameDetector` protocol.

    Args:
        weights: Path of the checkpoint.
        device: Ultralytics device string (``cpu``, ``mps``, ``cuda:0``…).
        half: Run in FP16 (CUDA only).
        conf: Minimum score kept (``CONF_DETECT``).
        classes: Class names to detect; others are filtered by the model.
    """

    def __init__(self, weights: str, *, device: str, half: bool, conf: float, classes: Sequence[str]) -> None:
        from ultralytics import YOLO

        self._model = YOLO(weights, task="detect")
        self._device = device
        self._half = half
        self._conf = conf
        names: dict[int, str] = dict(self._model.names)
        self._names = tuple(names[i] for i in sorted(names))
        self._class_ids = [i for i, name in names.items() if name in set(classes)]

    @property
    def class_names(self) -> tuple[str, ...]:
        """Class names of the checkpoint, in model order."""
        return self._names

    def _predict(self, images: list[npt.NDArray[np.uint8]], imgsz: int) -> list[list[tuple[Box, float, int]]]:
        """Run one batched inference and return ``(box, score, class_id)`` per image."""
        kwargs: dict[str, object] = {
            "imgsz": imgsz,
            "conf": self._conf,
            "device": self._device,
            "classes": self._class_ids,
            "verbose": False,
        }
        if self._half:
            kwargs["quantize"] = 16
        results = self._model.predict(images, **kwargs)
        out = []
        for result in results:
            boxes = result.boxes
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            out.append(
                [
                    ((float(b[0]), float(b[1]), float(b[2]), float(b[3])), float(c), int(k))
                    for b, c, k in zip(xyxy, confs, cls_ids, strict=True)
                ]
            )
        return out

    def detect(
        self, image: npt.NDArray[np.uint8], plan: Sequence[DetectionPass], frame_index: int
    ) -> list[Detection]:
        """Run every pass of the plan on one upright BGR frame.

        Tiles of the same size are batched in one inference call.
        """
        detections: list[Detection] = []
        tiles_by_size: dict[int, list[DetectionPass]] = {}
        for detection_pass in plan:
            if detection_pass.kind == "global":
                for box, score, cls_id in self._predict([image], detection_pass.imgsz)[0]:
                    detections.append(Detection(self._names[cls_id], score, box, [detection_pass.id]))
            else:
                tiles_by_size.setdefault(detection_pass.imgsz, []).append(detection_pass)
        for imgsz, tiles in tiles_by_size.items():
            crops = []
            for tile in tiles:
                assert tile.region is not None  # noqa: S101 - tiles always have a region
                x1, y1, x2, y2 = tile.region
                crops.append(np.ascontiguousarray(image[y1:y2, x1:x2]))
            for tile, results in zip(tiles, self._predict(crops, imgsz), strict=True):
                assert tile.region is not None  # noqa: S101
                for box, score, cls_id in results:
                    shifted = translate(box, tile.region[0], tile.region[1])
                    detections.append(Detection(self._names[cls_id], score, shifted, [tile.id]))
        return detections


def unrotate_box(box: Box, rotation: int, coded_width: int, coded_height: int) -> Box:
    """Map a box from the upright (display) image back to coded-frame pixels.

    Args:
        box: Box in the upright image produced by :func:`sgblur_video.core.decode.rotate_upright`.
        rotation: Display rotation in degrees counter-clockwise (multiple of 90).
        coded_width: Width of the coded frame.
        coded_height: Height of the coded frame.

    Returns:
        The box in coded-frame coordinates.
    """
    turns = (round(rotation / 90) % 4) if rotation else 0
    x1, y1, x2, y2 = box
    if turns == 1:
        return (coded_width - y2, x1, coded_width - y1, x2)
    if turns == 2:
        return (coded_width - x2, coded_height - y2, coded_width - x1, coded_height - y1)
    if turns == 3:
        return (y1, coded_height - x2, y2, coded_height - x1)
    return box


def rotate_box(box: Box, rotation: int, coded_width: int, coded_height: int) -> Box:
    """Map a box from coded-frame pixels to the upright (display) image (inverse of :func:`unrotate_box`)."""
    turns = (round(rotation / 90) % 4) if rotation else 0
    x1, y1, x2, y2 = box
    if turns == 1:
        return (y1, coded_width - x2, y2, coded_width - x1)
    if turns == 2:
        return (coded_width - x2, coded_height - y2, coded_width - x1, coded_height - y1)
    if turns == 3:
        return (coded_height - y2, x1, coded_height - y1, x2)
    return box
