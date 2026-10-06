"""Read and write "CVAT for video 1.1" XML (pre-annotations and ground-truth imports).

CVAT (https://github.com/cvat-ai/cvat, MIT) annotates videos in *track* mode:
a track holds boxes on key frames, CVAT interpolates linearly in between, and a
box with ``outside="1"`` ends the visibility of the object. A track without a
final ``outside`` box stays visible until the last frame. Exports usually list
every frame of every track; this reader also accepts key frames only.

Only rectangle tracks labelled ``face`` or ``plate`` are imported, with their
mutable ``readable`` checkbox attribute.
"""

import logging
import xml.etree.ElementTree as ET  # local files exported by the maintainer's own CVAT
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from sgblur_video.bench.dataset import DatasetError, GtBox, GtClass, GtTrack
from sgblur_video.core.geometry import Box, area, clip, lerp, scale

logger = logging.getLogger(__name__)

LABELS = ("face", "plate")
READABLE = "readable"


@dataclass(frozen=True)
class KeyBox:
    """A box of a pre-annotation track (annotation-proxy pixels)."""

    frame: int
    box: Box


@dataclass(frozen=True)
class PreTrack:
    """A pre-annotation track: key boxes, visible until ``end`` (exclusive) or the last frame."""

    label: str
    boxes: tuple[KeyBox, ...]
    end: int | None


def labels_json() -> list[dict[str, object]]:
    """Labels for CVAT's "Raw" label editor (face and plate rectangles with a ``readable`` checkbox)."""
    return [
        {
            "name": name,
            "type": "rectangle",
            "attributes": [
                {
                    "name": READABLE,
                    "mutable": True,
                    "input_type": "checkbox",
                    "default_value": "false",
                    "values": ["false"],
                }
            ],
        }
        for name in LABELS
    ]


def _label_xml(name: str) -> ET.Element:
    label = ET.Element("label")
    ET.SubElement(label, "name").text = name
    ET.SubElement(label, "type").text = "rectangle"
    attributes = ET.SubElement(label, "attributes")
    attribute = ET.SubElement(attributes, "attribute")
    ET.SubElement(attribute, "name").text = READABLE
    ET.SubElement(attribute, "mutable").text = "True"
    ET.SubElement(attribute, "input_type").text = "checkbox"
    ET.SubElement(attribute, "default_value").text = "false"
    ET.SubElement(attribute, "values").text = "false"
    return label


def _box_xml(parent: ET.Element, frame: int, box: Box, *, outside: bool) -> None:
    element = ET.SubElement(
        parent,
        "box",
        {
            "frame": str(frame),
            "keyframe": "1",
            "outside": "1" if outside else "0",
            "occluded": "0",
            "xtl": f"{box[0]:.2f}",
            "ytl": f"{box[1]:.2f}",
            "xbr": f"{box[2]:.2f}",
            "ybr": f"{box[3]:.2f}",
            "z_order": "0",
        },
    )
    ET.SubElement(element, "attribute", {"name": READABLE}).text = "false"


def write_preannotation(
    path: Path, tracks: Sequence[PreTrack], *, frames: int, width: int, height: int
) -> None:
    """Write a CVAT for video 1.1 file that CVAT can import into a task made from the proxy.

    Args:
        path: XML file to write.
        tracks: Pre-annotation tracks in proxy pixels.
        frames: Number of frames of the proxy.
        width: Proxy width.
        height: Proxy height.
    """
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    task = ET.SubElement(ET.SubElement(root, "meta"), "task")
    ET.SubElement(task, "size").text = str(frames)
    ET.SubElement(task, "mode").text = "interpolation"
    ET.SubElement(task, "start_frame").text = "0"
    ET.SubElement(task, "stop_frame").text = str(frames - 1)
    labels = ET.SubElement(task, "labels")
    for name in LABELS:
        labels.append(_label_xml(name))
    original = ET.SubElement(task, "original_size")
    ET.SubElement(original, "width").text = str(width)
    ET.SubElement(original, "height").text = str(height)
    for number, track in enumerate(tracks):
        element = ET.SubElement(root, "track", {"id": str(number), "label": track.label, "source": "auto"})
        for key in track.boxes:
            _box_xml(element, key.frame, key.box, outside=False)
        if track.end is not None and track.end < frames:
            _box_xml(element, track.end, track.boxes[-1].box, outside=True)
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


@dataclass(frozen=True)
class _Shape:
    frame: int
    box: Box
    outside: bool
    readable: bool


def _meta_int(meta: ET.Element | None, path: str) -> int | None:
    if meta is None:
        return None
    for owner in ("task", "job"):
        node = meta.find(f"{owner}/{path}")
        if node is not None and node.text and node.text.strip().isdigit():
            return int(node.text)
    return None


def _expand(shapes: list[_Shape], frames: int) -> list[tuple[int, Box, bool]]:
    """Every visible frame of a track, interpolating between key frames like CVAT."""
    by_frame = {s.frame: s for s in shapes}
    ordered = [by_frame[f] for f in sorted(by_frame)]
    visible: list[tuple[int, Box, bool]] = []
    for current, following in zip(ordered, [*ordered[1:], None], strict=False):
        if current.outside:
            continue
        stop = following.frame if following is not None else frames
        for frame in range(current.frame, min(stop, frames)):
            box = current.box
            if following is not None and not following.outside and frame > current.frame:
                box = lerp(current.box, following.box, (frame - current.frame) / (stop - current.frame))
            visible.append((frame, box, current.readable))
    return visible


def read_cvat_video(
    path: Path, *, frames: int, width: int, height: int, proxy_width: int, proxy_height: int
) -> tuple[list[GtTrack], Counter[str]]:
    """Import a CVAT for video 1.1 export as ground-truth tracks in full-resolution pixels.

    Args:
        path: The exported XML.
        frames: Number of frames of the clip.
        width: Clip width (boxes are scaled from the proxy to the clip).
        height: Clip height.
        proxy_width: Width of the proxy annotated in CVAT.
        proxy_height: Height of the proxy.

    Returns:
        The tracks, and a count of what was skipped (other labels, non-rectangle shapes).

    Raises:
        DatasetError: If the file is not a CVAT video export of this clip.
    """
    try:
        root = ET.parse(path).getroot()  # noqa: S314 - see the import above
    except ET.ParseError as exc:
        msg = f"{path.name}: not an XML file ({exc})"
        raise DatasetError(msg) from exc
    if root.tag != "annotations":
        msg = f"{path.name}: not a CVAT export (root <{root.tag}>)"
        raise DatasetError(msg)
    if root.find("image") is not None:
        msg = f"{path.name}: this is a 'CVAT for images' export; export as 'CVAT for video 1.1'"
        raise DatasetError(msg)
    meta = root.find("meta")
    size = _meta_int(meta, "size")
    if size is not None and size != frames:
        msg = f"{path.name}: {size} frames annotated, the clip has {frames}"
        raise DatasetError(msg)
    original = (_meta_int(meta, "original_size/width"), _meta_int(meta, "original_size/height"))
    if all(original) and original != (proxy_width, proxy_height):
        msg = (
            f"{path.name}: annotated a {original[0]}x{original[1]} video, "
            f"the proxy is {proxy_width}x{proxy_height}"
        )
        raise DatasetError(msg)
    sx, sy = width / proxy_width, height / proxy_height

    tracks: list[GtTrack] = []
    skipped: Counter[str] = Counter()
    for element in root.iter("track"):
        label = element.get("label", "")
        if label not in LABELS:
            skipped[f"label:{label}"] += 1
            continue
        shapes = []
        for child in element:
            if child.tag != "box":
                skipped[f"shape:{child.tag}"] += 1
                continue
            readable = any(
                a.get("name") == READABLE and (a.text or "").strip().lower() == "true"
                for a in child.iter("attribute")
            )
            corners = (
                child.get("xtl", "0"),
                child.get("ytl", "0"),
                child.get("xbr", "0"),
                child.get("ybr", "0"),
            )
            box = scale((float(corners[0]), float(corners[1]), float(corners[2]), float(corners[3])), sx, sy)
            shapes.append(_Shape(int(child.get("frame", 0)), box, child.get("outside") == "1", readable))
        boxes = [
            GtBox(frame=frame, box=clip(box, width, height), readable=readable)
            for frame, box, readable in _expand(shapes, frames)
            if area(clip(box, width, height)) > 0
        ]
        if boxes:
            track_id = f"{label}:{element.get('id', len(tracks))}"
            tracks.append(GtTrack(id=track_id, cls=cast(GtClass, label), boxes=boxes))
    if skipped:
        logger.warning("%s: skipped %s", path.name, dict(skipped))
    return tracks, skipped
