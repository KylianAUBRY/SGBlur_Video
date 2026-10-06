"""Allow-listed MP4 box transplant: what FFmpeg does not write back (``docs/adr/0008``).

After rendering, the output file is ``[ftyp][free][mdat][moov]``: its ``moov``
box is the last one, so it can be rewritten without moving any sample (chunk
offsets point into ``mdat``, before it). From the **original** file, and only
from an explicit allow-list, this module copies:

* Spherical Video **V1** (``uuid`` box ``ffcc8263-f855-4a93-8814-587a02521fdd``
  holding XMP, child of the video ``trak``) and **V2** (``sv3d`` and ``st3d``
  boxes inside the video sample entry): without them, players show a 360°
  video as a flat 2:1 picture;
* the video ``tkhd`` display matrix (rotation of phone videos);
* safe ``moov/udta`` children (GoPro camera settings and identifiers,
  QuickTime location and date strings).

Nothing else is copied: no thumbnails or previews (they show unblurred
people), no unknown ``uuid`` boxes, nothing from the original ``mdat``.
"""

import logging
import struct
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

SPHERICAL_V1_UUID = bytes.fromhex("ffcc8263f8554a938814587a02521fdd")
SPHERICAL_V2_BOXES = frozenset({b"sv3d", b"st3d"})
#: ``moov/udta`` children copied from the original (camera metadata without pictures).
UDTA_ALLOW_LIST = frozenset(
    {
        b"FIRM", b"LENS", b"CAME", b"SETT", b"MUID", b"HMMT", b"BCID", b"GUMI", b"GPMF",  # GoPro
        b"\xa9xyz", b"\xa9day", b"\xa9mak", b"\xa9mod",  # QuickTime location, date, make, model
    }
)  # fmt: skip
_CONTAINERS = frozenset({b"moov", b"trak", b"mdia", b"minf", b"stbl", b"udta", b"edts", b"dinf"})
# Visual sample entries: 8-byte box header + 78 bytes of fields before child boxes.
_VISUAL_ENTRIES = frozenset({b"avc1", b"avc3", b"hvc1", b"hev1", b"mp4v", b"av01", b"vp09"})
_VISUAL_ENTRY_FIELDS = 78
_STSD_FIELDS = 8  # version/flags + entry count
_TKHD_MATRIX_OFFSET = {0: 40, 1: 52}  # offset of the matrix inside the tkhd payload, per version
_MATRIX_BYTES = 36
_IDENTITY = struct.pack(">9i", 0x10000, 0, 0, 0, 0x10000, 0, 0, 0, 0x40000000)


class Mp4BoxError(ValueError):
    """The file does not have the layout this editor supports."""


@dataclass
class Box:
    """An MP4 box: either raw ``payload`` bytes or ``prefix`` fields followed by ``children``."""

    type: bytes
    payload: bytes = b""
    prefix: bytes = b""
    children: list[Box] | None = None

    def serialize(self) -> bytes:
        """Box bytes, sizes recomputed from the children."""
        body = (
            self.prefix + b"".join(c.serialize() for c in self.children)
            if self.children is not None
            else self.payload
        )
        size = 8 + len(body)
        if size > 0xFFFFFFFF:
            return struct.pack(">I4sQ", 1, self.type, size + 8) + body
        return struct.pack(">I4s", size, self.type) + body

    def find(self, box_type: bytes) -> Box | None:
        """First direct child of a type."""
        return next((c for c in self.children or [] if c.type == box_type), None)


@dataclass
class _Layout:
    boxes: list[tuple[bytes, int, int]] = field(default_factory=list)  # type, start, end


def _read_header(data: bytes, offset: int, end: int) -> tuple[bytes, int, int]:
    size, box_type = struct.unpack_from(">I4s", data, offset)
    header = 8
    if size == 1:
        size = struct.unpack_from(">Q", data, offset + 8)[0]
        header = 16
    elif size == 0:
        size = end - offset
    if size < header or offset + size > end:
        raise Mp4BoxError(f"invalid box size at {offset}")
    return box_type, header, size


def parse_boxes(data: bytes, start: int = 0, end: int | None = None) -> list[Box]:
    """Parse boxes, descending into the containers this module edits."""
    end = len(data) if end is None else end
    boxes = []
    offset = start
    while offset + 8 <= end:
        box_type, header, size = _read_header(data, offset, end)
        body_start, body_end = offset + header, offset + size
        if box_type in _CONTAINERS:
            boxes.append(Box(box_type, children=parse_boxes(data, body_start, body_end)))
        elif box_type == b"stsd":
            boxes.append(
                Box(
                    box_type,
                    prefix=data[body_start : body_start + _STSD_FIELDS],
                    children=parse_boxes(data, body_start + _STSD_FIELDS, body_end),
                )
            )
        elif box_type in _VISUAL_ENTRIES:
            fields_end = body_start + _VISUAL_ENTRY_FIELDS
            boxes.append(
                Box(
                    box_type,
                    prefix=data[body_start:fields_end],
                    children=parse_boxes(data, fields_end, body_end),
                )
            )
        else:
            boxes.append(Box(box_type, payload=data[body_start:body_end]))
        offset = body_end
    return boxes


def _top_level(path: Path) -> list[tuple[bytes, int, int]]:
    """``(type, start, end)`` of the top-level boxes of a file, without reading ``mdat``."""
    layout = []
    size_total = path.stat().st_size
    with path.open("rb") as handle:
        offset = 0
        while offset + 8 <= size_total:
            handle.seek(offset)
            header = handle.read(16)
            box_type, _header_size, size = _read_header(
                header + b"\0" * (16 - len(header)), 0, size_total - offset
            )
            layout.append((box_type, offset, offset + size))
            offset += size
    return layout


def _read_moov(path: Path) -> tuple[Box, int, int]:
    for box_type, start, end in _top_level(path):
        if box_type == b"moov":
            with path.open("rb") as handle:
                handle.seek(start)
                data = handle.read(end - start)
            return parse_boxes(data)[0], start, end
    raise Mp4BoxError(f"{path.name}: no moov box")


def _handler(trak: Box) -> bytes:
    mdia = trak.find(b"mdia")
    hdlr = mdia.find(b"hdlr") if mdia else None
    return hdlr.payload[8:12] if hdlr and len(hdlr.payload) >= 12 else b""


def _video_trak(moov: Box) -> Box | None:
    return next((c for c in moov.children or [] if c.type == b"trak" and _handler(c) == b"vide"), None)


def _sample_entry(trak: Box) -> Box | None:
    node: Box | None = trak
    for box_type in (b"mdia", b"minf", b"stbl", b"stsd"):
        node = node.find(box_type) if node else None
    return next((c for c in node.children or [] if c.type in _VISUAL_ENTRIES), None) if node else None


def _matrix(tkhd: Box) -> bytes | None:
    offset = _TKHD_MATRIX_OFFSET.get(tkhd.payload[0]) if tkhd.payload else None
    return tkhd.payload[offset : offset + _MATRIX_BYTES] if offset is not None else None


@dataclass
class TransplantReport:
    """What was copied into the output."""

    spherical_v1: bool = False
    spherical_v2: list[str] = field(default_factory=list)
    matrix: bool = False
    udta: list[str] = field(default_factory=list)


def transplant(source: Path, target: Path) -> TransplantReport:
    """Copy allow-listed metadata boxes from ``source`` into ``target`` (rewritten in place).

    Args:
        source: Original video.
        target: Rendered video whose ``moov`` is the last top-level box.

    Returns:
        What was copied.

    Raises:
        Mp4BoxError: If a file cannot be parsed or ``target``'s ``moov`` is not last.
    """
    report = TransplantReport()
    source_moov, _, _ = _read_moov(source)
    target_moov, moov_start, moov_end = _read_moov(target)
    if moov_end != target.stat().st_size:
        raise Mp4BoxError(f"{target.name}: moov is not the last box")
    source_video, target_video = _video_trak(source_moov), _video_trak(target_moov)
    if source_video is None or target_video is None:
        return report

    v1 = next(
        (c for c in source_video.children or [] if c.type == b"uuid" and c.payload[:16] == SPHERICAL_V1_UUID),
        None,
    )
    if v1 is not None and target_video.children is not None:
        target_video.children = [
            c
            for c in target_video.children
            if not (c.type == b"uuid" and c.payload[:16] == SPHERICAL_V1_UUID)
        ]
        target_video.children.append(v1)
        report.spherical_v1 = True

    source_entry, target_entry = _sample_entry(source_video), _sample_entry(target_video)
    if source_entry is not None and target_entry is not None and target_entry.children is not None:
        for child in source_entry.children or []:
            if child.type in SPHERICAL_V2_BOXES and target_entry.find(child.type) is None:
                target_entry.children.append(child)
                report.spherical_v2.append(child.type.decode())

    source_tkhd, target_tkhd = source_video.find(b"tkhd"), target_video.find(b"tkhd")
    if source_tkhd is not None and target_tkhd is not None:
        matrix = _matrix(source_tkhd)
        offset = _TKHD_MATRIX_OFFSET.get(target_tkhd.payload[0]) if target_tkhd.payload else None
        if matrix and matrix != _IDENTITY and offset is not None:
            payload = bytearray(target_tkhd.payload)
            payload[offset : offset + _MATRIX_BYTES] = matrix
            target_tkhd.payload = bytes(payload)
            report.matrix = True

    source_udta = source_moov.find(b"udta")
    if source_udta is not None and target_moov.children is not None:
        kept = [c for c in source_udta.children or [] if c.type in UDTA_ALLOW_LIST]
        if kept:
            target_udta = target_moov.find(b"udta")
            if target_udta is None:
                target_udta = Box(b"udta", children=[])
                target_moov.children.append(target_udta)
            existing = {c.type for c in target_udta.children or []}
            for child in kept:
                if child.type not in existing and target_udta.children is not None:
                    target_udta.children.append(child)
                    report.udta.append(child.type.decode("latin-1"))

    with target.open("r+b") as handle:
        handle.seek(moov_start)
        handle.write(target_moov.serialize())
        handle.truncate()
    logger.info("metadata transplanted: %s", report)
    return report
