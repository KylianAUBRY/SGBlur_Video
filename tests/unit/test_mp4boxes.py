"""Tests of the allow-listed MP4 box transplant (spherical metadata, rotation, udta)."""

import struct
from pathlib import Path

import av
import numpy as np
import pytest

from sgblur_video.config import Settings
from sgblur_video.core.mp4boxes import (
    SPHERICAL_V1_UUID,
    Box,
    Mp4BoxError,
    _read_moov,
    parse_boxes,
    transplant,
)
from sgblur_video.core.probe import probe

XMP = (
    b'<?xml version="1.0"?><rdf:SphericalVideo xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
    b'xmlns:GSpherical="http://ns.google.com/videos/1.0/spherical/"><GSpherical:Spherical>true</GSpherical:Spherical>'
    b"<GSpherical:Stitched>true</GSpherical:Stitched>"
    b"<GSpherical:StitchingSoftware>Spherical Metadata Tool</GSpherical:StitchingSoftware>"
    b"<GSpherical:ProjectionType>equirectangular"
    b"</GSpherical:ProjectionType></rdf:SphericalVideo>"
)
ROTATE_90 = struct.pack(">9i", 0, 0x10000, 0, -0x10000, 0, 0, 0, 0, 0x40000000)


def _video(path: Path, width: int = 320, height: int = 160) -> None:
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=30)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        for i in range(5):
            frame = av.VideoFrame.from_ndarray(np.full((height, width, 3), i * 40, np.uint8), format="rgb24")
            frame.pts = i
            container.mux(stream.encode(frame.reformat(format="yuv420p")))
        container.mux(stream.encode(None))


def _edit_moov(path: Path, edit: object) -> None:
    moov, start, _end = _read_moov(path)
    edit(moov)  # type: ignore[operator]
    with path.open("r+b") as handle:
        handle.seek(start)
        handle.write(moov.serialize())
        handle.truncate()


def _video_trak(moov: Box) -> Box:
    return next(c for c in moov.children or [] if c.type == b"trak")


def _make_source(path: Path) -> None:
    _video(path)

    def edit(moov: Box) -> None:
        trak = _video_trak(moov)
        assert trak.children is not None
        trak.children.append(Box(b"uuid", payload=SPHERICAL_V1_UUID + XMP))
        tkhd = trak.find(b"tkhd")
        assert tkhd is not None
        payload = bytearray(tkhd.payload)
        payload[40:76] = ROTATE_90
        tkhd.payload = bytes(payload)
        entry = trak.find(b"mdia").find(b"minf").find(b"stbl").find(b"stsd").children[0]  # type: ignore[union-attr]
        entry.children.append(Box(b"st3d", payload=b"\0\0\0\0\0"))  # type: ignore[union-attr]
        udta = moov.find(b"udta") or Box(b"udta", children=[])
        if udta not in (moov.children or []):
            moov.children.append(udta)  # type: ignore[union-attr]
        udta.children += [  # type: ignore[operator]
            Box(b"FIRM", payload=b"HD8.01.02.51.00"),
            Box(b"thmb", payload=b"\xff\xd8 unblurred thumbnail"),
            Box(b"uuid", payload=b"\x11" * 16 + b"unknown"),
        ]

    _edit_moov(path, edit)


def test_round_trip_keeps_the_file_identical(tmp_path: Path) -> None:
    path = tmp_path / "v.mp4"
    _video(path)
    before = path.read_bytes()
    _edit_moov(path, lambda _moov: None)
    assert path.read_bytes() == before
    assert parse_boxes(before)[0].type == b"ftyp"


def test_transplant_restores_spherical_rotation_and_safe_udta(tmp_path: Path) -> None:
    source, target = tmp_path / "source.mp4", tmp_path / "target.mp4"
    _make_source(source)
    _video(target)
    settings = Settings()
    assert probe(source, settings).projection_source == "spherical-metadata"
    assert probe(target, settings).projection_source == "aspect-ratio-heuristic"

    report = transplant(source, target)
    assert report.spherical_v1
    assert report.spherical_v2 == ["st3d"]
    assert report.matrix
    assert report.udta == ["FIRM"]  # no thumbnail, no unknown uuid

    info = probe(target, settings)
    assert info.projection_source == "spherical-metadata"
    assert abs(info.rotation) == 90
    with av.open(str(target)) as container:
        assert sum(1 for _ in container.decode(video=0)) == 5  # samples untouched
    data = target.read_bytes()
    assert b"unblurred thumbnail" not in data
    assert b"HD8.01.02.51.00" in data


def test_target_must_end_with_moov(tmp_path: Path) -> None:
    source, target = tmp_path / "source.mp4", tmp_path / "target.mp4"
    _video(source)
    _video(target)
    with target.open("ab") as handle:
        handle.write(struct.pack(">I4s", 12, b"free") + b"\0" * 4)
    with pytest.raises(Mp4BoxError, match="not the last box"):
        transplant(source, target)
