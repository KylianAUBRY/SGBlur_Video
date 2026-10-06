"""Tests of the GPMF parser and GPS track interpolation (synthetic payloads)."""

import struct

import pytest

from sgblur_video.telemetry.gpmf import gps_samples, parse
from sgblur_video.telemetry.gps import GpsTrack


def klv(key: str, type_char: str, size: int, values: bytes | list[bytes]) -> bytes:
    """Encode one KLV entry (nested when ``values`` is a list of entries)."""
    data = b"".join(values) if isinstance(values, list) else values
    repeat = len(data) // size if size else 0
    header = (
        key.encode()
        + (b"\0" if type_char == "\0" else type_char.encode())
        + bytes([size])
        + struct.pack(">H", repeat)
    )
    return header + data + b"\0" * ((4 - len(data) % 4) % 4)


def nested(key: str, children: list[bytes]) -> bytes:
    data = b"".join(children)
    return key.encode() + b"\0" + bytes([1]) + struct.pack(">H", len(data)) + data


def gps5_payload(fix: int, rows: list[tuple[int, int, int, int, int]]) -> bytes:
    stream = nested(
        "STRM",
        [
            klv("STNM", "c", 1, b"GPS (Lat., Long., Alt., 2D speed, 3D speed)"),
            klv("GPSF", "L", 4, struct.pack(">L", fix)),
            klv("SCAL", "l", 4, struct.pack(">5l", 10_000_000, 10_000_000, 1000, 1000, 100)),
            klv("GPS5", "l", 20, b"".join(struct.pack(">5l", *row) for row in rows)),
        ],
    )
    return nested("DEVC", [klv("DVID", "L", 4, struct.pack(">L", 1)), stream])


def test_klv_tree() -> None:
    entries = parse(gps5_payload(3, [(1, 2, 3, 4, 5)]))
    assert [e.key for e in entries] == ["DEVC"]
    assert [c.key for c in entries[0].children] == ["DVID", "STRM"]


def test_gps5_samples_are_scaled_and_spread_over_the_packet() -> None:
    rows = [(451_234_567, 57_654_321, 212_500, 0, 0), (451_234_667, 57_654_421, 213_000, 0, 0)]
    samples = gps_samples(gps5_payload(3, rows), start=10.0, duration=1.0)
    assert [s.time for s in samples] == [10.0, 10.5]
    assert samples[0].lat == pytest.approx(45.1234567)
    assert samples[0].lon == pytest.approx(5.7654321)
    assert samples[1].alt == pytest.approx(213.0)


def test_samples_without_fix_are_ignored() -> None:
    assert gps_samples(gps5_payload(0, [(451_234_567, 57_654_321, 0, 0, 0)]), 0.0, 1.0) == []


def test_gps9_complex_samples() -> None:
    type_string = b"lllllllSS"
    row = struct.pack(">7l2H", 451_234_567, 57_654_321, 212_500, 0, 0, 9000, 43_200_000, 150, 3)
    no_fix = struct.pack(">7l2H", 451_234_567, 57_654_321, 212_500, 0, 0, 9000, 43_200_000, 150, 0)
    stream = nested(
        "STRM",
        [
            klv("TYPE", "c", 1, type_string),
            klv("SCAL", "l", 4, struct.pack(">9l", 10_000_000, 10_000_000, 1000, 1000, 100, 1, 1000, 100, 1)),
            klv("GPS9", "?", 32, row + no_fix),
        ],
    )
    samples = gps_samples(nested("DEVC", [stream]), 0.0, 1.0)
    assert len(samples) == 1
    assert samples[0].lat == pytest.approx(45.1234567)


def test_track_interpolation_and_limits() -> None:
    track = GpsTrack(
        gps_samples(
            gps5_payload(3, [(450_000_000, 50_000_000, 0, 0, 0), (450_000_100, 50_000_100, 1000, 0, 0)]),
            0.0,
            2.0,
        ),
        "gpmf",
    )
    assert len(track) == 2
    middle = track.position_at(0.5)
    assert middle is not None
    assert middle.lat == pytest.approx(45.000005)
    assert middle.alt == pytest.approx(0.5)
    assert track.position_at(-2.0) is None
    assert track.position_at(10.0) is None


def test_malformed_payloads_do_not_crash() -> None:
    assert parse(b"\x00\x01garbage") == []
    assert gps_samples(b"DEVC\0\x01\xff\xff", 0.0, 1.0) == []
