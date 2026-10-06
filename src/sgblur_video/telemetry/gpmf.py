"""Minimal GoPro GPMF (KLV) parser for GPS positions.

GPMF payloads are sequences of KLV entries: a 4-character key, a 1-byte type,
a 1-byte structure size and a 2-byte big-endian repeat count, followed by
``size × repeat`` bytes padded to a multiple of 4. Type NUL (zero byte) means the
payload is itself a list of KLV entries (``DEVC`` → ``STRM`` → data).
Reference: https://github.com/gopro/gpmf-parser (Apache-2.0 OR MIT).

Supported GPS streams:

* ``GPS5`` (HERO5–HERO11): latitude, longitude, altitude, 2D speed, 3D speed as
  ``int32`` divided by ``SCAL``, fix quality in ``GPSF``;
* ``GPS9`` (HERO11 and later): a complex structure described by ``TYPE``
  (latitude, longitude, altitude, speeds, days, seconds, DOP, fix).

Samples of one payload are spread evenly over the payload's duration.
"""

import struct
from collections.abc import Iterator
from dataclasses import dataclass

# GPMF type characters → struct format characters (big-endian).
_FORMATS = {
    "b": "b",
    "B": "B",
    "c": "c",
    "d": "d",
    "f": "f",
    "j": "q",
    "J": "Q",
    "l": "i",
    "L": "I",
    "s": "h",
    "S": "H",
}
_MIN_FIX = 2  # 2D fix or better


@dataclass(frozen=True)
class KLV:
    """One GPMF entry; ``children`` is set for nested entries (type NUL)."""

    key: str
    type: str
    size: int
    repeat: int
    data: bytes
    children: tuple[KLV, ...] = ()


@dataclass(frozen=True)
class GpsSample:
    """One GPS position at a time offset from the start of the video."""

    time: float
    lat: float
    lon: float
    alt: float


def parse(buffer: bytes) -> list[KLV]:
    """Parse a GPMF payload into a tree of KLV entries (malformed tails are ignored)."""
    entries = []
    offset = 0
    while offset + 8 <= len(buffer):
        key = buffer[offset : offset + 4].decode("latin-1")
        type_char = chr(buffer[offset + 4])
        size = buffer[offset + 5]
        repeat = struct.unpack_from(">H", buffer, offset + 6)[0]
        length = size * repeat
        start = offset + 8
        data = buffer[start : start + length]
        if len(data) < length or not key.isprintable():
            break
        children = tuple(parse(data)) if type_char == "\0" else ()
        entries.append(KLV(key, type_char, size, repeat, data, children))
        offset = start + ((length + 3) & ~3)
    return entries


def _values(entry: KLV, type_string: str | None = None) -> list[tuple[float | int, ...]]:
    """Decode the samples of an entry into tuples of numbers."""
    if entry.type == "?":
        if not type_string:
            return []
        layout = ">" + "".join(_FORMATS.get(c, "x") for c in type_string)
    elif entry.type in _FORMATS:
        count = entry.size // struct.calcsize(">" + _FORMATS[entry.type])
        layout = ">" + _FORMATS[entry.type] * count
    else:
        return []
    width = struct.calcsize(layout)
    if width == 0:
        return []
    return [
        struct.unpack_from(layout, entry.data, i * width)
        for i in range(min(entry.repeat, len(entry.data) // width))
    ]


def _scales(stream: dict[str, KLV], fields: int) -> list[float]:
    scal = stream.get("SCAL")
    if scal is None:
        return [1.0] * fields
    flat = [float(v) for row in _values(scal) for v in row] or [1.0]
    return flat * fields if len(flat) == 1 else flat


def _streams(entries: list[KLV]) -> Iterator[dict[str, KLV]]:
    for entry in entries:
        if entry.key == "STRM":
            yield {child.key: child for child in entry.children}
        elif entry.children:
            yield from _streams(list(entry.children))


def gps_samples(payload: bytes, start: float, duration: float) -> list[GpsSample]:
    """GPS positions of one GPMF payload with a usable fix.

    Args:
        payload: Raw bytes of one ``gpmd`` packet.
        start: Time of the packet, in seconds from the start of the video.
        duration: Duration of the packet, in seconds.

    Returns:
        Positions spread evenly over ``[start, start + duration)``.
    """
    samples: list[GpsSample] = []
    for stream in _streams(parse(payload)):
        if "GPS5" in stream:
            rows = _values(stream["GPS5"])
            scales = _scales(stream, 5)
            fix = _values(stream["GPSF"])[0][0] if "GPSF" in stream and _values(stream["GPSF"]) else 3
            if fix < _MIN_FIX:
                continue
            fields = [(r[0] / scales[0], r[1] / scales[1], r[2] / scales[2]) for r in rows]
        elif "GPS9" in stream:
            type_string = (
                stream["TYPE"].data.decode("latin-1").rstrip("\0") if "TYPE" in stream else "lllllllSS"
            )
            rows = _values(stream["GPS9"], type_string)
            scales = _scales(stream, len(type_string))
            fields = [
                (r[0] / scales[0], r[1] / scales[1], r[2] / scales[2])
                for r in rows
                if len(r) >= 9 and r[8] / scales[8] >= _MIN_FIX
            ]
        else:
            continue
        step = duration / max(1, len(fields))
        samples += [
            GpsSample(start + i * step, lat, lon, alt)
            for i, (lat, lon, alt) in enumerate(fields)
            if (lat, lon) != (0.0, 0.0)
        ]
    return samples
