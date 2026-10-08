"""Tests of keep=1 encryption and of the streaming multipart upload."""

import asyncio
import io
import json
import os
import time
import zipfile
from collections.abc import AsyncIterator
from pathlib import Path

import av
import numpy as np
import pytest
from cryptography.exceptions import InvalidTag

from sgblur_video.api.upload import UploadError, receive_file
from sgblur_video.core.postprocess import BlurShape
from sgblur_video.privacy.keep import KeepRecorder, archive_path, decrypt, encrypt, purge_expired


def test_encryption_needs_the_secret_and_the_blurring_id() -> None:
    blob = encrypt(b"original pixels", secret="s3cret", blurring_id="id-1")
    assert b"original pixels" not in blob
    assert decrypt(blob, secret="s3cret", blurring_id="id-1") == b"original pixels"
    with pytest.raises(InvalidTag):
        decrypt(blob, secret="other", blurring_id="id-1")
    with pytest.raises(InvalidTag):
        decrypt(blob, secret="s3cret", blurring_id="id-2")


def test_recorder_keeps_only_low_confidence_detections(tmp_path: Path) -> None:
    frame = av.VideoFrame.from_ndarray(np.full((40, 60, 3), 200, np.uint8), format="rgb24")
    recorder = KeepRecorder(max_confidence=0.5)
    shapes = [BlurShape((5, 5, 25, 15), "plate", 0.3), BlurShape((30, 5, 50, 30), "face", 0.9)]
    recorder(7, frame, shapes)
    assert recorder.regions == 1
    path = recorder.save(tmp_path, secret="k", blurring_id="bid")
    assert path == archive_path(tmp_path, "bid")
    assert path is not None
    assert "bid" not in path.name  # the file name does not reveal the id
    archive = zipfile.ZipFile(io.BytesIO(decrypt(path.read_bytes(), secret="k", blurring_id="bid")))
    manifest = json.loads(archive.read("manifest.json"))
    assert [(r["class"], r["score"]) for r in manifest["regions"]] == [("plate", 0.3)]
    assert manifest["regions"][0]["frame"] == 7


def test_nothing_is_written_without_low_confidence_regions(tmp_path: Path) -> None:
    assert KeepRecorder(0.5).save(tmp_path, secret="k", blurring_id="bid") is None
    assert list(tmp_path.iterdir()) == []


def test_purge_expired(tmp_path: Path) -> None:
    old, fresh = tmp_path / "old.bin", tmp_path / "fresh.bin"
    old.write_bytes(b"x")
    fresh.write_bytes(b"x")
    os.utime(old, (time.time() - 3 * 3600, time.time() - 3 * 3600))
    assert purge_expired(tmp_path, ttl_hours=2) == 1
    assert [p.name for p in tmp_path.iterdir()] == ["fresh.bin"]
    assert purge_expired(tmp_path / "missing", 1) == 0


def _body(fields: list[tuple[str, str | None, bytes]], boundary: str = "XyZ") -> bytes:
    out = b""
    for name, filename, content in fields:
        disposition = f'form-data; name="{name}"' + (f'; filename="{filename}"' if filename else "")
        out += f"--{boundary}\r\nContent-Disposition: {disposition}\r\n\r\n".encode() + content + b"\r\n"
    return out + f"--{boundary}--\r\n".encode()


async def _chunks(data: bytes, size: int = 7) -> AsyncIterator[bytes]:
    for start in range(0, len(data), size):
        yield data[start : start + size]


def _receive(tmp_path: Path, body: bytes, *, max_bytes: int = 1000, content_type: str | None = None):
    return asyncio.run(
        receive_file(
            content_type or "multipart/form-data; boundary=XyZ",
            _chunks(body),
            field_name="video",
            destination=tmp_path / "upload",
            max_bytes=max_bytes,
        )
    )


def test_upload_writes_only_the_video_field(tmp_path: Path) -> None:
    body = _body([("note", None, b"hello"), ("video", "C:\\Users\\someone\\My Trip.MOV", b"0123456789" * 20)])
    received = _receive(tmp_path, body)
    assert received.size == 200
    assert received.suffix == ".mov"
    assert (tmp_path / "upload").read_bytes() == b"0123456789" * 20


@pytest.mark.parametrize(
    ("body", "content_type", "status"),
    [
        (_body([("video", "a.mp4", b"x" * 2000)]), None, 413),
        (_body([("other", "a.mp4", b"x" * 10)]), None, 422),
        (_body([("video", "a.mp4", b"")]), None, 422),
        (b"plain", "text/plain", 415),
    ],
)
def test_upload_errors_leave_no_file(
    tmp_path: Path, body: bytes, content_type: str | None, status: int
) -> None:
    with pytest.raises(UploadError) as excinfo:
        _receive(tmp_path, body, content_type=content_type)
    assert excinfo.value.status == status
    assert not (tmp_path / "upload").exists()
