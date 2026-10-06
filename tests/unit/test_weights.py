"""Tests of model download and verification (no network: mocked transport)."""

import hashlib
from pathlib import Path

import httpx
import pytest

from sgblur_video.models import ModelEntry, WeightsError, ensure_weights

PAYLOAD = b"fake weights" * 1000


def _entry(sha256: str | None = None) -> ModelEntry:
    return ModelEntry(
        name="fake",
        family="fake",
        version="0.1.0",
        file="fake.pt",
        url="https://example.org/-/raw/abc/fake.pt",  # type: ignore[arg-type]
        sha256=sha256 or hashlib.sha256(PAYLOAD).hexdigest(),
        size_bytes=len(PAYLOAD),
        classes=("face", "plate"),
        train_imgsz=640,
        min_memory_gib=1,
        licence="test",
        source="https://example.org",  # type: ignore[arg-type]
    )


def _client(calls: list[str], payload: bytes = PAYLOAD, status: int = 200) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(status, content=payload)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_download_verify_and_reuse(tmp_path: Path) -> None:
    calls: list[str] = []
    path = ensure_weights(_entry(), tmp_path, client=_client(calls))
    assert path.read_bytes() == PAYLOAD
    assert ensure_weights(_entry(), tmp_path, client=_client(calls)) == path
    assert len(calls) == 1  # second call reused the verified file


def test_hash_mismatch_is_rejected_and_cleaned(tmp_path: Path) -> None:
    with pytest.raises(WeightsError, match="verification failed"):
        ensure_weights(_entry(sha256="0" * 64), tmp_path, client=_client([]))
    assert list(tmp_path.iterdir()) == []


def test_corrupted_local_file_is_replaced(tmp_path: Path) -> None:
    (tmp_path / "fake.pt").write_bytes(b"corrupted")
    calls: list[str] = []
    path = ensure_weights(_entry(), tmp_path, client=_client(calls))
    assert path.read_bytes() == PAYLOAD
    assert calls


def test_http_error(tmp_path: Path) -> None:
    with pytest.raises(WeightsError, match="cannot download"):
        ensure_weights(_entry(), tmp_path, client=_client([], status=404))
