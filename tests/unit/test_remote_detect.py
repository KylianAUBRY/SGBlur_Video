"""Tests of the worker's client for the remote Detect API (mocked HTTP)."""

import json
from fractions import Fraction
from pathlib import Path

import httpx
import pytest

from sgblur_video.config import Settings
from sgblur_video.core.probe import VideoInfo
from sgblur_video.jobs.runner import RemoteDetectError, remote_detect

HEADER = {"type": "header", "schema": "sgblur-video/detections", "version": 1}
FRAME = {"type": "frame", "index": 0, "pts": 0, "time": 0.0, "detections": []}
FOOTER = {"type": "footer", "frames": 1, "complete": True, "elapsed_s": 0.1}


def _info(path: Path) -> VideoInfo:
    path.write_bytes(b"video")
    return VideoInfo(
        path=path, container="mp4", width=64, height=48, rotation=0, time_base=Fraction(1, 30), start_pts=0,
        avg_frame_rate=Fraction(30), frame_count=1, duration_s=0.03, codec="h264", codec_tag="avc1",
        pix_fmt="yuv420p", bit_rate=None, color={}, projection="flat", projection_source="default",
    )  # fmt: skip


def _ndjson(*lines: dict) -> bytes:
    return b"".join(json.dumps(line, separators=(", ", ": ")).encode() + b"\n" for line in lines)


def _client(responses: list[httpx.Response], seen: list[httpx.Request]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return responses.pop(0)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_streamed_detections_are_saved_with_progress(tmp_path: Path) -> None:
    seen: list[httpx.Request] = []
    progress: list[tuple[int, int | None]] = []
    responses = [
        httpx.Response(503, headers={"Retry-After": "0"}),
        httpx.Response(200, content=_ndjson(HEADER, FRAME, FOOTER)),
    ]
    output = tmp_path / "detections.jsonl"
    settings = Settings(detect_url="http://detect:8001")
    remote_detect(
        _info(tmp_path / "in.mp4"),
        output,
        settings,
        lambda d, t: progress.append((d, t)),
        client=_client(responses, seen),
    )
    assert [json.loads(line)["type"] for line in output.read_text().splitlines()] == [
        "header",
        "frame",
        "footer",
    ]
    assert progress == [(1, 1)]
    assert len(seen) == 2
    assert str(seen[0].url) == "http://detect:8001/detect/"
    assert b'name="video"' in seen[1].read()


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx.Response(500, content=b"boom"), "returned 500"),
        (httpx.Response(200, content=_ndjson(HEADER, FRAME)), "without a footer"),
    ],
)
def test_remote_failures(tmp_path: Path, response: httpx.Response, message: str) -> None:
    settings = Settings(detect_url="http://detect:8001")
    with pytest.raises(RemoteDetectError, match=message):
        remote_detect(
            _info(tmp_path / "in.mp4"),
            tmp_path / "d.jsonl",
            settings,
            lambda *_: None,
            client=_client([response], []),
        )
