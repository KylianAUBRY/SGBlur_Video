"""Media integration tests: timestamps, frame count and audio survive pass 2."""

from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

from sgblur_video.config import Settings
from sgblur_video.core.postprocess import BlurPlan, BlurShape
from sgblur_video.core.probe import UnsupportedVideoError, probe
from sgblur_video.core.render import render

pytestmark = pytest.mark.integration

# Variable frame rate: irregular gaps between frames (milliseconds).
VFR_PTS = [0, 33, 66, 100, 150, 183, 250, 283, 316, 400, 433, 466, 500, 533, 600, 633, 700, 733, 766, 800]


def _make_video(path: Path, *, with_audio: bool = True, pts_list: list[int] = VFR_PTS) -> None:
    with av.open(str(path), "w") as container:
        video = container.add_stream("libx264", rate=30, options={"crf": "18"})
        video.width, video.height, video.pix_fmt = 320, 240, "yuv420p"
        video.time_base = Fraction(1, 1000)
        video.codec_context.time_base = Fraction(1, 1000)
        audio = container.add_stream("aac", rate=48000) if with_audio else None
        rng = np.random.default_rng(0)
        for pts in pts_list:
            image = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24").reformat(format="yuv420p")
            frame.pts, frame.time_base = pts, Fraction(1, 1000)
            container.mux(video.encode(frame))
        container.mux(video.encode(None))
        if audio is not None:
            samples = (np.sin(np.arange(48000) * 2 * np.pi * 440 / 48000) * 0.2).astype(np.float32)
            for start in range(0, 48000, 1024):
                chunk = samples[start : start + 1024].reshape(1, -1)
                frame = av.AudioFrame.from_ndarray(chunk, format="fltp", layout="mono")
                frame.sample_rate, frame.pts = 48000, start
                container.mux(audio.encode(frame))
            container.mux(audio.encode(None))


def _video_pts(path: Path) -> list[float]:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        return [round(float(f.pts * stream.time_base), 3) for f in container.decode(stream)]


def _audio_packets(path: Path) -> list[bytes]:
    with av.open(str(path)) as container:
        return [bytes(p) for p in container.demux(audio=0) if p.size]


def test_timestamps_frames_and_audio_are_preserved(tmp_path: Path) -> None:
    source, output = tmp_path / "vfr.mp4", tmp_path / "out.mp4"
    _make_video(source)
    settings = Settings(encoder="libx264")
    info = probe(source, settings)
    plan = BlurPlan(frame_count=len(VFR_PTS))
    plan.frames[3] = [BlurShape("rect", (10, 10, 100, 100), "plate", "detected", "plate:1")]
    stats = render(info, plan, output, settings)
    assert stats.frames == len(VFR_PTS)
    assert stats.blurred_frames == 1
    assert _video_pts(output) == _video_pts(source)  # VFR timestamps kept exactly
    assert _audio_packets(output) == _audio_packets(source)  # audio copied bit for bit
    assert len(stats.copied_streams) == 1
    assert stats.dropped_streams == []


def test_probe_rejects_unsupported_inputs(tmp_path: Path) -> None:
    text = tmp_path / "notes.txt"
    text.write_text("hello")
    with pytest.raises(UnsupportedVideoError) as excinfo:
        probe(text, Settings())
    assert excinfo.value.code == "unsupported_media_type"
    insv = tmp_path / "VID.insv"
    insv.write_bytes(b"\x00" * 64)
    with pytest.raises(UnsupportedVideoError) as excinfo:
        probe(insv, Settings())
    assert excinfo.value.code == "unsupported_projection"


def test_video_too_long_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "v.mp4"
    _make_video(source, with_audio=False, pts_list=list(range(0, 2000, 100)))  # 2 seconds
    with pytest.raises(UnsupportedVideoError) as excinfo:
        probe(source, Settings(max_video_duration_s=1))
    assert excinfo.value.code == "video_too_long"
