"""Media integration tests: timestamps, frame count and audio survive pass 2."""

from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

from sgblur_video.config import Settings
from sgblur_video.core.postprocess import BlurPlan, BlurShape
from sgblur_video.core.probe import UnsupportedVideoError, probe
from sgblur_video.core.render import RenderError, render
from sgblur_video.core.trim import FrameRange, trim_video

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
    plan.frames[3] = [BlurShape((10, 10, 100, 100), "plate", 0.9)]
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


def _audio_times(path: Path) -> list[float]:
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        return [
            float(p.pts * stream.time_base) for p in container.demux(stream) if p.size and p.pts is not None
        ]


def test_frame_range_is_cut_with_its_audio(tmp_path: Path) -> None:
    source, cut = tmp_path / "vfr.mp4", tmp_path / "cut.mp4"
    _make_video(source)
    settings = Settings(encoder="libx264")
    info = probe(source, settings)
    assert trim_video(info, settings, FrameRange(5, 15), cut) == 10
    times = _video_pts(source)
    # Frames 5..14 of the source, shifted to start at 0 (VFR gaps kept).
    assert _video_pts(cut) == [round(t - times[5], 3) for t in times[5:15]]
    audio = _audio_times(cut)
    assert audio
    assert min(audio) >= 0
    # Audio covers the whole range and stops with it (one AAC packet is 21 ms; the muxer adds the
    # encoder priming delay of a few ms to the read-back times).
    assert max(audio) >= times[14] - times[5] - 0.03
    assert max(audio) < times[15] - times[5] + 0.03
    assert max(audio) < max(_audio_times(source)) - times[5] - 0.2  # the rest of the audio is not kept
    assert len(_audio_packets(cut)) < len(_audio_packets(source))
    with pytest.raises(RenderError, match="no frame"):
        trim_video(info, settings, FrameRange(len(VFR_PTS) + 5), tmp_path / "empty.mp4")


def test_frame_range_counts() -> None:
    assert FrameRange().is_whole
    assert FrameRange(10, 30).frames(100) == 20
    assert FrameRange(90, 130).frames(100) == 10
    assert FrameRange(10).frames(100) == 90
    assert FrameRange(10).frames(None) is None
    assert FrameRange(10, 12).frames(None) == 2
