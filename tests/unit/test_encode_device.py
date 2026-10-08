"""Tests of encoder selection and device helpers."""

from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import av
import pytest

from sgblur_video.config import Settings
from sgblur_video.core.device import available_memory_gib, resolve_device, use_half
from sgblur_video.core.encode import choose_encoder
from sgblur_video.core.probe import VideoInfo


def _info(codec: str = "h264", pix_fmt: str = "yuv420p", bit_rate: int | None = 4_000_000) -> VideoInfo:
    return VideoInfo(
        path=Path("in.mp4"),
        container="mov,mp4",
        width=320,
        height=240,
        rotation=0,
        time_base=Fraction(1, 30000),
        start_pts=0,
        avg_frame_rate=Fraction(30),
        frame_count=10,
        duration_s=0.33,
        codec=codec,
        codec_tag="",
        pix_fmt=pix_fmt,
        bit_rate=bit_rate,
        color={"color_range": 1},
        projection="flat",
        projection_source="default",
    )


def test_explicit_software_encoder_keeps_family_and_rate() -> None:
    choice = choose_encoder(_info(), Settings(encoder="libx264", encode_bitrate_factor=0.5))
    assert (choice.codec, choice.family, choice.pix_fmt, choice.codec_tag) == (
        "libx264",
        "h264",
        "yuv420p",
        "avc1",
    )
    assert choice.bit_rate == 2_000_000
    assert choice.options["crf"] == "20"
    assert choice.options["maxrate"] == str(3_000_000)


def test_hevc_ten_bit_and_full_range() -> None:
    choice = choose_encoder(_info("hevc", "yuv420p10le", None), Settings(encoder="libx265"))
    assert (choice.family, choice.pix_fmt, choice.codec_tag) == ("hevc", "yuv420p10le", "hvc1")
    assert "log-level=error" in choice.options["x265-params"]
    full_range = choose_encoder(_info("h264", "yuvj420p"), Settings(encoder="libx264"))
    assert full_range.pix_fmt == "yuv420p"
    assert full_range.color_range == 2


def test_x265_holds_few_frames_above_4k() -> None:
    settings = Settings(encoder="libx265")
    small = choose_encoder(_info("hevc", "yuv420p10le"), settings)
    assert "rc-lookahead" not in small.options["x265-params"]
    large = choose_encoder(replace(_info("hevc", "yuv420p10le"), width=7680, height=3840), settings)
    params = large.options["x265-params"].split(":")
    assert {"rc-lookahead=3", "bframes=1", "ref=1"} <= set(params)


def test_other_codecs_fall_back_to_h264() -> None:
    choice = choose_encoder(_info("vp9"), Settings())
    assert choice.family == "h264"


@pytest.mark.skipif("hevc_videotoolbox" not in av.codecs_available, reason="macOS only")
def test_videotoolbox_uses_p010_for_ten_bit() -> None:
    choice = choose_encoder(_info("hevc", "yuv420p10le"), Settings())
    assert choice.codec == "hevc_videotoolbox"
    assert choice.pix_fmt == "p010le"
    assert choice.options["profile"] == "main10"


def test_unknown_encoder_fails() -> None:
    with pytest.raises(RuntimeError, match="no usable encoder"):
        choose_encoder(_info(), Settings(encoder="h264_nvenc"))


def test_device_helpers() -> None:
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda") == "cuda:0"
    assert resolve_device("auto") in {"cpu", "mps", "cuda:0"}
    assert available_memory_gib("cpu") is None
    assert not use_half(Settings(), "cpu")
    assert use_half(Settings(), "cuda:0")
    assert use_half(Settings(), "mps")
    assert use_half(Settings(half=True), "cpu")
