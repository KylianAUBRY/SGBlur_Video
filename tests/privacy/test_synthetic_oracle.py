"""Privacy gate (option A): every frame of every face and plate must be blurred, no sign may be.

Runs the real pipeline (merge, Ultralytics tracking, post-processing,
rendering, encoding) with a scripted fake detector on generated videos, then
measures the output video. Thresholds come from ``benchmarks/privacy-thresholds.yaml``.
"""

from pathlib import Path

import av
import numpy as np
import pytest
import yaml

from sgblur_video.config import Settings
from sgblur_video.core.analyze import analyze
from sgblur_video.core.detections_io import read_detections
from sgblur_video.core.postprocess import build_blur_plan
from sgblur_video.core.probe import probe
from sgblur_video.core.render import render
from tests.privacy.synthetic import FRAMES, FakeDetector, scenario, texture_energy, write_video

pytestmark = pytest.mark.privacy

# A blurred checkerboard keeps a small fraction of its gradient energy; the raw texture is ~50.
BLURRED_MAX_ENERGY = 8.0
SHARP_MIN_ENERGY = 25.0


def _luma_frames(path: Path) -> list[np.ndarray]:
    with av.open(str(path)) as container:
        return [frame.reformat(format="gray").to_ndarray() for frame in container.decode(video=0)]


@pytest.fixture(scope="module")
def thresholds(request: pytest.FixtureRequest) -> dict[str, int]:
    root = Path(request.config.rootpath)
    data = yaml.safe_load((root / "benchmarks" / "privacy-thresholds.yaml").read_text(encoding="utf-8"))
    return dict(data["synthetic_oracle"])


@pytest.mark.parametrize(("codec", "pix_fmt"), [("libx264", "yuv420p"), ("libx265", "yuv420p10le")])
def test_every_face_and_plate_frame_is_blurred(
    tmp_path: Path, repo_root: Path, thresholds: dict[str, int], codec: str, pix_fmt: str
) -> None:
    objects = scenario()
    source = tmp_path / "synthetic.mp4"
    write_video(source, objects, codec=codec, pix_fmt=pix_fmt)
    settings = Settings(
        models_file=repo_root / "models" / "registry.yaml",
        tracker_config=repo_root / "configs" / "trackers" / "tracktrack-recall.yaml",
        encoder="libx265" if codec == "libx265" else "libx264",
    )
    info = probe(source, settings)
    detections_path = tmp_path / "detections.jsonl"
    footer = analyze(info, FakeDetector(objects), settings, detections_path, model={"name": "fake"})
    assert footer.frames == FRAMES
    detections = read_detections(detections_path)
    plan = build_blur_plan(detections, settings, frame_size=(info.width, info.height), fps=info.fps)
    output = tmp_path / "blurred.mp4"
    stats = render(info, plan, output, settings)
    assert stats.frames == FRAMES

    original = _luma_frames(source)
    blurred = _luma_frames(output)
    assert len(blurred) == FRAMES

    leaked: list[tuple[str, int, float]] = []
    blurred_sign_frames = 0
    for obj in objects:
        for frame in obj.visible:
            box = obj.box(frame)
            assert texture_energy(original[frame], box) > SHARP_MIN_ENERGY  # the test object is visible
            energy = texture_energy(blurred[frame], box)
            if obj.cls in ("face", "plate") and energy > BLURRED_MAX_ENERGY:
                leaked.append((obj.cls, frame, round(energy, 1)))
            if obj.cls == "sign" and energy < SHARP_MIN_ENERGY:
                blurred_sign_frames += 1
    assert len(leaked) <= thresholds["max_leaked_object_frames"], f"visible object-frames: {leaked}"
    assert blurred_sign_frames <= thresholds["max_blurred_sign_frames"], "a sign was blurred"


def test_oracle_detects_leaks_when_protections_are_disabled(tmp_path: Path, repo_root: Path) -> None:
    """Negative control: without padding, interpolation and linking, the scripted misses must leak."""
    objects = scenario()
    source = tmp_path / "synthetic.mp4"
    write_video(source, objects)
    settings = Settings(
        tracker_config=repo_root / "configs" / "trackers" / "tracktrack-recall.yaml",
        encoder="libx264",
        blur_temporal_padding_frames=0,
        max_interpolation_gap_s=0.0,
        link_max_gap_s=0.0,
    )
    info = probe(source, settings)
    detections_path = tmp_path / "detections.jsonl"
    analyze(info, FakeDetector(objects), settings, detections_path, model={"name": "fake"})
    plan = build_blur_plan(
        read_detections(detections_path), settings, frame_size=(info.width, info.height), fps=info.fps
    )
    output = tmp_path / "blurred.mp4"
    render(info, plan, output, settings)
    blurred = _luma_frames(output)
    leaks = sum(
        texture_energy(blurred[frame], obj.box(frame)) > BLURRED_MAX_ENERGY
        for obj in objects
        if obj.cls in ("face", "plate")
        for frame in obj.visible
    )
    assert leaks > 10
