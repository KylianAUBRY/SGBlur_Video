"""Privacy gate (option A): every confident face and plate is blurred, weak detections are not, no sign is.

Tracking favours precision ([ADR-0013](docs/adr/0013-precise-tracking.md)): an
object is blurred on the frames where it was detected with a score of at least
``CONF_BLUR`` (and briefly around them), an object only ever detected below it
is left alone rather than risk a false blur.

Runs the real pipeline (merge, Ultralytics tracking, post-processing,
rendering, encoding) with a scripted fake detector on generated videos, then
measures the output video. Thresholds come from ``benchmarks/privacy-thresholds.yaml``.
"""

import math
from pathlib import Path

import av
import numpy as np
import pytest
import yaml
from PIL import Image

from sgblur_video.config import Settings
from sgblur_video.core.analyze import analyze
from sgblur_video.core.detections_io import read_detections
from sgblur_video.core.frames import BestFrameWriter
from sgblur_video.core.geometry import Box, iou
from sgblur_video.core.postprocess import BlurPlan, build_blur_plan
from sgblur_video.core.probe import probe
from sgblur_video.core.render import render
from sgblur_video.semantics.annotations import build_annotations, find_sign_tracks
from tests.privacy.synthetic import (
    FRAMES,
    HEIGHT_360,
    WIDTH_360,
    FakeDetector,
    SyntheticObject,
    WrappedFakeDetector,
    scenario,
    scenario_360,
    texture_energy,
    wrapped_energy,
    write_video,
)

pytestmark = pytest.mark.privacy

#: The default tracker (BoT-SORT) and the optical-flow tracker must both pass.
TRACKERS = ["botsort.yaml", "flow.yaml"]

# A blurred checkerboard keeps a small fraction of its gradient energy; the raw texture is ~50.
BLURRED_MAX_ENERGY = 8.0
SHARP_MIN_ENERGY = 25.0


#: The fake detector jitters boxes by up to 2 px and BLUR_BOX_MARGIN is small (5 %): objects
#: are measured this far inside their edges.
JITTER = 2.0


def _measured(cls: str, box: Box) -> Box:
    """Where an object is measured: inside its edges, and for a face (an oval) inside its ellipse."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    rx, ry = (box[2] - box[0]) / 2 - JITTER, (box[3] - box[1]) / 2 - JITTER
    if cls == "face":
        rx, ry = rx / math.sqrt(2), ry / math.sqrt(2)  # the rectangle inscribed in the ellipse
    return (cx - rx, cy - ry, cx + rx, cy + ry)


def _confident(obj: SyntheticObject, settings: Settings) -> set[int]:
    """Frames where the fake detector reports the object with a score of at least ``CONF_BLUR``."""
    return {f for f in obj.detected & set(obj.visible) if obj.score(f) >= settings.conf_blur}


def _luma_frames(path: Path) -> list[np.ndarray]:
    with av.open(str(path)) as container:
        return [frame.reformat(format="gray").to_ndarray() for frame in container.decode(video=0)]


@pytest.fixture(scope="module")
def thresholds(request: pytest.FixtureRequest) -> dict[str, int]:
    root = Path(request.config.rootpath)
    data = yaml.safe_load((root / "benchmarks" / "privacy-thresholds.yaml").read_text(encoding="utf-8"))
    return dict(data["synthetic_oracle"])


@pytest.mark.parametrize("tracker", TRACKERS)
@pytest.mark.parametrize(("codec", "pix_fmt"), [("libx264", "yuv420p"), ("libx265", "yuv420p10le")])
def test_confident_faces_and_plates_are_blurred_weak_ones_are_not(
    tmp_path: Path, repo_root: Path, thresholds: dict[str, int], codec: str, pix_fmt: str, tracker: str
) -> None:
    objects = scenario()
    source = tmp_path / "synthetic.mp4"
    write_video(source, objects, codec=codec, pix_fmt=pix_fmt)
    settings = Settings(
        models_file=repo_root / "models" / "registry.yaml",
        tracker_config=repo_root / "configs" / "trackers" / tracker,
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
    weak_blurred: list[tuple[str, int]] = []
    for obj in objects:
        confident = _confident(obj, settings)
        for frame in obj.visible:
            box = _measured(obj.cls, obj.box(frame))
            assert texture_energy(original[frame], box) > SHARP_MIN_ENERGY  # the test object is visible
            energy = texture_energy(blurred[frame], box)
            if obj.cls in ("face", "plate") and frame in confident and energy > BLURRED_MAX_ENERGY:
                leaked.append((obj.cls, frame, round(energy, 1)))
            if obj.cls in ("face", "plate") and not confident and energy < SHARP_MIN_ENERGY:
                weak_blurred.append((obj.cls, frame))
            if obj.cls == "sign" and energy < SHARP_MIN_ENERGY:
                blurred_sign_frames += 1
    assert len(leaked) <= thresholds["max_leaked_object_frames"], f"visible confident detections: {leaked}"
    assert blurred_sign_frames <= thresholds["max_blurred_sign_frames"], "a sign was blurred"
    # Precision: the plate only ever detected at 0.3 (below CONF_BLUR) is never blurred.
    assert any(o.cls == "plate" and not _confident(o, settings) for o in objects)
    assert weak_blurred == [], f"weak-only objects blurred: {weak_blurred[:5]}"


def test_oracle_detects_leaks_without_blur(tmp_path: Path) -> None:
    """Negative control: with nothing blurred, every confident face and plate frame must count as a leak."""
    objects = scenario()
    source = tmp_path / "synthetic.mp4"
    write_video(source, objects)
    settings = Settings(encoder="libx264")
    info = probe(source, settings)
    output = tmp_path / "blurred.mp4"
    render(info, BlurPlan(frame_count=FRAMES), output, settings)
    blurred = _luma_frames(output)
    confident = [
        (obj, frame) for obj in objects if obj.cls in ("face", "plate") for frame in _confident(obj, settings)
    ]
    leaks = sum(
        texture_energy(blurred[frame], _measured(obj.cls, obj.box(frame))) > BLURRED_MAX_ENERGY
        for obj, frame in confident
    )
    assert leaks == len(confident) > 10


@pytest.mark.parametrize("tracker", TRACKERS)
def test_signs_are_annotated_once_and_best_frames_are_blurred(
    tmp_path: Path, repo_root: Path, tracker: str
) -> None:
    """One annotation per physical sign, short false positives dropped, best-frame JPEGs blurred."""
    objects = scenario()
    source = tmp_path / "synthetic.mp4"
    write_video(source, objects)
    settings = Settings(tracker_config=repo_root / "configs" / "trackers" / tracker, encoder="libx264")
    info = probe(source, settings)
    detections_path = tmp_path / "detections.jsonl"
    analyze(info, FakeDetector(objects), settings, detections_path, model={"name": "fake", "version": "1"})
    detections = read_detections(detections_path)
    tracks = find_sign_tracks(detections, settings, fps=info.fps)
    annotations = build_annotations(
        tracks, detections, settings, frame_size=(info.width, info.height), rotation=0
    )

    real_signs = [o for o in objects if o.cls == "sign" and len(o.detected) >= settings.sign_min_track_length]
    assert len(annotations) == len(real_signs) == 2
    for sign in real_signs:
        matches = [
            a for a in annotations if iou(tuple(map(float, a.shape)), sign.box(a.video.best_frame)) > 0.6
        ]
        assert len(matches) == 1, f"sign at {sign.box(0)} annotated {len(matches)} times"

    plan = build_blur_plan(detections, settings, frame_size=(info.width, info.height), fps=info.fps)
    writer = BestFrameWriter(tmp_path / "frames", annotations, info)
    render(info, plan, tmp_path / "blurred.mp4", settings, frame_sink=writer)
    index = writer.write_index()
    assert index.exists()
    assert len(writer.written) == len({a.video.best_frame for a in annotations})
    for entry in writer.written:
        with Image.open(tmp_path / "frames" / entry["file"]) as image:
            picture = np.asarray(image.convert("L"))
        for obj in objects:
            if obj.cls in ("face", "plate") and entry["frame"] in _confident(obj, settings):
                box = _measured(obj.cls, obj.box(entry["frame"]))
                assert texture_energy(picture, box) < BLURRED_MAX_ENERGY


@pytest.mark.parametrize("tracker", TRACKERS)
def test_objects_crossing_the_360_seam_are_blurred_on_both_sides(
    tmp_path: Path, repo_root: Path, tracker: str
) -> None:
    """A face crossing the 0°/360° seam (missed briefly while crossing) and a plate straddling it."""
    objects = scenario_360()
    source = tmp_path / "equirect.mp4"
    write_video(source, objects, width=WIDTH_360, height=HEIGHT_360, wrap=True)
    settings = Settings(tracker_config=repo_root / "configs" / "trackers" / tracker, encoder="libx264")
    info = probe(source, settings)
    assert info.projection == "equirectangular"
    detections_path = tmp_path / "detections.jsonl"
    analyze(info, WrappedFakeDetector(objects, WIDTH_360), settings, detections_path, model={"name": "fake"})
    detections = read_detections(detections_path)
    plan = build_blur_plan(
        detections, settings, frame_size=(info.width, info.height), fps=info.fps, wrap_width=info.width
    )
    # The face is one chain across the seam, not two.
    assert plan.stats["chains_face"] == 1
    output = tmp_path / "blurred.mp4"
    render(info, plan, output, settings)
    original, blurred = _luma_frames(source), _luma_frames(output)
    leaked = []
    for obj in objects:
        for frame in obj.visible:
            box = _measured(obj.cls, obj.box(frame))
            assert wrapped_energy(original[frame], box, WIDTH_360) > SHARP_MIN_ENERGY
            energy = wrapped_energy(blurred[frame], box, WIDTH_360)
            if energy > BLURRED_MAX_ENERGY:
                leaked.append((obj.cls, frame, round(energy, 1)))
    assert leaked == []  # including the 8 frames the face is missed: a short gap is interpolated

    # Negative control: the same detections without seam handling leak.
    flat_plan = build_blur_plan(detections, settings, frame_size=(info.width, info.height), fps=info.fps)
    render(info, flat_plan, output, settings)
    flat = _luma_frames(output)
    flat_leaks = sum(
        wrapped_energy(flat[frame], _measured(obj.cls, obj.box(frame)), WIDTH_360) > BLURRED_MAX_ENERGY
        for obj in objects
        for frame in obj.visible
    )
    assert flat_leaks > 10
