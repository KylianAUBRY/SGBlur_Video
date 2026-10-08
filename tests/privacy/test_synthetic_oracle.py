"""Privacy gate (option A): every face and plate detection is blurred on its frame, no sign is.

Every frame is blurred on its own detections, like SGBlur blurs a picture: a
frame where the detector misses an object leaves it visible (checked too, as
the documented behaviour). Runs the real pipeline (merge, sign tracking, blur
plan, rendering, encoding) with a scripted fake detector on generated videos,
then measures the output video. Thresholds come from
``benchmarks/privacy-thresholds.yaml``.
"""

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
from sgblur_video.core.postprocess import MIN_BLUR_SIZE, build_blur_plan
from sgblur_video.core.probe import probe
from sgblur_video.core.render import render
from sgblur_video.semantics.annotations import build_annotations, find_sign_tracks
from tests.privacy.synthetic import (
    FRAMES,
    HEIGHT_360,
    WIDTH_360,
    FakeDetector,
    WrappedFakeDetector,
    scenario,
    scenario_360,
    texture_energy,
    wrapped_energy,
    write_video,
)

pytestmark = pytest.mark.privacy

#: Sign tracking: the default optical-flow tracker and the overlap-based one must both pass.
TRACKERS = ["flow.yaml", "tracktrack-recall.yaml"]
#: Pixels left out on each side of a detected box when measuring it (codec bleed at the edges).
INSET = 2

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


def _inset(box: Box) -> Box:
    return (box[0] + INSET, box[1] + INSET, box[2] - INSET, box[3] - INSET)


def _blurred_detections(detections_path: Path, settings: Settings) -> list[tuple[str, int, Box]]:
    """Face and plate detections the plan must blur: ``(class, frame, box)``."""
    return [
        (det.class_, frame.index, det.box)
        for frame in read_detections(detections_path).frames
        for det in frame.detections
        if det.class_ in ("face", "plate")
        and det.score >= settings.conf_detect
        and min(det.box[2] - det.box[0], det.box[3] - det.box[1]) >= MIN_BLUR_SIZE
    ]


@pytest.mark.parametrize(("codec", "pix_fmt"), [("libx264", "yuv420p"), ("libx265", "yuv420p10le")])
def test_every_face_and_plate_detection_is_blurred_on_its_frame(
    tmp_path: Path, repo_root: Path, thresholds: dict[str, int], codec: str, pix_fmt: str
) -> None:
    objects = scenario()
    source = tmp_path / "synthetic.mp4"
    write_video(source, objects, codec=codec, pix_fmt=pix_fmt)
    settings = Settings(
        models_file=repo_root / "models" / "registry.yaml",
        encoder="libx265" if codec == "libx265" else "libx264",
    )
    info = probe(source, settings)
    detections_path = tmp_path / "detections.jsonl"
    footer = analyze(info, FakeDetector(objects), settings, detections_path, model={"name": "fake"})
    assert footer.frames == FRAMES
    plan = build_blur_plan(read_detections(detections_path), settings, frame_size=(info.width, info.height))
    output = tmp_path / "blurred.mp4"
    stats = render(info, plan, output, settings)
    assert stats.frames == FRAMES

    original = _luma_frames(source)
    blurred = _luma_frames(output)
    assert len(blurred) == FRAMES

    expected = _blurred_detections(detections_path, settings)
    assert len(expected) > 50
    leaked = [
        (cls, frame, round(energy, 1))
        for cls, frame, box in expected
        if (energy := texture_energy(blurred[frame], _inset(box))) > BLURRED_MAX_ENERGY
    ]
    assert len(leaked) <= thresholds["max_leaked_object_frames"], f"visible detections: {leaked}"

    # Nothing is carried between frames: a frame without a detection of an object leaves it sharp.
    blurred_frames = {(cls, frame) for cls, frame, _box in expected}
    missed = 0
    blurred_sign_frames = 0
    for obj in objects:
        for frame in obj.visible:
            box = obj.box(frame)
            assert texture_energy(original[frame], box) > SHARP_MIN_ENERGY  # the test object is visible
            energy = texture_energy(blurred[frame], box)
            if obj.cls in ("face", "plate") and (obj.cls, frame) not in blurred_frames:
                missed += 1
                assert energy > SHARP_MIN_ENERGY, f"{obj.cls} blurred on frame {frame} without a detection"
            if obj.cls == "sign" and energy < SHARP_MIN_ENERGY:
                blurred_sign_frames += 1
    assert missed > 10  # the scenario's misses (late onset, holes, low scores) stay visible
    assert blurred_sign_frames <= thresholds["max_blurred_sign_frames"], "a sign was blurred"


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

    plan = build_blur_plan(detections, settings, frame_size=(info.width, info.height))
    writer = BestFrameWriter(tmp_path / "frames", annotations, info)
    render(info, plan, tmp_path / "blurred.mp4", settings, frame_sink=writer)
    index = writer.write_index()
    assert index.exists()
    assert len(writer.written) == len({a.video.best_frame for a in annotations})
    expected = _blurred_detections(detections_path, settings)
    for entry in writer.written:
        with Image.open(tmp_path / "frames" / entry["file"]) as image:
            picture = np.asarray(image.convert("L"))
        for _cls, frame, box in expected:
            if frame == entry["frame"]:
                assert texture_energy(picture, _inset(box)) < BLURRED_MAX_ENERGY


def test_objects_crossing_the_360_seam_are_blurred_on_both_sides(tmp_path: Path) -> None:
    """Detections of a face crossing the 0°/360° seam and of a plate straddling it are blurred whole."""
    objects = scenario_360()
    source = tmp_path / "equirect.mp4"
    write_video(source, objects, width=WIDTH_360, height=HEIGHT_360, wrap=True)
    settings = Settings(encoder="libx264")
    info = probe(source, settings)
    assert info.projection == "equirectangular"
    detections_path = tmp_path / "detections.jsonl"
    analyze(info, WrappedFakeDetector(objects, WIDTH_360), settings, detections_path, model={"name": "fake"})
    detections = read_detections(detections_path)
    plan = build_blur_plan(detections, settings, frame_size=(info.width, info.height), wrap_width=info.width)
    output = tmp_path / "blurred.mp4"
    render(info, plan, output, settings)
    original, blurred = _luma_frames(source), _luma_frames(output)
    expected = _blurred_detections(detections_path, settings)
    straddling = [(cls, frame, box) for cls, frame, box in expected if box[2] > WIDTH_360]
    assert straddling, "the scenario must have detections crossing the seam"
    leaked = []
    for cls, frame, box in expected:
        assert wrapped_energy(original[frame], _inset(box), WIDTH_360) > SHARP_MIN_ENERGY
        energy = wrapped_energy(blurred[frame], _inset(box), WIDTH_360)
        if energy > BLURRED_MAX_ENERGY:
            leaked.append((cls, frame, round(energy, 1)))
    assert leaked == []

    # Negative control: the same detections without seam handling leak on the other side.
    flat_plan = build_blur_plan(detections, settings, frame_size=(info.width, info.height))
    render(info, flat_plan, output, settings)
    flat = _luma_frames(output)
    flat_leaks = sum(
        wrapped_energy(flat[frame], _inset(box), WIDTH_360) > BLURRED_MAX_ENERGY
        for _cls, frame, box in straddling
    )
    assert flat_leaks > 5
