"""The privacy benchmark (method D) run on synthetic ground truth.

The scenario of the oracle doubles as an annotated clip whose ground truth is
exact: the benchmark must find no leak with the default settings, find leaks
when the protections are disabled, and give the same answer through the CLI.
"""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from sgblur_video.bench.cache import DetectionCache
from sgblur_video.bench.clips import cut_clip
from sgblur_video.bench.cvat import KeyBox, PreTrack, write_preannotation
from sgblur_video.bench.dataset import ClipEntry, Dataset, GroundTruth, GtBox, GtTrack
from sgblur_video.bench.metrics import evaluate_clip, summarize
from sgblur_video.bench.runs import plan_for
from sgblur_video.cli import app
from sgblur_video.config import Settings
from sgblur_video.core.geometry import clip
from sgblur_video.core.pipeline import LoadedModel
from sgblur_video.core.probe import VideoInfo, probe
from sgblur_video.models import load_registry
from tests.privacy.synthetic import FPS, FRAMES, HEIGHT, WIDTH, FakeDetector, scenario, write_video

pytestmark = pytest.mark.privacy

CLIP = "synthetic"


def _truth(info: VideoInfo) -> GroundTruth:
    tracks = [
        GtTrack(
            id=f"{obj.cls}:{number}",
            cls=obj.cls,  # type: ignore[arg-type]
            boxes=[
                GtBox(frame=f, box=clip(obj.box(f), info.width, info.height), readable=True)
                for f in obj.visible
            ],
        )
        for number, obj in enumerate(scenario())
        if obj.cls in ("face", "plate") and obj.detected & set(obj.visible)
    ]
    return GroundTruth(clip_id=CLIP, frames=FRAMES, width=info.width, height=info.height, tracks=tracks)


@pytest.fixture
def prepared(tmp_path: Path, repo_root: Path) -> tuple[Dataset, Settings, VideoInfo, DetectionCache]:
    source = tmp_path / "synthetic.mp4"
    write_video(source, scenario())
    settings = Settings(
        models_file=repo_root / "models" / "registry.yaml",
        tracker_config=repo_root / "configs" / "trackers" / "tracktrack-recall.yaml",
        encoder="libx264",
    )
    dataset = Dataset(tmp_path / "dataset")
    dataset.clip_dir(CLIP).mkdir(parents=True)
    cut = cut_clip(
        probe(source, settings),
        settings,
        start_s=0.0,
        duration_s=FRAMES / FPS,
        clip_path=dataset.clip_video(CLIP),
        proxy_path=dataset.proxy_video(CLIP),
        proxy_max_width=WIDTH // 2,
    )
    assert cut.frames == FRAMES
    assert (cut.proxy_width, cut.proxy_height) == (WIDTH // 2, HEIGHT // 2)
    info = probe(dataset.clip_video(CLIP), settings)
    entry = load_registry(settings.models_file).get("yolo26s")
    loads: list[int] = []

    def loader() -> LoadedModel:
        loads.append(1)
        assert len(loads) == 1, "the detector must be loaded once, on the first cache miss"
        return LoadedModel(detector=FakeDetector(scenario()), entry=entry, device="cpu")  # type: ignore[arg-type]

    return dataset, settings, info, DetectionCache(dataset.cache_dir(CLIP), entry.sha256, loader)


def test_benchmark_finds_no_leak_with_defaults_and_leaks_without_protections(
    prepared: tuple[Dataset, Settings, VideoInfo, DetectionCache], repo_root: Path
) -> None:
    _dataset, settings, info, cache = prepared
    detections, spent = cache.get(info, settings)
    assert "analysis_s" in spent
    assert len(detections.frames) == FRAMES
    assert cache.get(info, settings) == (detections, {})

    # Another tracker replays tracking on the same boxes, without running the detector again.
    bytetrack = settings.model_copy(
        update={"tracker_config": repo_root / "configs" / "trackers" / "bytetrack-recall.yaml"}
    )
    retracked, spent = cache.get(info, bytetrack)
    assert "tracking_s" in spent
    assert retracked.header.tracking["tracker"] == "bytetrack"
    assert [[d.box for d in f.detections] for f in retracked.frames] == [
        [d.box for d in f.detections] for f in detections.frames
    ]

    truth = _truth(info)
    protected = summarize(
        [evaluate_clip(truth, plan_for(detections, info, settings), coverage_threshold=0.9, fps=FPS)]
    )
    assert protected["overall"]["unprotected"] == 0, protected

    weak = settings.model_copy(
        update={"blur_temporal_padding_frames": 0, "max_interpolation_gap_s": 0.0, "link_max_gap_s": 0.0}
    )
    leaking = summarize(
        [evaluate_clip(truth, plan_for(detections, info, weak), coverage_threshold=0.9, fps=FPS)]
    )
    assert leaking["overall"]["unprotected"] > 0
    assert leaking["tracks_ever_leaked"] > 0


def test_cli_import_and_privacy_gate(
    prepared: tuple[Dataset, Settings, VideoInfo, DetectionCache],
    repo_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, settings, info, cache = prepared
    cache.get(info, settings)  # cached: the CLI finds the detections and loads no model
    manifest = dataset.load_manifest()
    manifest.put(
        ClipEntry(
            id=CLIP,
            source_sha256="0" * 64,
            start_s=0.0,
            frames=FRAMES,
            fps=FPS,
            width=info.width,
            height=info.height,
            proxy_width=WIDTH // 2,
            proxy_height=HEIGHT // 2,
            projection=info.projection,
        )
    )
    dataset.save_manifest(manifest)
    # A "CVAT export" of the exact ground truth, in proxy pixels.
    truth = _truth(info)
    export = tmp_path / "export.xml"
    tracks = [
        PreTrack(
            track.cls,
            tuple(KeyBox(b.frame, tuple(v / 2 for v in b.box)) for b in track.boxes),  # type: ignore[arg-type]
            track.boxes[-1].frame + 1,
        )
        for track in truth.tracks
    ]
    write_preannotation(export, tracks, frames=FRAMES, width=WIDTH // 2, height=HEIGHT // 2)
    # The annotator ticks "readable" on every box.
    xml = export.read_text(encoding="utf-8")
    export.write_text(xml.replace('name="readable">false<', 'name="readable">true<'), encoding="utf-8")

    monkeypatch.setenv("MODELS_FILE", str(settings.models_file))
    monkeypatch.setenv("TRACKER_CONFIG", str(settings.tracker_config))
    monkeypatch.setenv("ENCODER", "libx264")
    runner = CliRunner()
    result = runner.invoke(
        app, ["annotate", "import", str(export), "--dataset", str(dataset.root), "--id", CLIP]
    )
    assert result.exit_code == 0, result.output
    imported = dataset.load_ground_truth(CLIP)
    assert imported is not None
    assert len(imported.tracks) == len(truth.tracks)
    assert dataset.load_manifest().get(CLIP).ground_truth_sha256 is not None

    thresholds = str(repo_root / "benchmarks" / "privacy-thresholds.yaml")
    reports = tmp_path / "reports"
    args = ["benchmark", "privacy", "--dataset", str(dataset.root), "--thresholds", thresholds]
    result = runner.invoke(
        app, [*args, "--report-dir", str(reports), "--sweep", "BLUR_TEMPORAL_PADDING_FRAMES=15,0"]
    )
    assert result.exit_code == 0, result.output
    assert "| BLUR_TEMPORAL_PADDING_FRAMES=15 | pass |" in result.output
    assert len(list(reports.glob("privacy-*.json"))) == 1

    # The gate fails (exit code 1) when the first run leaks readable objects.
    result = runner.invoke(
        app,
        [
            *args,
            "--sweep",
            "MAX_INTERPOLATION_GAP_S=0",
            "--sweep",
            "LINK_MAX_GAP_S=0",
            "--sweep",
            "BLUR_TEMPORAL_PADDING_FRAMES=0",
        ],
    )
    assert result.exit_code == 1, result.output
    assert "FAIL" in result.output
