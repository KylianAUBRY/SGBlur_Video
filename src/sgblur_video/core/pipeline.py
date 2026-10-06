"""Orchestration of the whole pipeline, shared by the CLI, the worker and the Detect API."""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sgblur_video.config import ClassAction, Settings
from sgblur_video.core.analyze import ProgressCallback, analyze
from sgblur_video.core.debug import DebugOverlay
from sgblur_video.core.detect import YoloDetector
from sgblur_video.core.detections_io import Detections, DetectionsFormatError, Footer, read_detections
from sgblur_video.core.device import available_memory_gib, resolve_device, use_half
from sgblur_video.core.postprocess import BlurPlan, build_blur_plan
from sgblur_video.core.probe import VideoInfo, probe
from sgblur_video.core.render import RenderStats, render
from sgblur_video.models import ModelEntry, check_class_policy, ensure_weights, load_registry, select_model

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadedModel:
    """A detector ready to run, with its registry entry and device."""

    detector: YoloDetector
    entry: ModelEntry
    device: str

    def header(self, settings: Settings) -> dict[str, Any]:
        """Model description written in the ``detections.jsonl`` header."""
        return {
            "name": self.entry.name,
            "version": self.entry.version,
            "sha256": self.entry.sha256,
            "classes": list(self.detector.class_names),
            "policy": {name: action.value for name, action in settings.class_policy.items()},
        }


def load_model(settings: Settings, model_name: str | None = None) -> LoadedModel:
    """Select, download (if needed), verify and load the detection model.

    Args:
        settings: Model, device and policy settings.
        model_name: Overrides ``MODEL_NAME``.

    Returns:
        The loaded model.

    Raises:
        sgblur_video.models.ClassPolicyError: If the model lacks a class to blur.
    """
    device = resolve_device(settings.device)
    registry = load_registry(settings.models_file)
    entry = select_model(
        registry,
        family=settings.model_family,
        available_memory_gib=available_memory_gib(device),
        name=model_name or settings.model_name,
    )
    weights = ensure_weights(entry, settings.models_dir)
    for warning in check_class_policy(entry.classes, settings.class_policy):
        logger.warning(warning)
    detector = YoloDetector(
        str(weights),
        device=device,
        half=use_half(settings, device),
        conf=settings.conf_detect,
        classes=list(settings.class_policy),
    )
    # The registry is informative; the checkpoint itself is authoritative.
    check_class_policy(detector.class_names, settings.class_policy)
    logger.info("model %s on %s", entry.tag, device)
    return LoadedModel(detector=detector, entry=entry, device=device)


def run_detect(
    input_video: Path,
    output: Path,
    settings: Settings,
    *,
    model_name: str | None = None,
    max_frames: int | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[VideoInfo, Footer]:
    """Pass 1 only: write ``detections.jsonl`` for a video."""
    info = probe(input_video, settings)
    model = load_model(settings, model_name)
    footer = analyze(
        info,
        model.detector,
        settings,
        output,
        model=model.header(settings),
        device=model.device,
        max_frames=max_frames,
        progress=progress,
    )
    return info, footer


@dataclass
class RenderResult:
    """Outcome of post-processing and pass 2."""

    plan: BlurPlan
    render: RenderStats


def run_render(
    input_video: Path,
    detections_path: Path,
    output: Path,
    settings: Settings,
    *,
    debug_output: Path | None = None,
    allow_partial: bool = False,
    progress: ProgressCallback | None = None,
) -> RenderResult:
    """Post-processing and pass 2 from an existing ``detections.jsonl``.

    Raises:
        DetectionsFormatError: If the file is incomplete and ``allow_partial`` is false, or does
            not match the video.
    """
    info = probe(input_video, settings)
    detections = read_detections(detections_path)
    if not detections.complete and not allow_partial:
        msg = (
            "detections.jsonl is incomplete (no footer or complete=false); use allow_partial to render anyway"
        )
        raise DetectionsFormatError(msg)
    _check_matches(detections, info)
    return _render(info, detections, output, settings, debug_output=debug_output, progress=progress)


def _check_matches(detections: Detections, info: VideoInfo) -> None:
    video = detections.header.video
    if (video.get("width"), video.get("height")) != (info.width, info.height):
        msg = "detections.jsonl was produced for a video of another size"
        raise DetectionsFormatError(msg)


def _render(
    info: VideoInfo,
    detections: Detections,
    output: Path,
    settings: Settings,
    *,
    debug_output: Path | None,
    progress: ProgressCallback | None,
) -> RenderResult:
    plan = build_blur_plan(detections, settings, frame_size=(info.width, info.height), fps=info.fps)
    logger.info("blur plan: %s", plan.stats)
    debug = None
    if debug_output is not None:
        debug = (debug_output, DebugOverlay(plan, detections, settings.classes_with(ClassAction.ANNOTATE)))
    stats = render(
        info,
        plan,
        output,
        settings,
        max_frames=len(detections.frames),
        debug=debug,
        progress=progress,
    )
    return RenderResult(plan=plan, render=stats)


def run_blur(
    input_video: Path,
    output: Path,
    settings: Settings,
    *,
    detections_path: Path,
    model_name: str | None = None,
    debug_output: Path | None = None,
    max_frames: int | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Full pipeline: pass 1, post-processing, pass 2.

    Args:
        input_video: Input video.
        output: Blurred video to write.
        settings: Settings.
        detections_path: Where to write ``detections.jsonl``.
        model_name: Overrides ``MODEL_NAME``.
        debug_output: Optional annotated debug video.
        max_frames: Process only the first frames (development).
        progress: Progress callback for both passes.

    Returns:
        A JSON-serialisable summary (frames, timings, blur statistics, streams).
    """
    info, footer = run_detect(
        input_video,
        detections_path,
        settings,
        model_name=model_name,
        max_frames=max_frames,
        progress=progress,
    )
    detections = read_detections(detections_path)
    result = _render(info, detections, output, settings, debug_output=debug_output, progress=progress)
    summary = {
        "frames": footer.frames,
        "complete": footer.complete,
        "analysis_s": footer.elapsed_s,
        "render_s": round(result.render.elapsed_s, 2),
        "detections": footer.counts,
        "blur": result.plan.stats,
        "render": asdict(result.render),
    }
    logger.info("summary: %s", json.dumps(summary))
    return summary
