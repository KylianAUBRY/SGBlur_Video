"""Orchestration of the whole pipeline, shared by the CLI, the worker and the Detect API."""

import json
import logging
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sgblur_video.config import ClassAction, Settings
from sgblur_video.core.analyze import ProgressCallback, analyze
from sgblur_video.core.debug import DebugOverlay
from sgblur_video.core.detect import YoloDetector
from sgblur_video.core.detections_io import Detections, DetectionsFormatError, Footer, read_detections
from sgblur_video.core.device import available_memory_gib, resolve_device, use_half
from sgblur_video.core.frames import BestFrameWriter, extract_best_frames
from sgblur_video.core.mp4boxes import Mp4BoxError, transplant
from sgblur_video.core.postprocess import BlurPlan, build_blur_plan
from sgblur_video.core.probe import VideoInfo, probe
from sgblur_video.core.render import RegionSink, RenderStats, render
from sgblur_video.models import ModelEntry, check_class_policy, ensure_weights, resolve_model
from sgblur_video.semantics.annotations import (
    Annotation,
    Metadata,
    build_annotations,
    dump_metadata,
    find_sign_tracks,
)
from sgblur_video.telemetry.gps import GOPRO_HANDLER, GpsTrack, read_gps

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
        model_name: Registry name or checkpoint path; overrides ``MODEL_PATH`` and ``MODEL_NAME``.

    Returns:
        The loaded model.

    Raises:
        sgblur_video.models.ClassPolicyError: If the model lacks a class to blur.
    """
    device = resolve_device(settings.device)
    entry = resolve_model(settings, model_name, available_memory_gib=available_memory_gib(device))
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
    """Outcome of post-processing and pass 2.

    Attributes:
        plan: Blur plan.
        render: Rendering statistics.
        metadata: Panoramax metadata (sign annotations, statistics).
        frames: Index entries of the best-frame pictures written (empty if not requested).
        region_sink: The region sink used during rendering (``keep=1`` recorder), if any.
    """

    plan: BlurPlan
    render: RenderStats
    metadata: Metadata
    frames: list[dict[str, Any]]
    region_sink: RegionSink | None = None


def run_render(
    input_video: Path,
    detections_path: Path,
    output: Path,
    settings: Settings,
    *,
    debug_output: Path | None = None,
    frames_dir: Path | None = None,
    allow_partial: bool = False,
    progress: ProgressCallback | None = None,
) -> RenderResult:
    """Post-processing and pass 2 from an existing ``detections.jsonl``.

    Raises:
        DetectionsFormatError: If the file is incomplete and ``allow_partial`` is false, or does
            not match the video.
    """
    info = probe(input_video, settings)
    detections = load_detections(detections_path, info, allow_partial=allow_partial)
    return render_detections(
        info,
        detections,
        output,
        settings,
        debug_output=debug_output,
        frames_dir=frames_dir,
        progress=progress,
    )


def load_detections(path: Path, info: VideoInfo, *, allow_partial: bool) -> Detections:
    """Read ``detections.jsonl`` and check it belongs to ``info``'s video.

    Raises:
        DetectionsFormatError: Incomplete file (unless ``allow_partial``) or another video's file.
    """
    detections = read_detections(path)
    if not detections.complete and not allow_partial:
        msg = "detections.jsonl is incomplete (no footer or complete=false); use allow_partial to render it"
        raise DetectionsFormatError(msg)
    video = detections.header.video
    if (video.get("width"), video.get("height")) != (info.width, info.height):
        msg = "detections.jsonl was produced for a video of another size"
        raise DetectionsFormatError(msg)
    return detections


def _video_summary(info: VideoInfo, gps: GpsTrack | None) -> dict[str, Any]:
    telemetry = (
        "gpmf" if any(s.handler_name == GOPRO_HANDLER and s.copyable for s in info.streams) else "none"
    )
    return {
        "width": info.width,
        "height": info.height,
        "duration_s": round(info.duration_s, 3),
        "frame_count": info.frame_count,
        "projection": info.projection,
        "telemetry": telemetry,
        "gps": gps is not None,
    }


def _wrap_width(info: VideoInfo) -> int | None:
    return info.width if info.projection == "equirectangular" else None


def _sign_annotations(
    detections: Detections, info: VideoInfo, settings: Settings, gps: GpsTrack | None
) -> list[Annotation]:
    wrap = _wrap_width(info)
    tracks = find_sign_tracks(detections, settings, fps=info.fps, wrap_width=wrap)
    return build_annotations(
        tracks,
        detections,
        settings,
        frame_size=(info.width, info.height),
        rotation=info.rotation,
        wrap=wrap is not None,
        gps=gps,
    )


def _stats(plan: BlurPlan, annotations: list[Annotation], detections: Detections) -> dict[str, Any]:
    return {
        "tracks": {
            "face": plan.stats.get("chains_face", 0),
            "plate": plan.stats.get("chains_plate", 0),
            "signage": len(annotations),
        },
        "blurred_boxes": {
            source: plan.stats.get(f"boxes_{source}", 0)
            for source in ("detected", "interpolated", "padded", "orphan")
        },
        "frames_with_blur": plan.stats.get("frames_with_blur", 0),
        "model": f"{detections.header.model.get('name')}/{detections.header.model.get('version')}",
        "tracker": detections.header.tracking.get("tracker"),
    }


def render_detections(
    info: VideoInfo,
    detections: Detections,
    output: Path,
    settings: Settings,
    *,
    debug_output: Path | None = None,
    frames_dir: Path | None = None,
    progress: ProgressCallback | None = None,
    region_sink_factory: Callable[[BlurPlan], RegionSink] | None = None,
    blurring_id: str | None = None,
) -> RenderResult:
    """Post-processing, sign annotations and pass 2 for already-analysed detections.

    Args:
        info: Probed video.
        detections: Its ``detections.jsonl``.
        output: Blurred video to write.
        settings: Settings.
        debug_output: Optional annotated debug video.
        frames_dir: Optional folder for best-frame pictures.
        progress: Progress callback.
        region_sink_factory: Builds a sink receiving original regions from the plan (``keep=1``).
        blurring_id: Identifier written in the metadata (random by default).

    Returns:
        Plan, rendering statistics, metadata and best-frame entries.
    """
    plan = build_blur_plan(
        detections, settings, frame_size=(info.width, info.height), fps=info.fps, wrap_width=_wrap_width(info)
    )
    logger.info("blur plan: %s", plan.stats)
    gps = read_gps(info)
    annotations = _sign_annotations(detections, info, settings, gps)
    logger.info("signs: %d annotations", len(annotations))
    debug = None
    if debug_output is not None:
        debug = (debug_output, DebugOverlay(plan, detections, settings.classes_with(ClassAction.ANNOTATE)))
    writer = BestFrameWriter(frames_dir, annotations, info, gps=gps) if frames_dir is not None else None
    region_sink = region_sink_factory(plan) if region_sink_factory is not None else None
    stats = render(
        info,
        plan,
        output,
        settings,
        max_frames=len(detections.frames),
        debug=debug,
        frame_sink=writer,
        region_sink=region_sink,
        progress=progress,
    )
    if writer is not None:
        writer.write_index()
    try:
        transplanted = asdict(transplant(info.path, output))
    except Mp4BoxError as exc:
        logger.warning("metadata boxes not transplanted: %s", exc)
        transplanted = {}
    metadata = Metadata(
        blurring_id=blurring_id or str(uuid.uuid4()),
        service_name=settings.api_name,
        annotations=annotations,
        video=_video_summary(info, gps),
        stats=_stats(plan, annotations, detections)
        | {"dropped_streams": stats.dropped_streams, "restored_metadata": transplanted},
    )
    return RenderResult(
        plan=plan,
        render=stats,
        metadata=metadata,
        frames=writer.written if writer else [],
        region_sink=region_sink,
    )


def write_metadata(metadata: Metadata, path: Path) -> None:
    """Write the metadata JSON (``class`` keys, no null fields)."""
    path.write_text(json.dumps(dump_metadata(metadata), indent=2, ensure_ascii=False), encoding="utf-8")


def run_blur(
    input_video: Path,
    output: Path,
    settings: Settings,
    *,
    detections_path: Path,
    metadata_path: Path | None = None,
    frames_dir: Path | None = None,
    model_name: str | None = None,
    debug_output: Path | None = None,
    max_frames: int | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Full pipeline: pass 1, post-processing, sign annotations, pass 2.

    Args:
        input_video: Input video.
        output: Blurred video to write.
        settings: Settings.
        detections_path: Where to write ``detections.jsonl``.
        metadata_path: Where to write the metadata JSON (annotations), if wanted.
        frames_dir: Folder for best-frame pictures of signs, if wanted.
        model_name: Registry name or checkpoint path; overrides ``MODEL_PATH`` and ``MODEL_NAME``.
        debug_output: Optional annotated debug video.
        max_frames: Process only the first frames (development).
        progress: Progress callback for both passes.

    Returns:
        A JSON-serialisable summary (frames, timings, blur statistics, streams, signs).
    """
    info, footer = run_detect(
        input_video,
        detections_path,
        settings,
        model_name=model_name,
        max_frames=max_frames,
        progress=progress,
    )
    detections = load_detections(detections_path, info, allow_partial=True)
    result = render_detections(
        info,
        detections,
        output,
        settings,
        debug_output=debug_output,
        frames_dir=frames_dir,
        progress=progress,
    )
    result.metadata.stats["processing_s"] = round(footer.elapsed_s + result.render.elapsed_s, 1)
    if metadata_path is not None:
        write_metadata(result.metadata, metadata_path)
    summary = {
        "frames": footer.frames,
        "complete": footer.complete,
        "analysis_s": footer.elapsed_s,
        "render_s": round(result.render.elapsed_s, 2),
        "detections": footer.counts,
        "blur": result.plan.stats,
        "signs": len(result.metadata.annotations),
        "best_frames": len(result.frames),
        "render": asdict(result.render),
    }
    logger.info("summary: %s", json.dumps(summary))
    return summary


def run_signs(
    input_video: Path,
    output: Path,
    settings: Settings,
    *,
    detections_path: Path,
    frames_dir: Path | None = None,
    model_name: str | None = None,
    max_frames: int | None = None,
    progress: ProgressCallback | None = None,
) -> Metadata:
    """Detect and deduplicate traffic signs without rendering a video.

    Faces and plates are still detected: best-frame pictures (``frames_dir``)
    are blurred like the video would be.

    Returns:
        Metadata with one annotation per physical sign (no ``blurring_id``: nothing is kept).
    """
    info, _footer = run_detect(
        input_video,
        detections_path,
        settings,
        model_name=model_name,
        max_frames=max_frames,
        progress=progress,
    )
    detections = load_detections(detections_path, info, allow_partial=True)
    plan = build_blur_plan(
        detections, settings, frame_size=(info.width, info.height), fps=info.fps, wrap_width=_wrap_width(info)
    )
    gps = read_gps(info)
    annotations = _sign_annotations(detections, info, settings, gps)
    if frames_dir is not None:
        writer = BestFrameWriter(frames_dir, annotations, info, gps=gps)
        extract_best_frames(info, plan, writer, settings, max_frames=len(detections.frames))
    metadata = Metadata(
        service_name=settings.api_name,
        annotations=annotations,
        video=_video_summary(info, gps),
        stats=_stats(plan, annotations, detections),
    )
    write_metadata(metadata, output)
    return metadata
