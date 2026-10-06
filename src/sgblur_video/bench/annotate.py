"""``annotate export`` and ``annotate import``: build the annotated privacy dataset."""

import json
import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path

from sgblur_video.bench.cache import DetectionCache
from sgblur_video.bench.clips import PROXY_MAX_WIDTH, ProgressCallback, cut_clip, preannotation_tracks
from sgblur_video.bench.cvat import labels_json, read_cvat_video, write_preannotation
from sgblur_video.bench.dataset import (
    ClipEntry,
    Dataset,
    DatasetError,
    GroundTruth,
    check_clip_id,
    sha256_file,
)
from sgblur_video.bench.runs import ModelSource
from sgblur_video.config import Settings
from sgblur_video.core.probe import probe

logger = logging.getLogger(__name__)

#: Labels to paste in CVAT's "Raw" label editor, written at the root of the dataset.
CVAT_LABELS = "cvat-labels.json"


def export_clip(
    source: Path,
    dataset: Dataset,
    clip_id: str,
    settings: Settings,
    *,
    start_s: float,
    duration_s: float,
    proxy_max_width: int = PROXY_MAX_WIDTH,
    progress: ProgressCallback | None = None,
) -> ClipEntry:
    """Cut a clip into the dataset with its CVAT proxy, and record it in the manifest.

    Re-exporting an id that is not annotated yet replaces the clip and clears
    its cached detections.

    Args:
        source: Original video.
        dataset: Dataset folder.
        clip_id: New clip identifier.
        settings: Pipeline settings (encoder).
        start_s: Clip start in ``source``.
        duration_s: Clip duration.
        proxy_max_width: Maximum width of the proxy.
        progress: Called with ``(frames_done, frames_total)``.

    Returns:
        The manifest entry.

    Raises:
        DatasetError: If the clip id exists and is already annotated, or the start is past the end.
    """
    check_clip_id(clip_id)
    if dataset.ground_truth_path(clip_id).exists():
        msg = f"clip {clip_id!r} is already annotated; choose another id"
        raise DatasetError(msg)
    info = probe(source, settings)
    if start_s >= info.duration_s:
        msg = f"start {start_s} s is past the end of the video ({info.duration_s:.1f} s)"
        raise DatasetError(msg)
    shutil.rmtree(dataset.cache_dir(clip_id), ignore_errors=True)
    dataset.clip_dir(clip_id).mkdir(parents=True, exist_ok=True)
    cut = cut_clip(
        info,
        settings,
        start_s=start_s,
        duration_s=duration_s,
        clip_path=dataset.clip_video(clip_id),
        proxy_path=dataset.proxy_video(clip_id),
        proxy_max_width=proxy_max_width,
        progress=progress,
    )
    clip_info = probe(dataset.clip_video(clip_id), settings)
    entry = ClipEntry(
        id=clip_id,
        source_sha256=sha256_file(source),
        start_s=start_s,
        frames=cut.frames,
        fps=round(clip_info.fps, 3),
        width=clip_info.width,
        height=clip_info.height,
        proxy_width=cut.proxy_width,
        proxy_height=cut.proxy_height,
        projection=clip_info.projection,
    )
    manifest = dataset.load_manifest()
    manifest.put(entry)
    dataset.save_manifest(manifest)
    (dataset.root / CVAT_LABELS).write_text(json.dumps(labels_json(), indent=2) + "\n", encoding="utf-8")
    return entry


def preannotate_clip(
    dataset: Dataset,
    clip_id: str,
    settings: Settings,
    *,
    conf: float = 0.25,
    min_frames: int = 5,
    keyframe_step_s: float = 0.5,
    model_name: str | None = None,
    progress: ProgressCallback | None = None,
) -> int:
    """Write ``preannotation.xml`` for a clip from the model's detections (computed once, then cached).

    Args:
        dataset: Dataset folder.
        clip_id: Clip identifier.
        settings: Pipeline settings: the detections are those the benchmark uses.
        conf: Minimum best score of a pre-annotated track.
        min_frames: Minimum number of frames on which a pre-annotated track was detected.
        keyframe_step_s: Time between two pre-annotation key boxes.
        model_name: Registry model.
        progress: Called with ``(frames_done, frames_total)`` during detection.

    Returns:
        The number of pre-annotated tracks.

    Raises:
        DatasetError: If the clip is unknown.
    """
    entry = dataset.load_manifest().get(clip_id)
    clip_info = probe(dataset.clip_video(clip_id), settings)
    models = ModelSource(settings, model_name)
    cache = DetectionCache(dataset.cache_dir(clip_id), models.sha256(), models.loader())
    detections, _ = cache.get(clip_info, settings, progress=progress)
    tracks = preannotation_tracks(
        detections,
        settings,
        conf=conf,
        fps=clip_info.fps,
        proxy_factor=entry.proxy_width / entry.width,
        wrap_width=clip_info.width if clip_info.projection == "equirectangular" else None,
        keyframe_step=max(1, round(keyframe_step_s * clip_info.fps)),
        min_frames=min_frames,
    )
    write_preannotation(
        dataset.preannotation(clip_id),
        tracks,
        frames=entry.frames,
        width=entry.proxy_width,
        height=entry.proxy_height,
    )
    return len(tracks)


def import_annotation(dataset: Dataset, clip_id: str, export: Path, *, annotator: str = "") -> GroundTruth:
    """Convert a CVAT for video 1.1 export into ``ground_truth.json`` and record it in the manifest.

    Args:
        dataset: Dataset folder.
        clip_id: Clip the export belongs to.
        export: The XML file exported from CVAT.
        annotator: Who annotated the clip.

    Returns:
        The imported ground truth.

    Raises:
        DatasetError: If the clip is unknown or the export does not match it.
    """
    manifest = dataset.load_manifest()
    entry = manifest.get(clip_id)
    tracks, _skipped = read_cvat_video(
        export,
        frames=entry.frames,
        width=entry.width,
        height=entry.height,
        proxy_width=entry.proxy_width,
        proxy_height=entry.proxy_height,
    )
    truth = GroundTruth(
        clip_id=clip_id, frames=entry.frames, width=entry.width, height=entry.height, tracks=tracks
    )
    dataset.save_ground_truth(truth)
    manifest.put(
        entry.model_copy(
            update={
                "annotator": annotator,
                "annotated_on": datetime.now(UTC).date(),
                "ground_truth_sha256": sha256_file(export),
            }
        )
    )
    dataset.save_manifest(manifest)
    return truth
