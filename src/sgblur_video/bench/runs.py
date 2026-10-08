"""The ``benchmark`` commands: privacy (annotated dataset) and speed.

Reports describe videos by resolution, projection, codec and frame count only:
never by file name, and they never contain pictures.
"""

import itertools
import logging
import statistics
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Any

import yaml

from sgblur_video import __version__
from sgblur_video.bench.cache import DetectionCache
from sgblur_video.bench.dataset import Dataset, DatasetError
from sgblur_video.bench.metrics import ClipEvaluation, check_gate, evaluate_clip, summarize
from sgblur_video.config import Settings
from sgblur_video.core.analyze import ProgressCallback
from sgblur_video.core.decode import iter_frames, rotate_upright, to_bgr
from sgblur_video.core.detect import build_plan, merge_detections, unrotate_box
from sgblur_video.core.detections_io import Detections
from sgblur_video.core.pipeline import LoadedModel, load_model
from sgblur_video.core.postprocess import BlurPlan, build_blur_plan
from sgblur_video.core.probe import VideoInfo, probe
from sgblur_video.models import ModelEntry, resolve_model

logger = logging.getLogger(__name__)

#: Settings reported with every benchmark run.
REPORTED_SETTINGS = ("detect_profile", "conf_detect", "tile_trigger_width", "equirect_pad_ratio")


def override(settings: Settings, values: dict[str, str]) -> Settings:
    """Settings with some fields replaced, validated like environment variables.

    Args:
        settings: Base settings.
        values: Field name (or environment variable name) → raw value.

    Raises:
        ValueError: On an unknown setting or an invalid value.
    """
    data = settings.model_dump()
    for name, value in values.items():
        field = name.lower()
        if field not in Settings.model_fields:
            msg = f"unknown setting {name!r}"
            raise ValueError(msg)
        data[field] = value
    return Settings.model_validate(data)


def parse_sweeps(items: Sequence[str]) -> list[dict[str, str]]:
    """``["CONF_DETECT=0.2,0.3", "DETECT_PROFILE=standard,thorough"]`` → every combination (one if none)."""
    axes: list[list[tuple[str, str]]] = []
    for item in items:
        name, sep, values = item.partition("=")
        if not sep or not name or not values:
            msg = f"invalid --sweep {item!r}: expected NAME=v1,v2,..."
            raise ValueError(msg)
        axes.append([(name.strip(), v.strip()) for v in values.split(",") if v.strip()])
    return [dict(combo) for combo in itertools.product(*axes)]


def _settings_summary(settings: Settings) -> dict[str, Any]:
    """Reported settings; paths are reduced to file names (reports must not reveal local folders)."""
    values = {name: getattr(settings, name) for name in REPORTED_SETTINGS}
    return {name: value.name if isinstance(value, Path) else str(value) for name, value in values.items()}


def describe_video(info: VideoInfo) -> dict[str, Any]:
    """What a report may say about a video (no file name)."""
    return {
        "width": info.width,
        "height": info.height,
        "projection": info.projection,
        "codec": info.codec,
        "fps": round(info.fps, 3),
        "frames": info.frame_count,
    }


def _wrap(info: VideoInfo) -> int | None:
    return info.width if info.projection == "equirectangular" else None


def plan_for(detections: Detections, info: VideoInfo, settings: Settings) -> BlurPlan:
    """Blur plan of a video, as the pipeline would compute it."""
    return build_blur_plan(detections, settings, frame_size=(info.width, info.height), wrap_width=_wrap(info))


@dataclass
class ModelSource:
    """Loads each model at most once per device, only when a cache miss needs it."""

    settings: Settings
    model_name: str | None

    def entry(self) -> ModelEntry:
        """Entry of the model that would be used (registry name or checkpoint path; not loaded)."""
        return resolve_model(self.settings, self.model_name)

    def sha256(self) -> str:
        """Checksum of the model that would be used."""
        return self.entry().sha256

    def loader(self) -> Callable[[], LoadedModel]:
        """A function loading the model on its first call and returning the same model afterwards."""

        @cache
        def load() -> LoadedModel:
            return load_model(self.settings, self.model_name)

        return load


def run_privacy(
    dataset: Dataset,
    settings: Settings,
    *,
    sweeps: Sequence[dict[str, str]],
    thresholds: dict[str, float],
    model_name: str | None = None,
    progress: Callable[[str], ProgressCallback] | None = None,
) -> dict[str, Any]:
    """Leakage metrics of every annotated clip, for each combination of settings.

    Args:
        dataset: Annotated dataset.
        settings: Base settings.
        sweeps: Setting overrides to evaluate (one report entry each).
        thresholds: ``annotated_dataset`` section of the thresholds file.
        model_name: Registry model to use.
        progress: Builds a progress callback for a label.

    Returns:
        The report (JSON-serialisable).

    Raises:
        DatasetError: If no clip of the dataset is annotated.
    """
    manifest = dataset.load_manifest()
    clips = [(c, t) for c in manifest.clips if (t := dataset.load_ground_truth(c.id)) is not None]
    if not clips:
        msg = f"no annotated clip in {dataset.root} (run 'sgblur-video annotate import' first)"
        raise DatasetError(msg)
    models = ModelSource(settings, model_name)
    sha = models.sha256()
    loader = models.loader()
    runs = []
    for overrides in sweeps:
        variant = override(settings, overrides)
        evaluations: list[ClipEvaluation] = []
        per_clip: dict[str, Any] = {}
        timings: Counter[str] = Counter()
        for clip, truth in clips:
            info = probe(dataset.clip_video(clip.id), variant)
            cache_ = DetectionCache(dataset.cache_dir(clip.id), sha, loader)
            detections, spent = cache_.get(
                info, variant, progress=progress(f"{clip.id} analysis") if progress else None
            )
            timings.update(spent)
            plan = plan_for(detections, info, variant)
            evaluation = evaluate_clip(
                truth, plan, coverage_threshold=thresholds["coverage_threshold"], fps=info.fps
            )
            evaluations.append(evaluation)
            per_clip[clip.id] = summarize([evaluation]) | {"video": describe_video(info)}
        summary = summarize(evaluations)
        failures = check_gate(summary, thresholds)
        runs.append(
            {
                "overrides": overrides,
                "settings": _settings_summary(variant),
                "summary": summary,
                "per_clip": per_clip,
                "gate": {"passed": not failures, "failures": failures},
                "timings_s": {k: round(v, 1) for k, v in timings.items()},
            }
        )
    return {
        "kind": "privacy",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "sgblur_video": __version__,
        "model": models.entry().tag,
        "thresholds": thresholds,
        "runs": runs,
    }


def run_speed(
    video: Path,
    settings: Settings,
    *,
    models: Sequence[str],
    devices: Sequence[str],
    profiles: Sequence[str],
    frames: int,
    warmup: int = 2,
    progress: Callable[[str], ProgressCallback] | None = None,
) -> dict[str, Any]:
    """Detection speed and detection counts per model, device and detection profile.

    Only detection and cross-pass merge are timed (decoding is excluded). Counts
    of detections are **not** a quality measure: more boxes can be more false
    positives; use the privacy benchmark for recall.

    Args:
        video: Any video.
        settings: Base settings.
        models: Registry model names.
        devices: ``cpu``, ``mps``, ``cuda``…
        profiles: Detection profiles.
        frames: Timed frames per combination.
        warmup: Untimed frames first (model initialisation).
        progress: Builds a progress callback for a label.

    Returns:
        The report.
    """
    info = probe(video, settings)
    wrap = _wrap(info)
    rows = []
    for model_name, device, profile in itertools.product(models, devices, profiles):
        variant = override(settings, {"device": device, "detect_profile": profile})
        loaded = load_model(variant, model_name)
        rotated = (round(info.rotation / 90) % 2) == 1 if info.rotation else False
        upright = (info.height, info.width) if rotated else (info.width, info.height)
        plan = build_plan(
            *upright,
            projection=info.projection,
            profile=variant.detect_profile,
            tile_trigger_width=variant.tile_trigger_width,
            equirect_pad_ratio=variant.equirect_pad_ratio,
        )
        durations: list[float] = []
        counts: Counter[str] = Counter()
        callback = progress(f"{model_name} {loaded.device} {profile}") if progress else None
        for decoded in iter_frames(info.path, max_frames=frames + warmup):
            image = rotate_upright(to_bgr(decoded.frame), info.rotation)
            started = time.perf_counter()
            raw = loaded.detector.detect(image, plan, decoded.index)
            for det in raw:
                det.box = unrotate_box(det.box, info.rotation, info.width, info.height)
            merged = merge_detections(raw, variant.class_policy, wrap_width=wrap)
            elapsed = time.perf_counter() - started
            if decoded.index >= warmup:
                durations.append(elapsed)
                counts.update(d.cls for d in merged)
            if callback is not None:
                callback(decoded.index + 1, frames + warmup)
        timed = max(1, len(durations))
        rows.append(
            {
                "model": loaded.entry.tag,  # never a local path (reports are shared)
                "device": loaded.device,
                "profile": profile,
                "passes": len(plan),
                "s_per_frame": round(statistics.median(durations), 3) if durations else None,
                "fps": round(1 / statistics.median(durations), 2) if durations else None,
                "detections_per_frame": {cls: round(n / timed, 2) for cls, n in sorted(counts.items())},
            }
        )
    return {
        "kind": "speed",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "sgblur_video": __version__,
        "video": describe_video(info),
        "frames": frames,
        "rows": rows,
    }


def load_thresholds(path: Path) -> dict[str, float]:
    """``annotated_dataset`` thresholds of ``benchmarks/privacy-thresholds.yaml``."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {k: float(v) for k, v in data["annotated_dataset"].items()}
