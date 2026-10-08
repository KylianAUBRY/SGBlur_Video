"""Smoke test of the real SGBlur model (downloads ~20 MB once, cached in MODELS_DIR)."""

from pathlib import Path

import numpy as np
import pytest

from sgblur_video.config import ClassAction, DetectProfile, Settings
from sgblur_video.core.detect import build_plan
from sgblur_video.core.pipeline import load_model

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def test_model_loads_and_detects(repo_root: Path) -> None:
    settings = Settings(models_file=repo_root / "models" / "registry.yaml", device="cpu")
    try:
        model = load_model(settings)
    except Exception as exc:
        if "cannot download" in str(exc):
            pytest.skip(f"model download unavailable: {exc}")
        raise
    assert model.entry.name == "yolo26s"
    assert {"face", "plate", "sign", "direction"} <= set(model.detector.class_names)
    assert settings.classes_with(ClassAction.BLUR) <= set(model.detector.class_names)
    image = np.full((360, 640, 3), 127, np.uint8)
    plan = build_plan(640, 360, projection="flat", profile=DetectProfile.STANDARD, tile_trigger_width=5760)
    detections = model.detector.detect(image, plan, 0)
    assert isinstance(detections, list)
    assert all(d.cls in model.detector.class_names for d in detections)


def test_model_path_loads_a_local_checkpoint(repo_root: Path) -> None:
    from sgblur_video.models import ensure_weights, load_registry

    registry_settings = Settings(models_file=repo_root / "models" / "registry.yaml", device="cpu")
    entry = load_registry(registry_settings.models_file).get("yolo26s")
    try:
        weights = ensure_weights(entry, registry_settings.models_dir)
    except Exception as exc:
        pytest.skip(f"model download unavailable: {exc}")
    model = load_model(registry_settings.model_copy(update={"model_path": weights}))
    assert model.entry.family == "local"
    assert model.entry.version == f"local-{entry.sha256[:8]}"  # same file as the registry entry
    assert model.entry.local_path == weights.resolve()
    assert "face" in model.detector.class_names
