"""Tests of sgblur_video.models (registry, selection, class policy)."""

from pathlib import Path

import pytest
import yaml

from sgblur_video.config import DEFAULT_CLASS_POLICY, ClassAction
from sgblur_video.models import (
    ClassPolicyError,
    ModelRegistry,
    RegistryError,
    check_class_policy,
    load_registry,
    select_model,
)


def _entry(name: str, family: str = "yolo26", memory: float = 2.0, classes: list[str] | None = None) -> dict:
    return {
        "name": name,
        "family": family,
        "version": "0.1.0",
        "file": f"{name}.pt",
        "url": f"https://example.org/{name}.pt",
        "sha256": "0" * 64,
        "classes": classes or ["sign", "plate", "face"],
        "train_imgsz": 2048,
        "min_memory_gib": memory,
        "licence": "test",
        "source": "https://example.org",
    }


def _registry(*entries: dict) -> ModelRegistry:
    return ModelRegistry.model_validate({"schema_version": 1, "models": list(entries)})


def test_repository_registry_is_valid(repo_root: Path) -> None:
    registry = load_registry(repo_root / "models" / "registry.yaml")
    yolo26 = registry.get("yolo26s")
    assert yolo26.classes == ("direction", "sign", "plate", "face")
    assert yolo26.tag == "yolo26s/0.1.0"
    for entry in registry.models:
        # Every shipped model must satisfy the default privacy policy.
        check_class_policy(entry.classes, DEFAULT_CLASS_POLICY)
        assert "/-/raw/" in str(entry.url), "download URLs must be pinned to a commit"


def test_unknown_model_lists_known_names() -> None:
    with pytest.raises(RegistryError, match=r"unknown model 'yolo99'.*yolo26s"):
        _registry(_entry("yolo26s")).get("yolo99")


def test_duplicated_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicated model names"):
        _registry(_entry("a"), _entry("a"))


def test_duplicated_classes_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicated class names"):
        _registry(_entry("a", classes=["face", "face"]))


def test_invalid_registry_file(tmp_path: Path) -> None:
    path = tmp_path / "registry.yaml"
    path.write_text(yaml.safe_dump({"schema_version": 2, "models": []}), encoding="utf-8")
    with pytest.raises(RegistryError, match="invalid model registry"):
        load_registry(path)
    with pytest.raises(RegistryError):
        load_registry(tmp_path / "missing.yaml")


def test_selection_by_name_wins() -> None:
    registry = _registry(_entry("yolo26s"), _entry("yolo11s", family="yolo11"))
    assert select_model(registry, family="yolo26", available_memory_gib=100, name="yolo11s").name == "yolo11s"


@pytest.mark.parametrize(
    ("memory", "expected"),
    [(None, "n"), (0.5, "n"), (2.5, "s"), (5.9, "s"), (6.0, "m"), (48.0, "m")],
)
def test_automatic_selection_by_memory(memory: float | None, expected: str) -> None:
    registry = _registry(_entry("m", memory=6), _entry("n", memory=1), _entry("s", memory=2))
    assert select_model(registry, family="yolo26", available_memory_gib=memory).name == expected


def test_unknown_family() -> None:
    with pytest.raises(RegistryError, match="no model of family"):
        select_model(_registry(_entry("a")), family="yolo27", available_memory_gib=None)


def test_missing_blur_class_fails_closed() -> None:
    with pytest.raises(ClassPolicyError, match=r"\['plate'\]"):
        check_class_policy(["sign", "face"], DEFAULT_CLASS_POLICY)


def test_policy_warnings() -> None:
    warnings = check_class_policy(
        ["sign", "plate", "face", "bicycle"],
        {"face": ClassAction.BLUR, "plate": ClassAction.BLUR, "direction": ClassAction.ANNOTATE},
    )
    assert any("'direction'" in warning for warning in warnings)
    assert any("'bicycle'" in warning for warning in warnings)
    assert any("'sign'" in warning for warning in warnings)


def _checkpoint(path: Path, names: dict[int, str]) -> Path:
    import torch

    torch.save(
        {"model": None, "names": names, "train_args": {"imgsz": 1024}, "date": "2026-01-02T00:00"}, path
    )
    return path


def test_local_checkpoint_entry(tmp_path: Path) -> None:
    from sgblur_video.models import ensure_weights, local_entry

    path = _checkpoint(tmp_path / "My Model v2.pt", {0: "face", 1: "plate", 2: "sign"})
    entry = local_entry(path)
    assert entry.name == "my-model-v2"
    assert entry.version == f"local-{entry.sha256[:8]}"
    assert entry.classes == ("face", "plate", "sign")
    assert entry.train_imgsz == 1024
    assert entry.local_path == path.resolve()
    assert ensure_weights(entry, tmp_path / "unused") == path.resolve()  # used in place, never downloaded


def test_model_precedence(tmp_path: Path, repo_root: Path) -> None:
    from sgblur_video.config import Settings
    from sgblur_video.models import WeightsError, is_model_path, resolve_model

    assert is_model_path("x.pt")
    assert is_model_path("models/x")
    assert not is_model_path("yolo11l")
    local = _checkpoint(tmp_path / "local.pt", {0: "face", 1: "plate"})
    registry = repo_root / "models" / "registry.yaml"
    base = Settings(models_file=registry)
    assert resolve_model(base).name == "yolo26s"  # automatic choice
    assert resolve_model(Settings(models_file=registry, model_name="yolo11l")).name == "yolo11l"
    with_path = Settings(models_file=registry, model_name="yolo11l", model_path=local)
    assert resolve_model(with_path).name == "local"  # MODEL_PATH beats MODEL_NAME
    assert resolve_model(with_path, "yolo11n").name == "yolo11n"  # --model beats both
    assert resolve_model(base, str(local)).family == "local"  # --model accepts a path
    with pytest.raises(WeightsError, match="not found"):
        resolve_model(base, str(tmp_path / "missing.pt"))


def test_model_path_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    from sgblur_video.config import Settings

    monkeypatch.setenv("MODEL_PATH", "~/models/x.pt")
    assert Settings().model_path == Path.home() / "models" / "x.pt"
    monkeypatch.setenv("MODEL_PATH", "")
    assert Settings().model_path is None


def test_registry_entries_need_a_url() -> None:
    entry = _entry("a")
    del entry["url"]
    with pytest.raises(ValueError, match="needs a url"):
        ModelRegistry.model_validate({"schema_version": 1, "models": [entry]})
