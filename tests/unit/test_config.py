"""Tests of sgblur_video.config."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from sgblur_video.config import (
    DEFAULT_CLASS_POLICY,
    BlurMethod,
    ClassAction,
    DetectProfile,
    Settings,
    get_settings,
)


def test_defaults_are_privacy_first() -> None:
    settings = Settings()
    assert settings.api_name == "SGBlur-Video"
    assert settings.conf_detect <= settings.conf_blur < settings.conf_sign
    assert settings.blur_method is BlurMethod.PIXELATE_BLUR
    assert settings.detect_profile is DetectProfile.STANDARD
    assert settings.blur_temporal_padding_frames > 0
    assert settings.blur_box_margin > 0
    assert settings.class_policy == DEFAULT_CLASS_POLICY
    assert settings.callback_allowed_hosts == []
    assert not settings.keep_enabled


def test_reads_environment_without_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONF_BLUR", "0.2")
    monkeypatch.setenv("DETECT_URL", "http://detect:8001")
    monkeypatch.setenv("blur_method", "solid")  # case insensitive
    settings = Settings()
    assert settings.conf_blur == pytest.approx(0.2)
    assert str(settings.detect_url) == "http://detect:8001/"
    assert settings.blur_method is BlurMethod.SOLID


@pytest.mark.parametrize(
    "name", ["MODEL_NAME", "DETECT_URL", "API_TOKEN", "KEEP_SECRET_KEY", "TMP_DIR", "KEEP_DIR"]
)
def test_empty_optional_variables_mean_unset(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    monkeypatch.setenv(name, "  ")
    assert getattr(Settings(), name.lower()) is None


def test_comma_separated_lists(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACCEPTED_CONTAINERS", "MP4, mov ,")
    monkeypatch.setenv("CALLBACK_ALLOWED_HOSTS", "panoramax.local,api.example.org")
    settings = Settings()
    assert settings.accepted_containers == ["mp4", "mov"]
    assert settings.callback_allowed_hosts == ["panoramax.local", "api.example.org"]


def test_class_policy_from_json(monkeypatch: pytest.MonkeyPatch) -> None:
    policy = {"face": "blur", "plate": "blur", "sign": "annotate"}
    monkeypatch.setenv("CLASS_POLICY", json.dumps(policy))
    settings = Settings()
    assert settings.classes_with(ClassAction.BLUR) == {"face", "plate"}
    assert settings.classes_with(ClassAction.ANNOTATE) == {"sign"}


def test_policy_without_blur_class_is_rejected() -> None:
    with pytest.raises(ValidationError, match="at least one class with action 'blur'"):
        Settings(class_policy={"sign": ClassAction.ANNOTATE})


def test_detect_threshold_above_blur_threshold_is_rejected() -> None:
    with pytest.raises(ValidationError, match="CONF_DETECT must be lower"):
        Settings(conf_detect=0.5, conf_blur=0.3)


@pytest.mark.parametrize("device", ["auto", "cpu", "mps", "cuda", "cuda:1"])
def test_valid_devices(device: str) -> None:
    assert Settings(device=device).device == device


@pytest.mark.parametrize("device", ["gpu", "cuda:x", "MPS ", ""])
def test_invalid_devices(device: str) -> None:
    with pytest.raises(ValidationError):
        Settings(device=device)


@pytest.mark.parametrize(("raw", "expected"), [("auto", "auto"), ("true", True), ("false", False)])
def test_half_accepts_auto_or_bool(monkeypatch: pytest.MonkeyPatch, raw: str, expected: object) -> None:
    monkeypatch.setenv("HALF", raw)
    assert Settings().half == expected


def test_derived_directories() -> None:
    settings = Settings(data_dir=Path("/srv/data"))
    assert settings.effective_tmp_dir == Path("/srv/data/tmp")
    assert settings.effective_keep_dir == Path("/srv/data/keep")
    custom = Settings(data_dir=Path("/srv/data"), tmp_dir=Path("/fast/tmp"))
    assert custom.effective_tmp_dir == Path("/fast/tmp")


def test_secrets_are_masked_in_dumps() -> None:
    settings = Settings(keep_secret_key="super-secret", api_token="token-value")
    dumped = json.dumps(settings.model_dump(mode="json"))
    assert "super-secret" not in dumped
    assert "token-value" not in dumped
    assert settings.keep_enabled


def test_settings_are_immutable() -> None:
    settings = Settings()
    with pytest.raises(ValidationError):
        settings.conf_blur = 0.9  # type: ignore[misc]


def test_get_settings_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    first = get_settings()
    monkeypatch.setenv("CONF_BLUR", "0.3")
    assert get_settings() is first
    get_settings.cache_clear()
    assert get_settings().conf_blur == pytest.approx(0.3)


def test_dotenv_file_is_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("BLUR_BOX_MARGIN=0.3\n", encoding="utf-8")
    assert Settings(_env_file=env_file).blur_box_margin == pytest.approx(0.3)  # type: ignore[call-arg]


def test_every_field_is_documented() -> None:
    for name, field in Settings.model_fields.items():
        assert field.description, f"{name} has no description"
        extra = field.json_schema_extra
        assert isinstance(extra, dict), f"{name} has no documentation hints"
        assert extra.get("group"), f"{name} has no documentation group"
