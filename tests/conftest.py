"""Shared pytest configuration.

Tests must never depend on the developer's environment: every setting
variable is removed from the environment and ``.env`` files are ignored,
so each test builds the exact configuration it needs.
"""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from sgblur_video.config import Settings, get_settings

REPO_ROOT = Path(__file__).resolve().parent.parent

# Keep Ultralytics' settings file out of the developer's home directory.
os.environ.setdefault("YOLO_CONFIG_DIR", str(REPO_ROOT / ".pytest_cache" / "ultralytics"))


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Remove every setting variable from the environment and clear the settings cache."""
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def repo_root() -> Path:
    """Path of the repository root."""
    return REPO_ROOT
