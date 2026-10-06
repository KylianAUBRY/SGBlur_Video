"""Tests of package-level behaviour (Ultralytics environment defaults, docs freshness)."""

import os
import subprocess
import sys
from pathlib import Path

import sgblur_video


def _run_python(code: str, env: dict[str, str]) -> str:
    result = subprocess.run(  # noqa: S603 - fixed interpreter and code
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True
    )
    return result.stdout.strip()


def test_ultralytics_privacy_defaults_are_set() -> None:
    env = {k: v for k, v in os.environ.items() if not k.startswith("YOLO_")}
    output = _run_python(
        "import os, sgblur_video; print(os.environ['YOLO_OFFLINE'], os.environ['YOLO_AUTOINSTALL'])", env
    )
    assert output == "true false"


def test_existing_environment_is_kept() -> None:
    env = {**os.environ, "YOLO_VERBOSE": "true"}
    output = _run_python("import os, sgblur_video; print(os.environ['YOLO_VERBOSE'])", env)
    assert output == "true"


def test_version_is_exposed() -> None:
    assert sgblur_video.__version__


def test_configuration_reference_is_up_to_date(repo_root: Path) -> None:
    result = subprocess.run(  # noqa: S603 - fixed interpreter and script
        [sys.executable, str(repo_root / "scripts" / "gen_config_reference.py"), "--check"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
