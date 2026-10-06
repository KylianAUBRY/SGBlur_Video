"""SGBlur-Video: privacy blurring of street-level videos for Panoramax.

The package detects and tracks faces, licence plates and traffic signs in a
video, irreversibly blurs faces and plates on every frame where they appear,
and returns one Panoramax annotation per physical traffic sign.

Importing this package sets privacy-preserving defaults for Ultralytics before
it can be imported anywhere else in the process: no telemetry or online checks
(``YOLO_OFFLINE``), no runtime ``pip install`` (``YOLO_AUTOINSTALL``) and quiet
logs (``YOLO_VERBOSE``). Variables already set in the environment are kept.
"""

import os
from importlib.metadata import PackageNotFoundError, version

#: Environment defaults applied to Ultralytics; see the module docstring.
ULTRALYTICS_ENV_DEFAULTS: dict[str, str] = {
    "YOLO_OFFLINE": "true",
    "YOLO_AUTOINSTALL": "false",
    "YOLO_VERBOSE": "false",
}

for _name, _value in ULTRALYTICS_ENV_DEFAULTS.items():
    os.environ.setdefault(_name, _value)

try:
    __version__: str = version("sgblur-video")
except PackageNotFoundError:  # pragma: no cover - only when running from an uninstalled tree
    __version__ = "0.0.0+unknown"

__all__ = ["ULTRALYTICS_ENV_DEFAULTS", "__version__"]
