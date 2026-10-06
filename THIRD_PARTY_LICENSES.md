# Third-party licences

This file lists the third-party components SGBlur-Video uses at runtime, the
model weights it downloads, and the data used for testing. Licences are taken
from the installed package metadata (versions locked in `uv.lock`) and from
upstream pages; see [docs/license.md](docs/license.md) for what they imply.

## Direct runtime dependencies

| Component | Version (lock) | Licence | Notes |
|---|---|---|---|
| [ultralytics](https://github.com/ultralytics/ultralytics) | 8.4.173 | AGPL-3.0 | Detection and trackers. Also pulls `ultralytics-platform` and `ultralytics-thop` (AGPL-3.0). |
| [PyTorch](https://pytorch.org) (`torch`) | 2.14.1 | BSD-style (Apache-2.0, BSD-2/3-Clause, MIT, BSL-1.0 parts) | via Ultralytics |
| `torchvision` | 0.29.1 | BSD-3-Clause | via Ultralytics |
| [PyAV](https://github.com/PyAV-Org/PyAV) (`av`) | 19.0.1 | BSD-3-Clause | Wheels bundle FFmpeg 9.0.2 built with `libx264` and `libx265` (GPL); treat the bundled libraries as GPL. |
| `opencv-python-headless` | 5.0.0.93 | Apache-2.0 | Wheels bundle FFmpeg libraries (LGPL) |
| `numpy` | 2.5.3 | BSD-3-Clause (and bundled permissive licences) | |
| `lap` | 0.5.13 | BSD-2-Clause | Linear assignment for trackers |
| `pydantic`, `pydantic-settings` | 2.13.5, 2.15.0 | MIT | |
| `PyYAML` | 6.0.3 | MIT | |
| `typer` | 0.27.2 | MIT | |
| `fastapi` | 0.142.2 | MIT | |
| `uvicorn` | 0.54.0 | BSD-3-Clause | |
| `python-multipart` | 0.0.32 | Apache-2.0 | |
| `httpx` | 0.28.1 | BSD-3-Clause | |
| `cryptography` | 50.0.2 | Apache-2.0 OR BSD-3-Clause | |
| `pillow` | 12.3.0 | MIT-CMU | Best-frame JPEGs with EXIF |

Transitive dependencies are listed in `uv.lock`; notable ones: `polars` (MIT),
`matplotlib` (PSF-based matplotlib licence), `requests`
(Apache-2.0), `psutil` (BSD-3-Clause).

## Model weights (downloaded, not distributed)

| Model | Source | Upstream licence statements |
|---|---|---|
| `yolo26s_panoramax.pt` | [SGBlur](https://gitlab.com/panoramax/server/sgblur) commit `34c318b` | Hugging Face card `Panoramax/detect_face_plate_sign`: etalab-2.0. Checkpoint metadata (written by Ultralytics): AGPL-3.0. SGBlur code: MIT. |
| `yolo11s_panoramax.pt` | same | same |

## Test data

No media is committed. Fixtures downloaded by the test suite will be listed
here with their source and licence when they are added (step 4).

## Documentation tooling (development only)

`mkdocs` (BSD-2-Clause), `mkdocs-material` (MIT), `mkdocstrings-python` (ISC),
`pytest` (MIT), `ruff` (MIT), `mypy` (MIT), `pre-commit` (MIT).
