# Installation

!!! note "Planned"
    Native installation works for development today. Docker images and the
    Panoramax `docker compose` setup arrive with roadmap step 6.

## Native (macOS Apple Silicon, Linux)

Requirements: Python 3.14 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/KylianAUBRY/SGBlur_Video.git
cd SGBlur_Video
uv sync
uv run sgblur-video --help
```

On macOS, native installation is the recommended way to process videos: Docker
containers on macOS cannot use the Apple GPU (MPS) or the VideoToolbox hardware
encoder, which makes 8K videos several times slower.

## CPU-only Linux

`uv sync` installs CPU-only PyTorch wheels on Linux (a few hundred MB instead of
several GB of CUDA libraries).

## NVIDIA GPU

*Planned (step 6).*

## Docker

*Planned (step 6): `Dockerfile.cpu`, `Dockerfile.gpu`, `docker-compose.yml`.*

## With a Panoramax instance

*Planned (step 6).*
