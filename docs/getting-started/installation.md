# Installation


## Native (macOS Apple Silicon, Linux)

Requirements: Python 3.14 and [uv](https://docs.astral.sh/uv/) (uv installs
Python 3.14 itself if needed). No system FFmpeg is needed: PyAV ships its own
FFmpeg libraries.

```bash
git clone https://github.com/KylianAUBRY/SGBlur_Video.git
cd SGBlur_Video
uv sync                                       # first run: downloads PyTorch, a few minutes
uv run sgblur-video models download yolo26s   # optional: weights are also fetched on first use
uv run sgblur-video --help
```

- Weights go to `~/.cache/sgblur-video/models` (`MODELS_DIR`) and are checked
  against their SHA-256 before every load.
- Run commands from the repository root: the defaults of `MODELS_FILE`
  (`models/registry.yaml`) and `TRACKER_CONFIG` (`configs/trackers/…`) are
  relative paths. Elsewhere, set them as absolute paths (environment or `.env`).
- On macOS, if the repository is in a folder synced by iCloud Drive (Desktop,
  Documents), `import sgblur_video` can fail after `uv sync`: see
  [troubleshooting](../guides/troubleshooting.md).
- The first command of a session takes a few extra seconds (PyTorch import,
  model load, GPU warm-up).

On macOS, native installation is the recommended way to process videos: it
uses Metal, through PyTorch MPS for detection (in FP16) and VideoToolbox for
hardware encoding. Docker containers on macOS run in a Linux virtual machine
that cannot reach Metal: detection and encoding fall back to the CPU, 8K videos
are several times slower (about 5 s per frame) and need about 6 GB of memory (see
[troubleshooting](../guides/troubleshooting.md)).

## CPU-only Linux

`uv sync` installs CPU-only PyTorch wheels on Linux (a few hundred MB instead of
several GB of CUDA libraries), **even on a machine with an NVIDIA GPU**. To use
CUDA, run the `gpu` Docker image (below).

## Docker

One Dockerfile with two targets (`docker/Dockerfile`): `cpu` (default) and
`gpu`. The default model is downloaded and verified at build time, the
service runs as a non-root user and stores everything in the `/data` volume.

```bash
docker compose -f docker/docker-compose.yml up --build      # API on :8000 + one worker
curl -s -F video=@my-video.mp4 http://localhost:8000/blur/
```

Configure it with environment variables in the Compose file (see the
[configuration reference](../reference/configuration.md)), e.g.
`KEEP_SECRET_KEY`, `CALLBACK_ALLOWED_HOSTS`, `API_TOKEN`, `RESULT_TTL_MINUTES`.

## NVIDIA GPU

Requires the NVIDIA driver and the NVIDIA Container Toolkit on the host. CUDA
comes with the PyTorch wheels (CUDA 12.6 build); NVENC/NVDEC come from the
host driver.

```bash
docker compose -f docker/docker-compose.yml -f docker/docker-compose.gpu.yml up --build
```

!!! warning "Untested"
    The GPU image builds the same way as the CPU one, but it has not been run
    on NVIDIA hardware yet (none was available during development).

## Split deployment (detection on a GPU machine)

On the GPU machine: `docker compose -f docker/docker-compose.yml -f docker/docker-compose.gpu.yml --profile split up detect`.
On the API machine: set `DETECT_URL=http://gpu-machine:8001` for the `worker`
service. Videos travel over the network: keep both machines on a private network.

## With a Panoramax instance

Panoramax does not send videos to its blurring service yet (picture uploads
only). Two ways to use SGBlur-Video today:

1. Blur videos with the API or the CLI, then upload the **best-frame
   pictures** (`frames=1`) to Panoramax with `isBlurred=true` and their
   annotations — see [Annotations and Panoramax](../concepts/annotations.md).
2. Follow the upstream discussion on video support
   ([panoramax/server/api#369](https://gitlab.com/panoramax/server/api/-/work_items/369)):
   the API contract was designed so that `API_BLUR_URL` can point to it once
   videos are accepted.
