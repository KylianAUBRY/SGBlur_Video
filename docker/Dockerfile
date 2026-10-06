# syntax=docker/dockerfile:1.7
#
# SGBlur-Video images. Two targets:
#   cpu (default)  docker build -f docker/Dockerfile -t sgblur-video:cpu .   (last stage)
#   gpu (NVIDIA)   docker build -f docker/Dockerfile --target gpu -t sgblur-video:gpu .
# The GPU image needs the NVIDIA Container Toolkit on the host (CUDA comes with the
# PyTorch wheels; NVENC/NVDEC come from the host driver).

ARG PYTHON_VERSION=3.14
ARG UV_VERSION=0.12.15

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM python:${PYTHON_VERSION}-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    YOLO_OFFLINE=true \
    YOLO_AUTOINSTALL=false \
    MODELS_DIR=/models \
    MODELS_FILE=/app/models/registry.yaml \
    TRACKER_CONFIG=/app/configs/trackers/tracktrack-recall.yaml \
    DATA_DIR=/data
COPY --from=uv /uv /uvx /bin/
WORKDIR /app

# Dependencies first (cached layer): CPU-only PyTorch wheels on Linux, see pyproject.toml.
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups --no-install-project
COPY src ./src
COPY configs ./configs
COPY models/registry.yaml ./models/registry.yaml
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups --no-editable

# Bake the default model into the image (downloaded from its pinned URL, SHA-256 verified).
RUN sgblur-video models download yolo26s

RUN useradd --system --uid 10001 --home-dir /app sgblur \
    && mkdir -p /data /tmp/ultralytics \
    && chown -R sgblur /data /tmp/ultralytics
USER sgblur
VOLUME ["/data"]
EXPOSE 8000 8001
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/', timeout=4)" || exit 1
CMD ["sgblur-video", "serve", "--host", "0.0.0.0", "--workers", "1"]

FROM base AS gpu
USER root
# Same PyTorch version as uv.lock, CUDA 12.6 build (broad driver compatibility, x86_64 only).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python --reinstall \
        --index-url https://download.pytorch.org/whl/cu126 \
        "torch==2.14.1" "torchvision==0.29.1"
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    DEVICE=auto
USER sgblur

# Last stage = default target.
FROM base AS cpu
