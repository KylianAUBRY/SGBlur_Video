"""Accelerator selection: CUDA, Apple MPS or CPU, and the memory available for models."""

import logging

from sgblur_video.config import Settings

logger = logging.getLogger(__name__)

GIB = 1024**3


def resolve_device(setting: str) -> str:
    """Turn ``DEVICE`` into an Ultralytics device string.

    Args:
        setting: ``auto``, ``cpu``, ``mps``, ``cuda`` or ``cuda:N``.

    Returns:
        ``cuda:N``, ``mps`` or ``cpu``. ``auto`` prefers CUDA, then MPS.
    """
    import torch

    if setting == "cuda":
        return "cuda:0"
    if setting != "auto":
        return setting
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def available_memory_gib(device: str) -> float | None:
    """Free accelerator memory in GiB, or ``None`` on CPU.

    On Apple Silicon the GPU shares system memory; the value is the memory
    PyTorch recommends for MPS minus what it already uses.
    """
    import torch

    if device.startswith("cuda"):
        index = int(device.split(":")[1]) if ":" in device else 0
        free, _total = torch.cuda.mem_get_info(index)
        return free / GIB
    if device == "mps":
        recommended = float(torch.mps.recommended_max_memory())
        return max(0.0, recommended - float(torch.mps.current_allocated_memory())) / GIB
    return None


def use_half(settings: Settings, device: str) -> bool:
    """Whether to run FP16 inference (``HALF=auto`` enables it on CUDA and Apple MPS).

    On MPS, FP16 made 8K detection 18 % faster and kept every detection above
    0.15 of FP32 on 60 busy frames (514 faces, 146 plates, 91 signs).
    """
    if settings.half == "auto":
        return device.startswith("cuda") or device == "mps"
    return bool(settings.half)
