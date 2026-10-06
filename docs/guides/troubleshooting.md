# FAQ and troubleshooting

## `ModuleNotFoundError: No module named 'sgblur_video'` after `uv sync` (macOS)

Python 3.14 ignores `.pth` files that carry the macOS *hidden* flag. The files
inside `.venv` sometimes get that flag (we have seen it applied to the whole
directory after a `uv sync`), and the editable install of `sgblur_video` then
disappears. Fix:

```bash
chflags -R nohidden .venv
```

The test suite is not affected: pytest adds `src/` to the import path itself.

## `objc: Class AVF… is implemented in both …` warnings (macOS)

PyAV and OpenCV each bundle their own FFmpeg libraries. The duplicated classes
belong to `libavdevice` (camera capture), which SGBlur-Video never uses; the
warning is harmless.

## Processing is very slow in Docker on a Mac

Containers on macOS cannot use the Apple GPU (MPS) or the VideoToolbox
encoder: detection and 8K encoding fall back to the CPU. Install natively on
macOS (see [Installation](../getting-started/installation.md)).

## Why is my 360° file rejected?

Only equirectangular videos are supported in v1. GoPro `.360` files (EAC
layout) and Insta360 `.insv` files (dual fisheye) must first be exported to an
equirectangular MP4 with the camera vendor's software. See
[ADR-0007](../adr/0007-360-video.md).
