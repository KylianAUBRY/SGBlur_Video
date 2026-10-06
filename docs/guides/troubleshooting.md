# FAQ and troubleshooting

## `ModuleNotFoundError: No module named 'sgblur_video'` (macOS, iCloud Drive)

When the project lives in a folder synced by iCloud Drive (by default
`~/Desktop` and `~/Documents`), iCloud marks the content of dot-directories
such as `.venv` as *hidden*. Python 3.14 ignores hidden `.pth` files, so the
editable install of `sgblur_video` disappears. Keep the virtualenv in a
non-hidden folder and point `.venv` to it:

```bash
mv .venv venv && ln -s venv .venv && chflags -R nohidden venv
```

(or clone the project outside iCloud-synced folders). The test suite is not
affected either way: pytest adds `src/` to the import path itself.

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
