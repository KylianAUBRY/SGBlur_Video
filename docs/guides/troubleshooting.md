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

## `invalid model registry models/registry.yaml: No such file or directory`

The defaults of `MODELS_FILE` (`models/registry.yaml`) and `TRACKER_CONFIG`
(`configs/trackers/tracktrack-recall.yaml`) are relative to the current
folder. Run commands from the repository root, or set both to absolute paths
in the environment or in a `.env` file. The Docker image sets them for you.

## `objc: Class AVF… is implemented in both …` warnings (macOS)

PyAV and OpenCV each bundle their own FFmpeg libraries. The duplicated classes
belong to `libavdevice` (camera capture), which SGBlur-Video never uses; the
warning is harmless.

## Processing is very slow in Docker on a Mac

Containers on macOS cannot use the Apple GPU (MPS) or the VideoToolbox
encoder: detection and 8K encoding fall back to the CPU. Install natively on
macOS (see [Installation](../getting-started/installation.md)).

## Does the blurred 360° video still play as 360°?

Yes: the Spherical Video V1/V2 metadata of the original is copied back into
the output (`stats.restored_metadata` in the job metadata says what was
restored). If a player still shows a flat picture, check that the original
itself carried spherical metadata (`sgblur-video` logs `projection=…
(spherical-metadata)` when probing it).

## Where is the GPS of my video?

GoPro GPMF telemetry (HERO5–HERO11, MAX, and later models with GPS) is copied
unchanged into the blurred video and used to geolocate signs. HERO12 has no
GPS receiver. Other telemetry formats (CAMM, DJI, Insta360, dashcam formats)
are not supported in v1 and are listed in `stats.dropped_streams`.

## Why is my 360° file rejected?

Only equirectangular videos are supported in v1. GoPro `.360` files (EAC
layout) and Insta360 `.insv` files (dual fisheye) must first be exported to an
equirectangular MP4 with the camera vendor's software. See
[ADR-0007](../adr/0007-360-video.md).
