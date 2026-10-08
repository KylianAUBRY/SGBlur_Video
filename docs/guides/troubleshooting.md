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
(`configs/trackers/botsort.yaml`) are relative to the current
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

## A job fails with `worker_crash` ("killed by the system, most likely for lack of memory")

The operating system killed the job process because it ran out of memory. It
happens with 8K video when detection and encoding fall back to the CPU (Docker on
a Mac, Linux without NVIDIA GPU): an 8K 10-bit frame weighs about 90 MB, and
every frame x265 holds ahead costs about 300 MB. Above 4K, x265 therefore keeps
only 3 frames of lookahead, 1 B-frame and 1 reference, and detection on CPU runs
the 360° tiles one at a time: an 85-frame range of an 8K 10-bit 360° video, with
the debug video, peaked at 5.7 GB in a container limited to 7 GB.

- On a Mac, run natively (`uv run sgblur-video serve`): the hardware encoder
  needs little memory and is much faster.
- With Docker, give it more memory (Docker Desktop → Settings → Resources) and
  check `docker inspect <container> --format '{{.State.OOMKilled}}'`. The default
  Docker Desktop VM (7.7 GB) is enough for 8K with nothing else running; allow
  10–12 GB to keep a margin.
- Send a shorter frame range, or a lower-resolution video.

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
