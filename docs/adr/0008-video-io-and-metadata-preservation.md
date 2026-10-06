---
status: proposed
date: 2026-10-06
---

# Video I/O with PyAV and MP4 box post-processing

## Context and Problem Statement

The video must be re-encoded (no equivalent of SGBlur's lossless JPEG trick
for H.264/HEVC) while preserving timestamps (VFR), audio, container metadata,
telemetry (GPS) and spherical metadata. How do we decode, encode and remux?

## Decision Drivers

- Exact timestamps (VFR phones, audio and GPS sync).
- 10-bit HEVC at 8K, hardware encoders when present (VideoToolbox, NVENC).
- Preservation of the GoPro GPMF track; honest reporting of what cannot be kept.
- No unblurred pixel may survive in the output (previews, thumbnails, original samples).
- Minimal external tooling.

## Considered Options

1. PyAV for decode + encode + mux in one pass, then a pure-Python MP4 box editor for what FFmpeg cannot write.
2. Raw frames piped to an `ffmpeg` subprocess, then `ffmpeg -c copy` remux.
3. Re-encode with PyAV, then "replace the video track" in the original container with GPAC MP4Box.

## Decision Outcome

Chosen option: **1**.

Facts verified on 2026-10-06 (PyAV 19.0.1 / FFmpeg 9.0.2, local files):

- PyAV wheels include `libx264`, `libx265`, `hevc_videotoolbox`, `h264_videotoolbox` (and NVENC on Linux x86_64); no `ffmpeg` binary is needed.
- Copying audio and the GoPro `gpmd` stream with `add_stream_from_template` works; the GoPro `tmcd` stream (codec "none") makes the MP4 header fail (`Could not find tag for codec none`) and must be skipped; the muxer then recreates a timecode track from the video stream's `timecode` metadata.
- GoPro `udta` boxes (`FIRM`, `LENS`, `GPMF`, …) and the Q360 Spherical V1 `uuid` box are not written by FFmpeg: they need box-level copying.
- PyAV tags HEVC as `hev1` by default; the source uses `hvc1`.

Design:

- One render loop demuxes the original: video packets are decoded, blurred, encoded; audio/subtitle/`gpmd` packets are copied with their timestamps; codec-"none" streams are dropped and reported in `stats.dropped_streams`.
- The muxer writes `moov` after `mdat`; `core/mp4boxes.py` then rewrites `moov` (no sample offset changes) to transplant, **by allow-list**: Spherical V1/V2 boxes, the video `tkhd` matrix, safe `udta` children (GoPro settings and GPMF, QuickTime location/date). Thumbnails, previews, trailers and unknown boxes are never copied.
- Encoder: same codec family as the source; hardware encoder when available; bit rate matched to the source; colour properties and codec tag copied.
- Telemetry formats that FFmpeg cannot mux (CAMM, DJI `djmd`, Sony `rtmd`, Apple `mebx`, Insta360 trailer, Novatek `gps `) are **not supported in v1**: they are dropped (and reported); supporting them means writing their sample tables ourselves, tracked as future work.

### Consequences

- Good: single dependency for media (PyAV), exact timestamps, no intermediate file, no pipe bandwidth (≈ 25 MB per 4K RGB frame).
- Good: allow-list transplanting guarantees no unblurred preview or original sample is carried over.
- Bad: we own a small MP4 box editor; it is limited to `moov` rewriting (no offset fix-ups), unit-tested on generated files and checked on real files in integration tests.
- Bad: no fast-start (`moov` at the end) in v1; acceptable for download use, can be added with chunk-offset rewriting.
- Option 2 rejected: raw pipes lose timestamps (VFR drift) and still need a remux; option 3 adds a GPL tool and keeps the original `mdat` layout, which risks carrying original samples.
