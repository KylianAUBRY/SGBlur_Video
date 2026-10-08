# `detections.jsonl` format (version 1)

> Status: **accepted** (step 2), implemented in step 4 (`sgblur_video.core.detections_io`).

`detections.jsonl` is the contract between pass 1 (analysis) and the rest of
the pipeline. It is written by the in-process analyser **and** streamed by the
Detect API (`POST /detect/`, media type `application/x-ndjson`), so both paths
produce byte-compatible files.

Design rules:

- one JSON object per line (UTF-8, `\n`), written incrementally, so a file can be streamed and a truncated file is detectable;
- every line has a `type`: exactly one `header` first, one `frame` line per decoded frame in decoding order, one `footer` last;
- readable keys rather than compact arrays: a 10-minute 30 fps video is ~18 000 lines, so size is not a concern, and contributors can read it with `jq`;
- boxes are `[x1, y1, x2, y2]` floats in **coded-frame pixels** (before display rotation), origin top-left; for equirectangular video `x2` may exceed `width` when a box wraps around the 0°/360° seam;
- no pixels, file names or paths are ever written.

A consumer must refuse a file whose `version` it does not know, and treat a
file without `footer` as incomplete (the renderer refuses it unless
`--allow-partial` is given on the CLI).

## Header

```json
{
  "type": "header",
  "schema": "sgblur-video/detections",
  "version": 1,
  "created_at": "2026-10-06T14:02:11Z",
  "video": {
    "width": 7680,
    "height": 3840,
    "rotation": 0,
    "time_base": "1/90000",
    "avg_frame_rate": "30/1",
    "variable_frame_rate": false,
    "frame_count": 2893,
    "duration_s": 96.43,
    "codec": "hevc",
    "pix_fmt": "yuv420p10le",
    "projection": "equirectangular",
    "projection_source": "spherical-metadata"
  },
  "model": {
    "name": "yolo26s",
    "version": "0.1.0",
    "sha256": "9efa3df0c719713c79151f55fb5b3b694d168ee793901ef8f11dda753e7152a7",
    "classes": ["direction", "sign", "plate", "face"],
    "policy": {"direction": "annotate", "sign": "annotate", "plate": "blur", "face": "blur"}
  },
  "detection": {
    "profile": "standard",
    "conf": 0.10,
    "equirect_pad_px": 480,
    "passes": [
      {"id": "g1024", "kind": "global", "imgsz": 1024},
      {"id": "g2048", "kind": "global", "imgsz": 2048},
      {"id": "tL", "kind": "tile", "imgsz": 3840, "region": [-480, 960, 4320, 2880]},
      {"id": "tR", "kind": "tile", "imgsz": 3840, "region": [3360, 960, 8160, 2880]}
    ]
  },
  "tracking": {
    "tracker": "flow",
    "config": "configs/trackers/flow.yaml",
    "config_sha256": "…",
    "track_width": 1920,
    "groups": {"face": ["face"], "plate": ["plate"], "signage": ["sign", "direction"]}
  },
  "software": {
    "sgblur_video": "0.1.0",
    "ultralytics": "8.4.173",
    "torch": "2.14.1",
    "av": "19.0.1",
    "ffmpeg": "9.0.2",
    "device": "mps"
  }
}
```

`region` is `[x1, y1, x2, y2]` in coded-frame pixels, negative or beyond the
width when it includes circular padding.

## Frame

```json
{"type": "frame", "index": 431, "pts": 1293000, "time": 14.367,
 "detections": [
   {"class": "face", "score": 0.412, "box": [5120.5, 1810.0, 5161.2, 1866.4], "track_id": "face:7", "passes": ["tL"]},
   {"class": "plate", "score": 0.183, "box": [7650.0, 2101.3, 7712.8, 2122.0], "track_id": null, "passes": ["g2048", "tR"]},
   {"class": "sign", "score": 0.874, "box": [2201.0, 1502.5, 2290.4, 1590.1], "track_id": "signage:3", "passes": ["g2048", "tL"]}
 ]}
```

| Field | Type | Meaning |
|---|---|---|
| `index` | int | 0-based decoding order index |
| `pts` | int | presentation timestamp in `video.time_base` units, copied from the stream |
| `time` | float | `pts × time_base − start_time`, seconds |
| `detections[].class` | str | class name (never an index) |
| `detections[].score` | float | detector confidence after cross-pass merge (3 decimals) |
| `detections[].box` | float[4] | merged box (union for `blur` classes), coded-frame pixels |
| `detections[].track_id` | str \| null | `"<group>:<id>"`, unique within the file; `null` = orphan |
| `detections[].passes` | str[] | passes that produced the merged box (debugging, benchmark) |

Frames without detections are still written (`"detections": []`) so that the
frame count and timestamps can be checked against the video in pass 2.

## Footer

```json
{"type": "footer", "frames": 2893, "complete": true, "elapsed_s": 958.2,
 "counts": {"face": 812, "plate": 2210, "sign": 1505, "direction": 12, "orphans": 344}}
```

`complete: false` is written when analysis stops early (cancel, CLI
`--max-frames`).

## Evolution rules

- Adding optional fields keeps `version: 1`; consumers ignore unknown fields.
- Renaming/removing fields or changing units bumps `version`.
- The format is defined by the Pydantic models of `sgblur_video.core.detections_io`, and the reader rejects unknown versions. A published JSON Schema is possible future work (not generated today).
