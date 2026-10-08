# Benchmark results

Reports behind the project's decisions: numbers only, no picture, no file name.
Clip ids refer to the maintainer's private dataset (`docs/guides/benchmarks.md`).

## 2026-10-06 — step 8 (Apple M4 Pro, 24 GB, macOS, sgblur-video 0.1.0.dev0)

### Speed (`2026-10-06-speed-8k-equirect-m4pro.json`)

8 timed frames (median) of an 8K (7680×3840) equirectangular HEVC city video.
Detection and cross-pass merge only. Counts are detections with score ≥
`CONF_BLUR` per frame: **not** a quality measure (see the privacy benchmark).

| Model | Device | Profile | s/frame | Faces/frame | Plates/frame |
|---|---|---|---|---|---|
| YOLO26s | MPS | fast | 0.06 | 1.4 | 1.0 |
| YOLO26s | MPS | **standard** | **0.43** | **4.5** | **5.1** |
| YOLO26s | MPS | thorough | 1.41 | 4.9 | 5.0 |
| YOLO26s | CPU | fast | 0.19 | 1.4 | 1.0 |
| YOLO26s | CPU | standard | 1.88 | 4.5 | 5.1 |
| YOLO26s | CPU | thorough | 7.11 | 4.9 | 5.0 |
| YOLO11s | MPS | standard | 0.40 | 2.1 | 5.0 |
| YOLO11s | MPS | thorough | 38.2 | 2.1 | 5.0 |
| YOLO11s | CPU | standard | 1.71 | 2.1 | 5.0 |

- `fast` finds about 3× fewer faces than `standard` on 8K: not privacy-safe at
  this resolution.
- `thorough` (full-height tiles) adds ~8 % faces for 3.2× the time.
- YOLO11s is as fast but reports half as many faces as YOLO26s (and has no
  `direction` class); YOLO11s with 4096 px full-height tiles is pathologically
  slow on MPS (38 s/frame). YOLO26s stays the default until the privacy
  benchmark says otherwise.
- No NVIDIA GPU was available: CUDA is not measured.

### Apple Silicon speed-ups, 2026-10-07

Analysis of 90 frames of the 8K equirectangular city video, YOLO26s `standard`,
end to end (decode, detection, merge, tracking, `detections.jsonl`), runs made
back to back (absolute numbers vary with the machine's temperature and load):

| Change | Frames/s |
|---|---|
| Before: FP32, 360° tiles cropped with an index array, stages one after the other | 2.1 |
| + tiles cropped with slices, FP16 on MPS (`HALF=auto`) | 3.55 |
| + preparation and tracking overlapped with inference (three-stage pipeline) | **3.9** |

Detections are identical with and without the faster crop and the pipeline (90
frames). FP16 kept every detection above `CONF_BLUR` of FP32 on 60 busy frames
(514 faces, 146 plates, 91 signs). In steady state, 78 % of a frame is model
inference, 70 % for the two 4096 px tiles alone. Core ML (Neural Engine) could
not be tried: `coremltools` has no wheels for Python 3.14 yet.

### Registered models compared, 2026-10-07

Same 8K video, MPS (FP16), `standard` profile, 12 timed frames (median);
detections with a score ≥ `CONF_BLUR` per frame (counts, **not** recall).

| Model | s/frame | Faces | Plates | Signs |
|---|---|---|---|---|
| **yolo26s** (default) | 0.25 | **4.1** | 5.1 | 2.6 |
| yolo11n | 0.11 | 2.2 | 6.1 | 1.2 |
| yolo11s | 0.23 | 2.4 | 5.0 | 2.8 |
| yolo11m | 0.53 | 2.8 | 5.0 | 2.0 |
| yolo11l | 0.68 | 3.3 | 5.0 | 2.5 |

YOLO26s reports the most faces, 23 % more than the largest YOLO11 at 2.7× its
speed, and is the only one with the `direction` class: it stays the default.
The privacy benchmark on annotated clips will tell whether the extra faces are
real.

### Trackers against another model, 2026-10-07

Cached YOLO26s detections of four clips re-tracked, default post-processing.
"Covered" counts the faces and plates found by YOLO11l (score ≥ 0.4) on every
frame that fall inside a blur region (a proxy until the clips are annotated).
Details and sweeps: [ADR-0003](../../docs/adr/0003-default-tracker.md#optical-flow-tracker-2026-10-07).

| Clip | Tracker | Plate chains | Plate area | Plates covered | Face chains | Face area | Faces covered |
|---|---|---|---|---|---|---|---|
| parking (8K 360°, 38 frames) | TrackTrack | 107 | 3.25 % | 90/92 | 16 | 0.32 % | 2/3 |
| | **flow** | 45 | 2.66 % | 90/92 | 11 | 0.46 % | 2/3 |
| q360-ville (8K 360°, 300) | TrackTrack | 232 | 1.24 % | 558/591 | 463 | 2.75 % | 587/647 |
| | **flow** | 96 | 1.20 % | 560/591 | 178 | 2.02 % | 588/647 |
| q360-fin (8K 360°, 300) | TrackTrack | 367 | 2.70 % | 863/881 | 167 | 0.70 % | 116/142 |
| | **flow** | 115 | 2.03 % | 866/881 | 82 | 0.77 % | 120/142 |
| gopro-pietons (1080p, 300) | TrackTrack | — | — | — | 30 | 20.74 % | 551/644 |
| | **flow** | — | — | — | 14 | 19.37 % | 553/644 |

Same frames and model, SGBlur's per-picture detection finds 55 plates on 13
frames of the parking sample and sgblur-video 70 (54 in common, 13 of the
extra ones below SGBlur's 0.30 threshold): the blurred area differs because of
the temporal post-processing (padding, margin), not of the detection.
