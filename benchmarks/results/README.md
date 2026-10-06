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

### Trackers (`2026-10-06-trackers-clips.json`)

Same detections (YOLO26s, `standard`), tracking replayed per tracker on the
three dataset clips (10 s each): `gopro-pietons` (1080p, pedestrians close to
the camera), `q360-ville` and `q360-fin` (8K 360°, city streets and a market
square). `tracktrack-nogmc` is TrackTrack without camera-motion compensation.

| Tracker | Tracked share (gopro / q360-ville / q360-fin) | Blurred chains after linking | Signs | Tracking fps (1080p / 8K) |
|---|---|---|---|---|
| TrackTrack (default) | 70.7 % / 11.9 % / 2.8 % | 30 / 695 / 534 | 0 / 20 / 34 | — (ran during analysis) |
| TrackTrack, no GMC | 70.7 % / 11.6 % / 2.2 % | 30 / 693 / 534 | 0 / 20 / 36 | 1585 / 65 |
| BoT-SORT | 71.4 % / 10.7 % / 2.2 % | 26 / 682 / 534 | 0 / 19 / 34 | 64 / 38 |
| ByteTrack | 70.0 % / 10.2 % / 1.9 % | 27 / 681 / 532 | 0 / 19 / 36 | 1714 / 67 |

- On 8K 360° footage, **every tracker follows only 2–12 % of the detections**:
  the model finds small objects on about one frame in two (median fill ratio
  0.54 inside linked chains) and their merged boxes jump between passes
  (median IoU 0.52 between consecutive detections). Trackers drop a new track
  that is not matched on the very next frame.
- Offline linking (ADR-0011), interpolation and padding carry continuity: the
  blur plans of the four trackers differ by less than 3 %.
- Camera-motion compensation costs ~15 ms per frame (1080p) and brings no
  measurable gain here; it is kept because its cost is small next to detection
  (0.43 s per 8K frame).
- Leakage per tracker needs the annotated ground truth (pending).
