# Pipeline design

> Status: **accepted** (step 2), implemented in steps 4–8; implementation notes
> are inline. Numbers in *italics* are defaults that remain provisional until
> the privacy benchmark on annotated real clips ([testing-strategy.md](testing-strategy.md)).

## 0. Vocabulary

| Term | Meaning |
|---|---|
| **Class** | A model output label: `face`, `plate`, `sign`, `direction`. Always referenced **by name**: the SGBlur YOLO26 model uses the order `direction, sign, plate, face`, YOLO11 used `sign, plate, face`. |
| **Policy** | What we do with a class: `blur` (`face`, `plate`) or `annotate` (`sign`, `direction`). A model missing a `blur` class is refused at start-up (fail closed). |
| **Class group** | Classes merged together across passes: `face`, `plate`, `signage` (= `sign` + `direction`, because the model can hesitate between both on the same object). Only `signage` is tracked. |
| **Detection** | One box from the detector on one frame, after cross-pass merging. |
| **Track** | Sign detections linked by the tracker under one `track_id`. Faces and plates are never tracked ([ADR-0012](../adr/0012-independent-frames.md)). |
| **Coded frame** | The frame as stored in the stream (before display rotation). All boxes in `detections.jsonl` and the blur plan use coded coordinates; annotations use display coordinates. |

## 1. Probe and validation

`core/probe.py` opens the file with PyAV and builds a `VideoInfo`:
container, video codec and tag (`hvc1`/`avc1`…), `pix_fmt`, colour properties,
width/height, display rotation, `time_base`, average and real frame rate
(VFR detection), frame count (exact count from the index when available),
duration, bit rate, audio/data/subtitle streams with handler names and tags,
spherical side data, telemetry kind (GPMF `gpmd`, CAMM, none).

Rejections (HTTP 415/422, CLI exit code 1 with a one-line message):

| Case | Detection | Reason |
|---|---|---|
| No video stream / undecodable | PyAV | — |
| Duration > `MAX_VIDEO_DURATION_S` | container duration | resource limit |
| GoPro MAX `.360` | two video streams + GoPro handlers, or udta GPMF `PRJT=EACO` | custom EAC layout, not supported in v1 |
| Insta360 `.insv` | extension or trailer magic `8db42d69…` | unstitched dual fisheye; also embeds an unblurred JPEG preview |
| Spherical side data with projection ≠ equirectangular | side data | cubemap/mesh not supported in v1 |

Projection: `PROJECTION=auto` uses spherical side data (V1 or V2); without
metadata, an exact 2:1 aspect ratio is treated as equirectangular (heuristic,
logged as such). `flat` / `equirectangular` force it.

## 2. Pass 1 — analysis

### 2.1 Decoding

Frames are decoded sequentially with PyAV (software decoding: faster than
VideoToolbox once the RGB conversion is counted, and identical on every
platform). For each frame we keep `index`, `pts` and `time` (`pts × time_base`):
timestamps are never recomputed from a nominal frame rate, so VFR videos stay
aligned with audio and GPS.

The detector input is produced with `frame.reformat(width, height, "bgr24")`
(swscale) at the resolution each pass needs, so the 8K frame is converted to
full-resolution BGR only when a tile pass needs it. If the stream has a display
rotation, detector inputs are rotated upright (the model expects upright faces)
and boxes are mapped back to coded coordinates.

### 2.2 Detection plan

`core/detect.py` builds a plan once per video from its size and projection.
It reproduces SGBlur's multi-scale logic, adapted to video:

| Pass | Input | `imgsz` | When |
|---|---|---|---|
| `g1024` | full frame | 1024 | always (large, close objects) |
| `g2048` | full frame | min(2048, long side rounded to 32) | when long side > 1024 |
| `tiles` | equirect: left and right halves of the middle band (`h/4 … 3h/4`), each with circular overlap; flat: grid of ≈ 3840 px tiles with 10 % overlap | tile long side, capped at 4096 | long side ≥ `TILE_TRIGGER_WIDTH` (*5760*, SGBlur value) |

`DETECT_PROFILE`: `standard` (table above, default), `fast` (global passes
only), `thorough` (tiles cover the full height, for nadir/zenith).

Measured on 8 frames of an 8K equirectangular city video, Apple M4 Pro, YOLO26s
(`benchmarks/results/2026-10-06-speed-8k-equirect-m4pro.json`): `fast` runs at
0.06 s/frame (MPS) but finds about **3× fewer faces** than `standard`
(1.4 vs 4.5 per frame at score ≥ 0.15, the blur threshold at the time); `standard` takes 0.43 s/frame
(MPS) or 1.9 s/frame (CPU); `thorough` adds about 8 % more faces for 3.2× the
time. `fast` is therefore not a privacy-safe choice for 8K video.

**360° circular padding.** For equirectangular frames each pass input is padded
horizontally by `P = round(w × EQUIRECT_PAD_RATIO)` (*1/16*: 480 px at 8K)
with pixels copied from the opposite edge, so an object straddling the 0°/360°
seam is seen whole at least once. Boxes are then mapped to canonical
coordinates: `x1 ∈ [0, w)`, and `x2` may exceed `w` for a box that wraps.

Inference uses `conf = CONF_DETECT` (*0.30*, SGBlur's `MIN_CONF`), the model's default NMS head
(`nms=None`), `quantize=16` on CUDA only, and `classes` restricted to classes
with a policy.

### 2.3 Cross-pass merge

All boxes of all passes for one frame are merged per class group with a greedy,
score-ordered procedure, using wrap-aware geometry for equirect. The cluster
keeps the best score and class.

- `blur` groups follow SGBlur: two boxes are duplicates if IoU > *0.33* or if
  the smaller one is ≥ *80 %* inside the larger (IoMin), and the cluster keeps
  its **smallest** box.
- `signage`: duplicates if IoU ≥ *0.5* or IoMin ≥ *80 %*; the cluster keeps the
  box of the best-scoring member (annotation quality).

Unlike SGBlur, a face or plate is never merged with a sign (SGBlur merges
across classes, so a plate overlapping a larger sign box could be dropped).

### 2.4 Sign tracking

Only signs are tracked, to produce one annotation per physical sign (§3.1);
faces and plates are never tracked, and without any `annotate` class no
tracker runs at all. The tracker runs on a downscaled frame. The default is
the optical-flow tracker
(`core/flowtrack.py`, `tracker_type: flow`,
[ADR-0003](../adr/0003-default-tracker.md)): each track's box is moved by the
median Lucas-Kanade flow of a grid of points over the box and one box size
around it (forward-backward checked), then tracks and detections are matched
by centre distance (optimal assignment, distances wrapped around the 360°
seam), with a gate of 2 box sizes for boxes up to 32 px and half a box size
beyond. Every detection receives an id (new track if unmatched). The
Ultralytics trackers are instantiated directly (`TRACKTRACK(args)`,
`BOTSORT(args)`…, see
[ADR-0002](../adr/0002-own-detection-loop-with-ultralytics-trackers.md)); the
notes below apply to them.

- **Tracking space**: frame scaled to `TRACK_WIDTH` (*1920* px wide); merged boxes are scaled into it; the same downscaled BGR frame is passed as `img` for global motion compensation (GMC `sparseOptFlow`, which needs every frame at a constant size).
- Input: `ultralytics.engine.results.Boxes([[x1, y1, x2, y2, score, cls]], shape)`; never with an `xywhr` attribute.
- Output rows `[x1, y1, x2, y2, track_id, score, cls, idx]`: `idx` links the track back to our merged detection, which receives `track_id = "<group>:<id>"`. The Kalman box is not used for blurring; observations keep the detector box.
- Sign detections the tracker does not return keep `track_id = null` and are linked offline (§3.1).
- `track_buffer` is a frame count in Ultralytics ≥ 8.4.38: our tracker YAMLs express it in seconds (`track_buffer_s`), converted with the video frame rate before the tracker is built.
- Track ids come from a class attribute shared by every tracker in the process (`BaseTrack._count`): isolation between jobs is guaranteed by running each job in its own process.

### 2.5 Output

Each frame is appended to `detections.jsonl` as soon as it is processed
(see [detections-format.md](detections-format.md)), with progress reported to
the job store at most once per second.

### 2.6 Overlapping CPU and accelerator work

*Implemented after step 9.* Pass 1 runs in three stages connected by bounded
queues: a thread decodes frames, converts them to BGR and prepares the detector
inputs (360° padding, tile crops: `YoloDetector.prepare`) and the sign-tracking
image; the calling thread runs the inference passes (`YoloDetector.infer`) and
the cross-pass merge; a thread runs the sign tracker and writes `detections.jsonl`
in frame order. At most two prepared frames wait for inference (about 150 MB
each at 8K). An error in a stage stops the pipeline and is re-raised. Output is
identical to running the stages in sequence; 8K analysis is about 10 % faster
on an Apple M4 Pro (3.55 → 3.9 frames/s).

## 3. Blur plan and sign annotations (no GPU)

`core/postprocess.py` reads `detections.jsonl` and produces a **blur plan**:
for each frame, the rectangles to blur. `semantics/annotations.py` produces the
**sign annotations** (§3.1).

Every frame is blurred on its own detections, like SGBlur blurs a picture
([ADR-0012](../adr/0012-independent-frames.md)):

- every `face` and `plate` detection of the frame with a score ≥ `CONF_DETECT`
  is blurred (the plan checks the score again, so a `detections.jsonl` produced
  with a lower threshold blurs the same);
- with a **rectangle exactly on the detected box** (no margin);
- boxes smaller than *12* px on a side are skipped (SGBlur value);
- nothing is carried from one frame to the next: no tracking, interpolation,
  padding or smoothing.

```
frame          0    5    10   15   20   25
detections     .    .    ███ ██  ·  ███████     · = frame missed by the detector
blurred        .    .    ███ ██  ·  ███████     the miss stays visible, as on a picture
```

A face or plate the detector misses on a frame is therefore visible on that
frame. The privacy benchmark on annotated clips measures how often and how
long ([testing-strategy.md](testing-strategy.md)).

On 360° video, a box may cross the seam (`x2 > w`); the renderer blurs its part
on each side.

### 3.1 Sign deduplication and annotations

For each `signage` track:

0. Sign fragments (tracker tracks, and detections the tracker left alone) are linked offline ([ADR-0011](../adr/0011-offline-linking.md)): one starting at most `LINK_MAX_GAP_S` (*1 s*) after a chain ends, within `LINK_MAX_DISTANCE` (*1* box size, +10 % per frame of gap) of where the chain was heading, joins it, so a sign detected intermittently is still one sign. Distances are measured around the 360° seam.
1. Drop it if it has fewer than `SIGN_MIN_TRACK_LENGTH` (*5*) observations or a max score below `CONF_SIGN` (*0.6*, SGBlur value).
2. **Best frame** = detection maximising `score × box area`; the class reported is the majority class of the track.
3. Emit one annotation (format in [api.md](api.md#metadata)): shape = best-frame box in display coordinates, integer pixels, clipped to the frame (for a box that wraps around the seam, the larger part is kept); semantics identical to SGBlur (`osm|traffic_sign=yes`, `detection_model[…]`, `detection_confidence[…]`, string values) for both `sign` and `direction` (maintainer decision, until SGBlur defines specific tags); `video` extension object with track id, first/best/last frame and timestamp, max/mean confidence, and the GPS position at the best timestamp when telemetry provides one.

Future work (documented, not implemented): a second-stage classifier of sign
types (e.g. Panoramax `classified_fr_road_signs`, Prolix-style tags
`osm|traffic_sign=FR:A15b`) running on the best-frame crop.

## 4. Pass 2 — rendering

### 4.1 Decode, blur, encode, mux in one loop

`core/render.py` demuxes the original once more:

- **video packets** → decoded (software) → blurred → encoded → muxed;
- **audio, subtitle and supported data packets** (`gpmd`) → muxed unchanged (stream copy), keeping their timestamps;
- streams that FFmpeg cannot mux into MP4 (codec "none": `tmcd`, `fdsc`, `camm`, `djmd`, `rtmd`, `mebx`) are dropped and listed in the job statistics. The GoPro timecode track is **recreated** by the muxer from the video stream's `timecode` metadata (verified on a HERO12 file).

Decoded frames are never modified in place (their buffers can be reference
frames of the decoder): planes are copied to NumPy, blurred, and wrapped in a
new frame carrying the original `pts`.

### 4.2 Blur methods

Applied directly on the YUV planes (luma at full resolution, chroma at the
subsampled resolution), in the source bit depth
(8 or 10 bit): no RGB round-trip, no colour shift, half the memory traffic.

| `BLUR_METHOD` | Operation | Notes |
|---|---|---|
| `pixelate_blur` (default) | Area-average down to at most `PIXELATE_CELLS` (*6*) cells on the box's long side, smooth at that resolution, interpolate back bilinearly | Only ≤ 6×6 averages survive. Mosaics with many cells can be partly re-identified by machine learning (McPherson et al., 2016, arXiv:1609.00408), hence few cells and no block edges. |
| `gaussian_strong` | Gaussian blur with σ = long side / 4, plus low-amplitude noise | Kept for users who prefer the look; weaker than mosaic against deconvolution. |
| `solid` | Constant neutral grey | Maximum guarantee, least pleasant. |

Known limitation: averaging many frames of the same blurred face could leak a
little information (multi-frame attacks); `solid` removes this risk.

### 4.3 Encoding

`ENCODER=auto` keeps the source codec family (H.264 → H.264, HEVC → HEVC) and
picks, in order: VideoToolbox (macOS), NVENC (CUDA available), then
`libx264`/`libx265`. Resolution, `pix_fmt` (including 10-bit), colour
range/primaries/transfer/matrix, `time_base` and every frame `pts` are copied
from the source; the codec tag (`hvc1`/`avc1`) is set to the source's (PyAV
defaults HEVC to `hev1`, which Apple players reject). Target bit rate =
source video bit rate × `ENCODE_BITRATE_FACTOR` (*1.0*) for hardware encoders;
CRF *20* capped at that bit rate for software encoders.

### 4.4 MP4 box post-processing

The muxer writes `moov` after `mdat`, so `core/mp4boxes.py` can rewrite `moov`
without touching sample offsets. From the original file it transplants, by
**allow-list only**:

- spherical metadata: V1 `uuid ffcc8263-…` box copied verbatim into the video `trak`, V2 `sv3d`/`st3d` copied into the sample entry (FFmpeg never writes V1 and writes V2 only with `-strict unofficial`);
- the video `tkhd` display matrix (rotation);
- `udta` children known to be safe (GoPro `FIRM`, `LENS`, `CAME`, `SETT`, `HMMT`, `GPMF`…; QuickTime `©xyz`/`©day`);
- global `creation_time` (also set through container metadata).

Never copied: thumbnails and previews (QuickTime `ThumbnailImage`/`PreviewImage`,
Insta360 trailer), unknown `uuid` boxes, anything inside `mdat`.

## 5. Telemetry and positions

`telemetry/gps.py` exposes `position_at(t) -> (lat, lon, alt) | None` by linear
interpolation of the GPS track (no extrapolation beyond 1 s). v1 reads GoPro
GPMF (`GPS5`/`GPS9`, samples without a 2D/3D fix ignored) from the `gpmd`
stream with a small built-in KLV parser (`telemetry/gpmf.py`); other formats
are listed as unsupported. Positions are used for sign annotations and best-frame JPEG EXIF
only, and the GPS track itself is preserved in the output through the copied
`gpmd` stream.

The test GoPro files available locally are HERO12 recordings: their GPMF
contains IMU data (`ACCL`, `GYRO`, `CORI`) but **no GPS** (HERO12 has no GPS
receiver), so GPS tests rely on the public `gopro/gpmf-parser` samples.

## 6. Best frames (`/frames`)

When `frames=1` (API) or `--frames-dir` (CLI) is requested, pass 2 saves, for
each distinct best frame of the sign annotations, the **blurred** output frame
as a JPEG (signs sharing a best frame share a picture; `frames.json` lists
them) (quality *92*, display
orientation, EXIF date from `creation_time + t`, GPS when available), with the
annotation shape expressed in that JPEG's pixel coordinates. These files can be
uploaded to today's Panoramax with `isBlurred=true` and per-file annotations
(backend ≥ 2.16).
