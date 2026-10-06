# Step 1 — Analysis (research notes)

> Date: 2026-10-06. Status: **validated**. The maintainer's answers to §9 are
> recorded in the ADRs (`docs/adr/`): MIT licence kept (ADR-0006), GitHub forge,
> Python 3.14, SGBlur `yolo26s` weights reused without training (ADR-0009, the
> model has 4 classes including `direction`), privacy tests A + D (ADR-0010).
> Where this file recommends otherwise, the ADRs win.
> These notes record what was verified before any design or code work, so that
> later ADRs can cite them. Facts marked **[V]** were verified in source code or
> official docs; **[I]** marks inferences; **UNVERIFIED** marks claims that could
> not be checked (nothing was executed against real camera files yet).

## 1. SGBlur in ten lines

Read from `gitlab.com/panoramax/server/sgblur`, branch `master`, commit `e39fb20` (2026-05-17).

1. Two FastAPI apps from one Docker image: blur API (`:8000`, CPU, `WEB_CONCURRENCY=8`) calls detect API (`:8001`, GPU, 1 worker) with `POST /detect/` (multipart field `picture`, optional `cls` filter). `DETECT_URL=""` runs detection in-process.
2. Custom YOLO11 model (n/s/m/l), classes `{0: sign, 1: plate, 2: face}`. Weights are committed in git (`models/*.pt`, ~140 MB). The `MODELS` registry (`name`, `version`, `path`) lives in `src/detect/detect.py`, not `config.py`; all versions are hard-coded `0.1.0`.
3. Auto model size: no CUDA → `yolo11n`; free VRAM < 6 GiB → `yolo11s`; otherwise `yolo11m`. Overridable with `MODEL_NAME`.
4. Three passes: `imgsz=1024`, `imgsz=2048`, and for width ≥ 5760 px the middle band (`h/4 → 3h/4`) split left/right at `imgsz=4096`. Custom class-agnostic merge (IoU > 0.33 → smaller box wins). No overlap at the split line, no 0°/360° seam handling.
5. Thresholds: `conf=0.30` for all classes at inference; signs kept only if ≥ 0.6.
6. Blurring in the JPEG DCT domain (MCU-aligned lossless crops + `jpegtran -drop`), so nothing outside blurred zones is recompressed. Effect: pixelate then box-blur, rectangle shape (no ellipse), no margin.
7. Response: multipart (`metadata` JSON part + `image` JPEG part) when the `Accept` header contains `multipart/form-data`; otherwise JPEG + `x-sgblur` header (legacy).
8. Metadata: `{annotations, service_name, blurring_id}`. Annotations only for signs; `shape` = absolute pixel bbox `[x1, y1, x2, y2]` (top-left origin); all semantic values are strings; model tag value is `f"{API_NAME}-{model}/{version}"`, e.g. `SGBlur-yolo11n/0.1.0`.
9. `keep=1`: original crops with confidence < 0.5 stored **unencrypted** under a `sha256(salt + info)` path; no TTL, no cleanup. `/deblur/` is broken (reads key `conf` instead of `confidence`). Undocumented `keep=2` = signs only, no blur.
10. Fully synchronous, whole files in RAM, no timeout on the detect call, possible infinite retry loop in `model_detect`. Code MIT; `ultralytics==8.3.121`; no tags, no CHANGELOG, no CODE_OF_CONDUCT in the repo.

### Conventions to reuse
- Two-service split (CPU blur/render vs GPU detect), `DETECT_URL`, `MODEL_NAME`, `API_NAME`, `TMP_DIR`, pydantic-settings.
- Multipart `image` + `metadata` selected by `Accept: multipart/form-data`.
- Metadata `{annotations, service_name, blurring_id}`; tags `osm|traffic_sign=yes`, `detection_model[osm|traffic_sign=yes]`, `detection_confidence[osm|traffic_sign=yes]`; model string `{API_NAME}-{model}/{version}`; integer pixel xyxy shapes; string values.
- Class names and order `sign, plate, face`.

### Things not to reproduce
- Sign crops are always saved to disk **with GPS EXIF**, no TTL (silent dataset collection).
- Detection info written into a JPEG COM segment of every output.
- Public `debug` flag on the API.
- Unencrypted `keep=1` crops, no TTL.
- Bug: the multipart `metadata` part is emitted with header `Content-Type: {'Content-Type': 'application/json'}` (urllib3 tuple misuse).
- Weights in git; unpinned torch; bare `except`.

## 2. Panoramax blur contract as actually implemented

Sources: docs `deep_dive/blur_api`, `deep_dive/semantics`, `tags/syntax`; backend `gitlab.com/panoramax/server/api` branch `develop` commit `50efe97` (2026-10-05).

- **[V]** `API_BLUR_URL` enables blurring; originals are not kept by the backend.
- **[V]** The backend (`geovisio/utils/pictures.py::createBlurredHDPicture`) sends `POST {API_BLUR_URL}/blur/`, file `picture` (`picture.jpg`, `image/jpeg`), **always** `Accept: multipart/form-data`, `keep=1` only if `PICTURE_PROCESS_KEEP_UNBLURRED_PARTS=true`. **No timeout.**
- **[V]** Any exception or non-2xx (including 4xx) becomes a retryable error, retried up to `PICTURE_PROCESS_NB_RETRIES` (default 5).
- **[V]** The response must be a JPEG with EXIF intact and orientation applied.
- **[V] `service_name` is ignored by the backend.** `runner_pictures.py` hard-codes `service_name="SGBlur"` and deletes previous annotation tags whose `detection_model[...]` value **starts with `SGBlur-`**. Cleanup is skipped when the new response has no annotations.
- **[V]** Annotation `shape`: bbox `[minx, miny, maxx, maxy]` (ints, top-left origin) or GeoJSON-like `Polygon`; must lie within the picture; `semantics` items `{key ≤ 256 chars, value ≤ 2048 chars}`, values must be strings.
- **[V]** Unknown top-level metadata keys are tolerated. **[I]** Unknown keys inside an annotation (e.g. a `video` object) are silently dropped by Pydantic (`extra="ignore"`), so they would not be stored. Keys `id`, `picture_id`, `account_id` must be avoided inside annotations.
- **[V]** `metadata` must be a JSON object (an array crashes the job).

## 3. Panoramax and video today

- **[V]** The backend has no video support (`FileType.picture` only); the CLI has no video features.
- **[V]** NLnet "Panoramax video" (NGI0 Commons Fund, started 2026-04) funds server-side processing of direct video contributions (frame extraction, GoPro and Qoocam metadata). <https://nlnet.nl/project/Panoramax-video/>
- **[V]** Issue `panoramax/server/api#369` (opened 2026-09-15 by cquest, open): "Implementing video support: how about storing them as... videos". Lists "adapt the blurring API to accept video and return blurred videos". Discussion leans towards equirectangular-only uploads for 360°. <https://gitlab.com/panoramax/server/api/-/work_items/369>
- **[V]** Older issues: `api#151` (video + GPX, 1 frame/s, low priority), `cli#40` (video upload), `cli#35` (GoPro `.360`).
- **[V]** Since backend v2.16.0 (2026-07-15), `POST /api/upload_sets/:id/files` accepts per-file `semantics` and `annotations`, which allows uploading pre-blurred frames (`isBlurred=true`) with sign annotations today.
- **[I]** The video contract does not exist yet upstream: our async video API is an extension that nothing consumes today. Engaging with api#369 early is the main mitigation.

## 4. Ultralytics

Checked against PyPI, `ultralytics` main at `1110d9d` (2026-10-06) and the 8.4.173 sdist (trackers and cfg identical).

### Versions and models
- **[V]** `ultralytics==8.4.173` (2026-10-04), AGPL-3.0, Python ≥ 3.8. Releases are almost daily; breaking changes happen in patch releases → **pin exactly**.
- **[V]** **YOLO26** is the latest released family (8.4.0, 2026-01-14). **YOLO27 is announced only** ("not yet available, no launch date"). Plan: YOLO26 now, registry entry for YOLO27 later.
- **[V]** YOLO26: DFL removed, dual head (one-to-many + one-to-one NMS-free), STAL (small-target-aware label assignment). `yolo26-p2.yaml` exists without pretrained weights.
- **[V]** Since 8.4.142 the `nms` argument selects the head: `None` (default) and `True` → one-to-many + NMS; `False` → NMS-free one-to-one. `end2end` is deprecated.
- **[V]** `half` is deprecated (8.4.80) in favour of `quantize=16`. MPS is used only with explicit `device="mps"`.
- COCO (640 px, docs): n 40.9 / s 48.6 / m 53.1 / l 55.0 / x 57.5 mAP50-95.

### Trackers
- **[V]** Files in `ultralytics/cfg/trackers/`: `botsort.yaml`, `bytetrack.yaml`, `deepocsort.yaml`, `fasttrack.yaml` (not `fasttracker`), `ocsort.yaml`, `tracktrack.yaml`. **TrackTrack is the Ultralytics default** since 8.4.76.
- **[V]** `tracktrack.yaml` defaults: `track_high_thresh 0.6`, `track_low_thresh 0.25`, `new_track_thresh 0.7`, `track_buffer 30`, `match_thresh 0.7`, `min_track_len 3`, `gmc_method sparseOptFlow`, `with_reid False`. These are precision-oriented.
- **[V]** Camera-motion compensation: BoT-SORT (`sparseOptFlow`), TrackTrack (`sparseOptFlow`), Deep OC-SORT (`none` by default). ByteTrack, OC-SORT, FastTracker have none.
- **[V]** `track_buffer` is a plain frame count (no fps scaling since 8.4.38) → must be scaled by fps on our side.

### `model.track()` behaviours that conflict with privacy goals
- **[V]** Boxes below `conf` are removed before the tracker; inside the tracker the low floor is `max(conf, track_low_thresh)`.
- **[V]** When at least one track is output, results are sliced to tracked boxes only: **unmatched / low-score / unconfirmed detections are dropped** (orphans lost).
- **[V]** New tracks are not emitted on their birth frame (ByteTrack/BoT-SORT), and TrackTrack waits for `min_track_len=3` matched frames.
- **[V]** Output coordinates are the Kalman state, not the raw detection; coasting (lost) tracks are never output.
- **[V]** Tracker state lives on `model.predictor.trackers`; `persist=True` never resets when the video changes; the predictor is silently rebuilt (state lost) if `device`, `nms`, `channels_last` or `quantize` change.
- **[V]** Track IDs come from the class attribute `BaseTrack._count`, reset globally by any tracker `__init__`/`reset()` → IDs collide across concurrent videos in one process.
- **[V]** Thread-safety guide: one `YOLO` instance per thread, or `ThreadingLocked`, or processes.
- **Conclusion [I]:** use our own detection loop and the tracker classes directly (needed anyway for multi-scale/tiling): every raw detection ≥ `CONF_BLUR` is blurred regardless of tracking; the tracker only adds association, gap filling and padding. One job per process.

### Direct tracker API (8.4.173)
- **[V]** `from ultralytics.trackers import BYTETracker, BOTSORT, OCSORT, DeepOCSORT, FASTTracker, TRACKTRACK`.
- **[V]** Constructor: `Tracker(args)` only (`frame_rate` removed in 8.4.49). `args = IterableSimpleNamespace(**YAML.load(check_yaml(path)))`.
- **[V]** `BYTETracker.update(results, img=None, feats=None, **kwargs) -> np.ndarray`; `TRACKTRACK.update(results, img=None, dets_del=None, **kwargs)`.
- **[V]** Input needs `.conf`, `.cls`, `.xywh`, `.xyxy`, `len()`, boolean-mask indexing; simplest is `ultralytics.engine.results.Boxes(np.array([[x1,y1,x2,y2,conf,cls], ...]), orig_shape)`. Never expose `xywhr` (switches to OBB mode).
- **[V]** Output `(N, 8)`: `[x1, y1, x2, y2, track_id, score, cls, idx]`; `idx` = input row, `-1` for TrackTrack recovered boxes.
- **[V]** GMC needs the full BGR frame on every `update()` at constant resolution (internal `downscale=2`), estimates a single partial affine. **[I]** Weak model for forward motion (zoom) and wrong for equirectangular wrap-around.

### Licence
- **[V]** Library: AGPL-3.0 (or Enterprise). <https://www.ultralytics.com/license> states trained/fine-tuned models fall under AGPL-3.0 by default and that SaaS/API deployment requires open-sourcing the "entire project" under AGPL-3.0 or an Enterprise licence. This is the vendor's position, not settled law.
- **[V]** Inconsistency upstream: SGBlur weights are published on Hugging Face (`Panoramax/detect_face_plate_sign`) as **etalab-2.0**, while the checkpoints embed `license: AGPL-3.0` metadata.

## 5. Video I/O, telemetry and 360°

Nothing in this section was executed (no ffmpeg/PyAV locally). Sources: FFmpeg master and `release/9.0`, PyAV v19.0.1, gpmf-parser, ExifTool, spatial-media.

### Tooling
- **[V]** FFmpeg 9.0.2 (2026-09-18). PyAV 19.0.1 (2026-10-03), BSD-3-Clause, **requires Python ≥ 3.12**, wheels bundle FFmpeg 9.0.2.
- **[V]** PyAV wheels link libx264/libx265 with a patched configure (reported as LGPLv3) — treat as GPL in practice; no `ffmpeg` binary included (`--disable-programs`) → the Docker image must ship an ffmpeg CLI for remuxing.
- **[V]** Wheels include NVENC/NVDEC (Linux/Windows x86_64) and VideoToolbox (macOS).
- **[V]** PyAV keeps `pts`/`time_base` (VFR-safe); raw-frame pipes to an ffmpeg subprocess lose timestamps and cost ~25 MB per 4K RGB frame. Decision candidate: PyAV for decode/encode, ffmpeg CLI for final remux.
- **[V]** Rotation: the ffmpeg CLI autorotates pixels when re-encoding; PyAV keeps raw frames (`frame.rotation`). Plan: blur in raw orientation, rotate only the detector input, restore the display matrix at remux.

### Telemetry preservation matrix (ffmpeg remux)

| Source | Storage | Survives ffmpeg remux to MP4? |
|---|---|---|
| GoPro GPS (`gpmd`, "GoPro MET") | data track, `bin_data` | **Yes**, if mapped explicitly by index (FFmpeg ≥ 4.1) |
| GoPro `tmcd` | timecode track (codec none) | No as-is; can be **recreated** from the `timecode` tag |
| GoPro `fdsc` (SOS) | codec none | No (safe to drop) |
| GoPro `udta/GPMF` (settings, HiLights) | udta box | No |
| CAMM (`camm`) | codec none | **No — no FFmpeg version can mux it** (patches never merged) |
| DJI `djmd`/`dbgi`, Sony `rtmd`, Apple `mebx` | codec none | No |
| Insta360 `.insv` trailer | bytes after last box | No |
| Novatek `gps ` / `freeGPS` | moov index into mdat | No |
| Apple ISO6709 location | `moov/meta` keys | Only with `-movflags use_metadata_tags` |
| Android `©xyz` | udta | Rewritten as 3GPP `loci` in MP4 |
| NMEA subtitle track | `text`/`tx3g` | Copied as `tx3g` |

- **[V]** `-map 0 -c copy` to MP4 fails on any codec-none stream (`Could not find tag for codec none`); `-copy_unknown`, `-map_metadata 0`, `-movflags use_metadata_tags` do not copy tracks. Matroska rejects all data streams.
- **[I]** Full preservation requires MP4 box-level surgery ("replace only the video track": GPAC MP4Box or a custom rewriter). Any such approach **must drop original video samples from `mdat`** and strip embedded previews.
- **[V] Privacy:** Insta360 trailer record `0x200` is a JPEG preview; QuickTime `ThumbnailImage`/`PreviewImage`; GoPro LRV/THM sidecars — all contain unblurred faces.
- **[V]** GPS reading: ExifTool `-ee` (124 timed-GPS formats, Perl, Artistic/GPL) is the widest; gpmf-parser (Apache-2.0 OR MIT) for GoPro; telemetry-parser (MIT/Apache) Python wheel stale (0.3.0, 2024); gopro2gpx lacks GPS9 (HERO13).

### Spherical metadata
- **[V]** FFmpeg reads V2 (`sv3d`/`st3d`, equirect + cubemap) and V1 XMP (`uuid`), exposed as stream side data.
- **[V]** FFmpeg writes `sv3d`/`st3d` **only in MP4 mode with `-strict unofficial`**; V1 XMP is never written.
- **[V]** google/spatial-media (Apache-2.0) can inject V1+V2, equirectangular only; Python 3 in source, last tag 2018.
- **UNVERIFIED** whether PyAV `add_stream_from_template()` carries spherical side data into the encoded stream.

### 360° camera outputs
- **[V]** GoPro MAX `.360`: two HEVC 4096×1344 streams in a GoPro-specific EAC layout, **no spherical side data**; FFmpeg `v360` does not handle it.
- **[V]** Insta360 `.insv`: unstitched dual fisheye (stitching needs Insta360 Studio/SDK). Ricoh Theta Z1: dual fisheye unless stitched on device.
- **[V]** Exports from vendor software are equirectangular MP4.
- **[I]** v1 should accept equirectangular only and reject raw formats with an explicit error (consistent with the api#369 discussion).

## 6. Data, models, prior art, privacy metrics

### Baseline model
- **[V]** `Panoramax/detect_face_plate_sign` (HF, etalab-2.0) = SGBlur weights (YOLO11l, imgsz 2048). Validation on 329 images: **face mAP50 0.657, recall 0.619**; plate 0.889; sign 0.898. Training data (`smartphones.yaml`) not published.
- **[I]** Face recall is the weakest link: tracking can fill gaps but cannot recover a face never detected.

### Training data licences (for redistributed weights)
- Usable: Open Images V7 (face, plate, sign; annotations CC BY 4.0, images CC BY 2.0), CCPD (MIT, Chinese plates), Panoramax HF sign datasets (etalab-2.0 / CC BY-SA 4.0), Panoramax pictures (signs only — faces/plates already blurred).
- OSM-France terms (2026-02-02) allow derived data "including AI models" under Licence Ouverte 2.0, CC-BY 4.0 or ODbL 1.0 — **possible tension with Ultralytics' AGPL claim on weights (legal question)**.
- Not usable (NC/ND/research-only): WIDER FACE, CrowdHuman, PP4AV, UFPR-ALPR, LSV-LP, MTSD, Vistas, TT100K, DFG, MOT17/20, KITTI, BDD100K, Waymo, nuScenes, InsightFace models.
- Practical route: EgoBlur (Apache-2.0) as teacher to pseudo-label CC footage, then human review (CVAT, MIT).

### Candidate test videos (keep out of git; download script + SHA-256)
- Wikimedia Commons "Dashcam Recording (urban).ogv" — CC0, 1080p, 120 s.
- Commons "City Driving 4K- Kraków Poland 2024.webm" — CC BY 3.0.
- Commons UK cyclist close-pass clips — CC BY 3.0, readable plates.
- Internet Archive "Shenzhen 360º: Insta360 One X" — CC BY-SA 4.0, 5760×2880 equirectangular.
- Internet Archive `MarinCountyCADriving/GP030064.MP4` — CC BY 4.0, HERO5 with GPMF GPS (4 GB, rural).
- `gopro/gpmf-parser` sample files — Apache-2.0 OR MIT, small, good CI fixtures for telemetry.
- Commons only stores WebM/Ogg transcodes (telemetry lost). Avoid Pexels/Pixabay (no ML use, no redistribution).
- CC-licensed clips with identifiable people are still personal data.

### Prior art
- deface (MIT, CenterFace, frame-by-frame), EgoBlur (Apache-2.0, Faster R-CNN ~104M params), DashcamCleaner (AGPL-3.0, "blur memory"), dashcam_anonymizer (MIT), UrbanAnonymizer (Sept 2026, CC BY-NC: forward-backward tracking, interpolation, track extension; MOT17/20 head-frames masked 87.4 % → 95.8 %, "transient exposures" 4540 → 303). None handles the 360° seam.
- Tracker reference implementations (ByteTrack, BoT-SORT, OC-SORT, Deep OC-SORT, Hybrid-SORT, BoostTrack, TrackTrack, FastTracker) are MIT; StrongSORT GPL-3.0; `boxmot` AGPL-3.0; roboflow `trackers` Apache-2.0; TrackEval MIT.

### Privacy metrics and regulation
- No standard "X % covered" rule. Proposal: coverage = |GT box ∩ blur mask| / |GT box| (not IoU, over-blur is harmless); protected if coverage ≥ 0.9–1.0; leakage rate = unprotected object-frames / object-frames; also per-track "ever leaked", longest exposed run, transient exposures, size buckets, over-blur ratio.
- Weak blur overstates privacy against an informed attacker (PoPETs 2024, arXiv 2304.01635) → irreversible masking.
- EDPB Guidelines 3/2019: irreversible blurring counts as erasure. Art. 29 WP letter to Google (2010): unblurred originals ≤ 6 months. Panoramax legal guide (2025-03-14): no originals kept.

## 7. Discrepancies between the project brief and current sources

| # | Brief says | Sources say | Proposed handling |
|---|---|---|---|
| 1 | SGBlur `MODELS` in `config.py` | In `src/detect/detect.py` | Ours goes in a config file (as briefed) |
| 2 | SGBlur keep=1 zones "chiffrées/salées" | Stored in clear, salt only hides the path; no TTL; deblur broken | Design real encryption + TTL |
| 3 | `service_name` used for cleanup | Backend ignores it, hard-codes prefix `SGBlur-` | Choose `API_NAME` so tags start with `SGBlur-` (e.g. `SGBlur-Video`) or patch upstream |
| 4 | Python ≥ 3.11 | PyAV 19 requires ≥ 3.12 | Python ≥ 3.12 |
| 5 | "FastTracker" | File is `fasttrack.yaml` | Use the real name |
| 6 | `tracktrack.yaml` + `sparseOptFlow` as a custom choice | Already the Ultralytics default, with precision-oriented thresholds | Ship a recall-oriented custom YAML |
| 7 | `model.track(persist=True, stream=True)` | Drops orphan detections, delays new tracks, global IDs | Own detection loop + tracker classes, one job per process |
| 8 | Preserve GPMF, CAMM… | GPMF GPS track OK; CAMM and other codec-none tracks impossible with ffmpeg | Support matrix + GPS always exported in metadata; box-level surgery later |
| 9 | 360° = equirectangular | Raw GoPro MAX / Insta360 / Theta Z1 are not | Equirect only in v1, explicit 415 for raw formats |
| 10 | `HALF` setting | `half` deprecated → `quantize` | Map `HALF` to `quantize=16` |
| 11 | Panoramax consumes the blur API | Synchronous, one JPEG per call; no video support upstream | Video API is an extension; engage on api#369; `/frames` as bridge |
| 12 | Forge GitLab | Local remote is GitHub | Maintainer decision |
| 13 | Licence AGPL | Local `LICENSE` is MIT | Maintainer decision (recommend AGPL-3.0-or-later) |

## 8. Risks

| Risk | Level | Mitigation |
|---|---|---|
| Detector misses faces (baseline face recall 0.62) — tracking cannot fix total misses | High | Low `CONF_BLUR`, multi-scale, tiling, optional second face/plate detector (EgoBlur, Apache-2.0), privacy regression test |
| Licence chain for weights (Ultralytics AGPL claim vs OSM-FR AI terms vs etalab-2.0 upstream) | High | Document in `docs/license.md`, ask Panoramax/OSM-FR, keep weights out of git |
| Telemetry preservation beyond GPMF needs MP4 box surgery | Medium-high | Phased support matrix, GPS sidecar in metadata, dedicated tests |
| 360°: raw formats, seam, GMC not suited to equirect, 5.7K–8K cost | Medium-high | Equirect only, circular padding, downscaled GMC input, benchmarks |
| Throughput: multi-scale at 2048/4096 px on every frame (e.g. 10 min × 30 fps = 18 000 frames × 2–3 passes) | Medium-high | GPU + TensorRT + NVENC; CPU mode for small videos/dev only; measure in step 8 |
| Ultralytics API churn (daily releases, breaking changes in patches) | Medium | Exact pin, thin adapter module, contract tests on the tracker API |
| Upstream integration undefined (no video in Panoramax yet) | Medium | Engage on api#369 before freezing the API |
| Privacy test set needs annotated videos containing personal data | Medium | Small self-annotated set, stored outside git, access-controlled |
| Embedded previews/sidecars leaking unblurred faces | Medium | Strip everything not explicitly allow-listed at remux |
| Local dev: no ffmpeg, no CUDA (Apple M4 Pro), system Python 3.9 | Low | `uv` with Python 3.12, MPS + VideoToolbox for dev, GPU CI/runner for benchmarks |

## 9. Open questions for the maintainer

1. Licence: switch `LICENSE` from MIT to AGPL-3.0-or-later?
2. Forge: GitHub (current remote) or GitLab (Panoramax ecosystem)? Determines CI and Pages.
3. Python ≥ 3.12 instead of ≥ 3.11?
4. Primary output for Panoramax: blurred video + annotations (api#369 direction), with `/frames` (geotagged blurred JPEGs) as a bridge for today's backend?
5. `API_NAME` default: `SGBlur-Video` (compatible with the backend's `SGBlur-` prefix cleanup) or something else?
6. Model bootstrap: start with the existing SGBlur YOLO11 weights while the YOLO26 training pipeline matures? Is there access to SGBlur's training data or to the Panoramax team?
7. Hardware: is an NVIDIA GPU available for benchmarks/training, and what is the target deployment hardware?
8. v1 scope: equirectangular-only 360°, and telemetry support limited to GoPro GPMF track + GPS sidecar in metadata?

## 10. Sources

- SGBlur: <https://gitlab.com/panoramax/server/sgblur> (commit `e39fb20`)
- Panoramax backend: <https://gitlab.com/panoramax/server/api> (commit `50efe97`)
- Blur API spec: <https://docs.panoramax.fr/backend/install/deep_dive/blur_api/>
- Semantics: <https://docs.panoramax.fr/backend/install/deep_dive/semantics/>, <https://docs.panoramax.fr/tags/syntax/>
- Video issue: <https://gitlab.com/panoramax/server/api/-/work_items/369>; NLnet: <https://nlnet.nl/project/Panoramax-video/>
- Ultralytics: <https://docs.ultralytics.com/modes/track/>, <https://docs.ultralytics.com/models/>, <https://docs.ultralytics.com/models/yolo27/>, <https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/trackers>, <https://www.ultralytics.com/license>, <https://docs.ultralytics.com/guides/yolo-thread-safe-inference/>
- PyAV: <https://pypi.org/project/av/>; pyav-ffmpeg patch: <https://github.com/PyAV-Org/pyav-ffmpeg/blob/master/patches/ffmpeg.patch>
- FFmpeg gpmd muxing: <https://github.com/FFmpeg/FFmpeg/commit/981178f3b0b547a228804ff36641b2237eb75a16>; CAMM patches: <https://ffmpeg.org/pipermail/ffmpeg-devel/2017-July/213338.html>
- gpmf-parser: <https://github.com/gopro/gpmf-parser>; spatial-media: <https://github.com/google/spatial-media>; ExifTool: <https://exiftool.org/TagNames/QuickTime.html>
- HF model: <https://huggingface.co/Panoramax/detect_face_plate_sign>; EgoBlur: <https://arxiv.org/abs/2308.13093>; UrbanAnonymizer: <https://huggingface.co/mehmetkeremturkcan/UrbanAnonymizer>
- EDPB Guidelines 3/2019: <https://edpb.europa.eu/sites/default/files/files/file1/edpb_guidelines_201903_video_devices_en_0.pdf>
