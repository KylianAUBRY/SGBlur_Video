# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **Every frame is blurred on its own detections, like SGBlur blurs a picture** ([ADR-0012](docs/adr/0012-independent-frames.md)). Faces and plates are no longer tracked: no offline linking, interpolation, temporal padding, smoothing or margin. Each detection with a score ≥ `CONF_DETECT` (now 0.30, SGBlur's `MIN_CONF`) is blurred with a rectangle exactly on its box; duplicates across passes are merged like SGBlur (IoU > 0.33 or nested, the smallest box is kept); boxes under 12 px are skipped. On 300 frames of 8K 360° city video, the blurred area falls from 1.35 % to 0.011 % of the frame; timelapse files (captured at ~2 fps, stored at 30 fps) need no special case. A face or plate the model misses on a frame is visible on that frame, as with SGBlur. Only signs are tracked (one annotation per sign). Job statistics: `blurred_boxes` per class, `signs`; `keep=1` keeps regions by detection score.

- Documentation review (step 9): privacy and GDPR page written from the implemented behaviour; design documents and ADRs marked accepted with their stale statements fixed (seam stitching replaced by wrap-aware linking, package layout, metrics, error codes, `GET /` device); installation, quick start, CLI and contributing guides completed for newcomers (test markers, relative default paths, no system FFmpeg); README and French README brought up to date; status is now alpha.
- `sgblur-video config` shows the home folder as `~` (its output is pasted in public issues); CLI help renders Markdown.
- Invalid query parameters return `422 {"detail", "code": "invalid_parameter"}` like every other API error.

### Performance

- Faster analysis, 2.1 → 3.9 frames/s for 8K 360° video on an Apple M4 Pro: FP16 inference on MPS by default (`HALF=auto`), 360° tiles cropped with slices instead of an index array, and a three-stage pipeline (frame preparation and tracking run in threads while the accelerator runs inference; identical detections). `YoloDetector` exposes `prepare` (CPU) and `infer` (accelerator).

### Fixed

- 8K 10-bit jobs killed for lack of memory in Docker on a Mac, during the frame range cut or the analysis. Above 4K, x265 keeps 3 frames of lookahead, 1 B-frame and 1 reference instead of 20, 4 and 3 (decoding and encoding 70 frames: above 7 GB → 4.4 GB, as fast). On CPU, the 360° tiles are inferred one at a time and one frame is prepared ahead instead of two (5.95 → 4.2 GB, as fast). An 85-frame range of 8K 10-bit 360° video with the debug video now peaks at 5.7 GB.
- GPS positions are timed from the first video frame instead of the start of the telemetry track (they differ in a frame range).
- 8K jobs killed for lack of memory with CPU encoding (Docker on a Mac): frames above 4K are decoded with 4 threads (2.6 GB → 0.9 GB for 8K HEVC, as fast), the detector's memory is released before rendering, and the encoder availability probe opens the encoder with its real settings. A frame range is encoded at the bit rate of the original video, not of the intermediate cut. A job killed by the system now says so ("most likely for lack of memory") instead of "Processing stopped".

### Removed

- Settings `CONF_BLUR`, `BLUR_BOX_MARGIN`, `BLUR_TEMPORAL_PADDING_FRAMES`, `BLUR_PADDING_GROWTH`, `MAX_INTERPOLATION_GAP_S`; the `benchmark trackers` command; elliptical face shapes; the `chains/track` benchmark column (ADR-0012).

- `WORKER_CONCURRENCY`, which nothing read: parallelism comes from the number of worker processes (`serve --workers`, `worker`).

### Added

- Optical-flow tracker, now the default sign tracker (`TRACKER_CONFIG=configs/trackers/flow.yaml`, ADR-0003): each track's box follows the optical flow of the image around it and is matched to detections by centre distance, so small plates that move more than their own width between frames of 8K 360° video are followed. TrackTrack followed 0–16 % of the detections on 8K 360° clips and turned each plate into a dozen fragments, each padded on both sides; the flow tracker follows 96–100 %, divides the number of blurred chains by 2 to 4 and covers at least as many faces and plates found by another model (YOLO11l) on each of four clips. The Ultralytics trackers remain available. Since ADR-0012 it only tracks signs.
- Debug video from the HTTP API and the web page: `POST /blur/?debug=1` also writes the annotated video of `--debug`, served at `GET /jobs/{id}/debug` (`links.debug`); setting `DEBUG_VIDEOS` (on by default) refuses it with `422 debug_unavailable`. The web page has a *Debug video* option, a switch between the blurred and the debug video at the same moment, and a legend. The overlay now shows the class by colour (face, plate, sign, direction sign), detections as solid outlines labelled with their score, and regions blurred from the track without a detection on that frame (interpolated, padded) as unlabelled dashed outlines.
- Switching models: `MODEL_PATH` (any local `.pt` checkpoint, classes read from the file and checked against `CLASS_POLICY`, tag `SGBlur-Video-<file>/local-<sha>`), `--model` accepting a registry name or a path in every command (benchmarks compare several), `sgblur-video models inspect` (classes, SHA-256, usability, registry entry to complete), `models list` marking the model in use; SGBlur `yolo11n`, `yolo11m` and `yolo11l` added to the registry (pinned URLs). Checkpoints are read with Ultralytics' loader, so older formats are recognised (and refused when their classes are unnamed).
- Frame range: `POST /blur/?start_frame=N&end_frame=M` processes only frames `[N, M)`. The range is cut exactly (re-encoded at the source bit rate) with its audio and telemetry shifted to start at 0 and its 360°/rotation metadata; the rest of the upload is deleted before processing. The web page has a preview with a two-handle bar, frame number fields and a "20 frames" shortcut.
- Web page at `/ui` (setting `WEB_UI`, on by default): upload a video with progress, follow the job, watch and download the blurred video, list the sign annotations and their best pictures, delete the results. Self-contained, it only calls the HTTP API.
- A test checks that both HTTP applications serve exactly the routes of `docs/design/openapi.yaml`.

- Benchmarks (step 8): privacy dataset tooling (`annotate export` cuts a clip, encodes a ≤ 3840 px CVAT proxy and a model pre-annotation in "CVAT for video 1.1"; `annotate import` converts the corrected CVAT export into full-resolution ground truth), coverage-based leakage metrics (per class, size and readability, tracks ever leaked, longest and transient exposures, re-identifications, over-blur) with the gate of `benchmarks/privacy-thresholds.yaml`; `benchmark privacy` with setting sweeps, `benchmark trackers` replaying tracking on cached detections, `benchmark speed` per model, device and detection profile; JSON and Markdown reports without file names; the privacy benchmark also runs on the synthetic scenario in CI.
- 360° and telemetry (step 7): circular padding of detection passes and tiles across the 0°/360° seam, wrap-aware merge, linking, rendering and sign shapes; allow-listed MP4 box transplant restoring Spherical Video V1/V2 metadata, the display matrix and GoPro camera boxes; GoPro GPMF GPS parser (`GPS5`, `GPS9`) geolocating sign annotations and best-frame EXIF; free GoPro sample fetched for tests; synthetic seam scenario with negative control in the privacy oracle.
- HTTP API and Docker (step 6): asynchronous Blur API (upload streamed to disk with size limit, jobs, progress and ETA, video with range requests, metadata, best frames, delete, Prometheus metrics, optional bearer token, `sync=1` with SGBlur-style multipart response, allow-listed callbacks), Detect API streaming `detections.jsonl` for split deployments, SQLite job store, worker with one process per job, timeouts, cancellation and a retention janitor, `keep=1` regions encrypted with AES-256-GCM and expiring; CLI `serve`, `worker`, `serve-detect`; Docker image with `cpu` and `gpu` targets and Compose files.
- Traffic signs (step 5): offline deduplication into one annotation per physical sign (minimum length, `CONF_SIGN`), best frame by score × area, Panoramax annotations with SGBlur tags and a `video` extension, metadata JSON written next to the output, best-frame JPEGs (blurred, EXIF date) with `frames.json`; CLI `signs`, `--frames-dir` on `blur` and `render`.
- Core pipeline (step 4): probe (projection, rotation, copyable streams), multi-scale detection plan with tiles for 8K, cross-pass merge, Ultralytics tracker adapter (one tracker per class group), `detections.jsonl` v1 reader/writer, offline linking of fragments (ADR-0011), gap filling, envelope smoothing, temporal padding, margins, ellipse/rectangle shapes, irreversible blur on YUV planes (8/10 bit), encoder selection (VideoToolbox, NVENC, x264/x265), single-loop render with audio/GPMF stream copy and timecode recreation, annotated debug video.
- CLI: `blur`, `detect`, `render`, `models download` (hash-verified).
- Tests: synthetic privacy oracle with negative control, media integration (VFR timestamps, bit-exact audio), real-model smoke test, CLI end-to-end with a fake detector; CI coverage gate (≥ 85 % on core, privacy, semantics).

- Step 1 analysis of SGBlur, the Panoramax blur API and Ultralytics tracking (`docs/research/`).
- Design documents (`docs/design/`): architecture, pipeline, `detections.jsonl` v1 format, HTTP API contract (OpenAPI 3.1 draft), configuration, testing strategy.
- Architecture decision records ADR-0001 to ADR-0010 (`docs/adr/`).
- Project skeleton: `pyproject.toml` (Python 3.14, uv), package layout, typed configuration with generated reference, model registry with SGBlur `yolo26s` weights (pinned URL and SHA-256), recall-oriented tracker configurations, CLI skeleton, unit tests.
- Tooling: ruff, mypy (strict), pytest with coverage, pre-commit, GitHub Actions CI, MkDocs Material documentation site.
- Community files: README (English and French), contributing guide, code of conduct, security and privacy policy (public issue template without media in v1; private reporting planned for v2), third-party licences.
