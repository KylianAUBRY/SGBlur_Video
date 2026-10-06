# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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
