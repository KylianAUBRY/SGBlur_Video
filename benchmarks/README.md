# Benchmarks

Planned for step 8 (`sgblur-video benchmark`):

- **Privacy** — leakage metrics on the manually annotated dataset (kept outside
  git), gated by `privacy-thresholds.yaml`. Definitions in
  [docs/design/testing-strategy.md](../docs/design/testing-strategy.md).
- **Trackers** — TrackTrack, BoT-SORT, ByteTrack (and others shipped by
  Ultralytics): fragmentation, identity switches, FPS, and leakage after
  post-processing. Feeds [ADR-0003](../docs/adr/0003-default-tracker.md).
- **Models and devices** — YOLO26s vs YOLO11s, CPU vs MPS vs CUDA, detection
  profiles.

Results are written as Markdown/JSON reports without any image or file name
of the dataset.
