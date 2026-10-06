# Architecture Decision Records

Decisions are recorded with [MADR 4](https://adr.github.io/madr/). One file per
decision, numbered, never renumbered. A superseded ADR stays in place with
status `superseded by ADR-XXXX`.

| ADR | Title | Status |
|---|---|---|
| [0001](0001-two-pass-architecture.md) | Two-pass architecture with an intermediate `detections.jsonl` | proposed |
| [0002](0002-own-detection-loop-with-ultralytics-trackers.md) | Own detection loop driving Ultralytics tracker classes | proposed |
| [0003](0003-default-tracker.md) | Default tracker: TrackTrack with a recall-oriented configuration | proposed (provisional until step 8) |
| [0004](0004-irreversible-blur.md) | Irreversible blur method and shapes | proposed |
| [0005](0005-job-queue.md) | Job queue: SQLite and worker processes, no Redis | proposed |
| [0006](0006-licence.md) | Licence: MIT code with an AGPL-3.0 runtime dependency | accepted (maintainer, 2026-10-06) |
| [0007](0007-360-video.md) | 360° video: equirectangular only, circular padding, seam stitching | accepted (step 7) |
| [0008](0008-video-io-and-metadata-preservation.md) | Video I/O with PyAV and MP4 box post-processing | accepted (steps 4 and 7) |
| [0009](0009-model-registry-and-class-policy.md) | Reuse SGBlur YOLO26 weights; registry and class policy by name | accepted (maintainer, 2026-10-06) |
| [0010](0010-privacy-testing-strategy.md) | Privacy testing: synthetic oracle + annotated real clips | accepted (maintainer, 2026-10-06) |
| [0011](0011-offline-linking.md) | Offline linking of detections the tracker leaves alone | accepted (step 4 measurements) |

Template for new records: copy the structure of any file above (Context and
Problem Statement, Decision Drivers, Considered Options, Decision Outcome,
Consequences, More Information).
