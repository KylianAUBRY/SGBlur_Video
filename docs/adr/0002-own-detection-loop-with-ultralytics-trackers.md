---
status: accepted (implemented in steps 4–6)
date: 2026-10-06
---

# Own detection loop driving Ultralytics tracker classes

> **Update 2026-10-08.** Since [ADR-0012](0012-independent-frames.md), only signs are tracked: faces and plates are blurred frame by frame.

## Context and Problem Statement

The project brief suggests `model.track(..., persist=True, stream=True)`.
Reading `ultralytics 8.4.173` (and checking with a spike) shows behaviours that
conflict with the privacy requirements. How should detection and tracking be
wired?

## Decision Drivers

- Every face/plate detection above the blur threshold must be blurred, tracked or not.
- Multi-scale passes and tiles (indispensable for 8K 360°) must feed the tracker.
- One tracker state per video, no interference between concurrent jobs.
- Ultralytics releases almost daily with breaking changes in patch versions.

## Considered Options

1. `model.track()` with `persist=True`.
2. Own detection loop (`model.predict` per pass, merge) + Ultralytics tracker classes called directly.
3. Third-party tracking library (`boxmot`, AGPL-3.0; roboflow `trackers`, Apache-2.0).

## Decision Outcome

Chosen option: **2**, with trackers instantiated as `TRACKTRACK(args)` /
`BOTSORT(args)` / `BYTETracker(args)` and fed
`ultralytics.engine.results.Boxes` built from our merged detections. Each job
runs in its own process. All Ultralytics calls go through one adapter module
(`core/track.py`, `core/detect.py`) covered by contract tests, and the
`ultralytics` version is pinned exactly.

Facts behind the choice (verified in source and in a spike, 2026-10-06):

- `model.track()` returns only boxes matched to a track as soon as one track exists: unmatched and low-score detections are dropped (orphans lost).
- New tracks are not emitted on their first frame; TrackTrack waits `min_track_len` (3) frames.
- Track ids come from a class attribute shared by every tracker in the process (`BaseTrack._count`), reset by any tracker creation.
- `persist=True` never resets when the video changes; the predictor is silently rebuilt (state lost) when `device`, `nms` or `quantize` change.
- `model.track()` runs one inference per frame at one `imgsz`; it cannot merge several passes or tiles.
- Tracker classes take `(args)` only (`frame_rate` was removed in 8.4.49) and return `[x1, y1, x2, y2, id, score, cls, idx]`, where `idx` maps back to our input rows.
- Trackers import `lap`, which Ultralytics tries to `pip install` at runtime if missing: we declare it as a dependency and set `YOLO_AUTOINSTALL=false`.

### Consequences

- Good: orphans are kept and blurred; multi-scale and tiles work; tracker state is explicit and per job.
- Good: one tracker per class group (face, plate, signage) prevents cross-class id switches.
- Bad: we depend on semi-internal classes; mitigated by the adapter, exact pin and contract tests run on every Ultralytics upgrade.
- Neutral: option 3 stays available for the tracker benchmark (step 8); `boxmot` is AGPL-3.0, which is compatible with using it as a dependency of an MIT project only under the same conditions as Ultralytics (see ADR-0006).
