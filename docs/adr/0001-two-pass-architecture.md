---
status: proposed
date: 2026-10-06
---

# Two-pass architecture with an intermediate `detections.jsonl`

## Context and Problem Statement

Blurring a face on every frame where it is visible requires knowing, for each
frame, about detections in *later* frames: gap filling needs the next
observation, temporal padding before a track's first detection needs to know
where the track starts. How do we organise analysis and rendering?

## Decision Drivers

- Privacy: gaps and track starts must be covered, which needs future frames.
- Memory: an 8K frame is 88 MB in BGR; buffering seconds of video is not an option.
- Debuggability and reproducibility for volunteer contributors.
- Testability of post-processing without GPU.
- Ability to run detection on another machine (SGBlur's two-service split).

## Considered Options

1. Single pass with a sliding frame buffer (delay rendering by N frames).
2. Two passes over the file with an intermediate detection file.
3. Single pass writing detections, then a full re-decode (same as 2 but without a stable file format).

## Decision Outcome

Chosen option: **2**. Pass 1 decodes, detects and tracks, and appends one line
per frame to `detections.jsonl` ([format](../design/detections-format.md)).
Post-processing turns it into a blur plan and sign annotations. Pass 2
decodes again, blurs, encodes and remuxes.

### Consequences

- Good: gap filling and padding use complete tracks; no frame buffer; memory stays bounded by one frame.
- Good: `sgblur-video render` can re-render with new blur settings without re-running detection; bugs can be reproduced from the file alone; post-processing is unit-tested on CPU with synthetic files.
- Good: the Detect API streams exactly this file, so the remote and in-process paths share one contract.
- Bad: the video is decoded twice. Measured cost: 8K HEVC decodes at ~50 fps on an M4 Pro, small next to ~3 fps of analysis.
- Bad: the original must stay on disk until pass 2 ends (it is deleted right after, see the retention table in the architecture).

## More Information

Option 1 would need ≥ padding + max gap frames in memory (≥ 2 s at 30 fps =
60 frames = 5 GB at 8K), and still could not fill gaps longer than the buffer.
