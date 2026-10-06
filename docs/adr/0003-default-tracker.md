---
status: accepted (thresholds provisional until the privacy benchmark on annotated clips)
date: 2026-10-06
---

# Default tracker: TrackTrack with a recall-oriented configuration

## Context and Problem Statement

The camera is moving (car, bike, pedestrian, 360° rig). Ultralytics 8.4.173
ships six trackers: `botsort`, `bytetrack`, `ocsort`, `deepocsort`, `fasttrack`,
`tracktrack`. Which one, configured how, is the default?

## Decision Drivers

- Camera-motion compensation (GMC) for moving cameras.
- Few identity breaks inside one physical object (gap filling and sign deduplication rely on ids).
- Recall over precision: low-score boxes must be able to continue a track.
- CPU cost of GMC on large frames.

## Considered Options

| Tracker | GMC | Notes |
|---|---|---|
| ByteTrack | no | two-stage association with low-score boxes; fast |
| BoT-SORT | yes (`sparseOptFlow`) | ByteTrack + GMC (+ optional ReID) |
| OC-SORT | no | observation-centric motion |
| Deep OC-SORT | yes but `none` by default | needs appearance features for best results |
| FastTracker (`fasttrack`) | no | occlusion-aware, vehicle-oriented |
| TrackTrack | yes (`sparseOptFlow`) | CVPR 2025, Ultralytics default since 8.4.76; strict default thresholds |

## Decision Outcome

Chosen option (provisional): **TrackTrack**, with a project YAML
`configs/trackers/tracktrack-recall.yaml` derived from the Ultralytics default:

| Key | Ultralytics default | Proposed | Why |
|---|---|---|---|
| `track_high_thresh` | 0.6 | *0.35* | faces at distance rarely score 0.6 |
| `track_low_thresh` | 0.25 | *0.10* | let low-score boxes continue tracks |
| `new_track_thresh` | 0.7 | *0.40* | start tracks on plausible faces |
| `min_track_len` | 3 | *2* | confirm faster (orphans are blurred anyway) |
| `track_buffer` | 30 frames | `track_buffer_s: 2.0` (converted with the video fps) | Ultralytics counts frames regardless of fps |
| `gmc_method` | sparseOptFlow | sparseOptFlow | moving camera |
| `with_reid` | false | false | ReID cost not justified before measurement |

Tracking runs on a frame downscaled to `TRACK_WIDTH` (1920 px) so GMC stays
cheap on 8K input. `botsort-recall.yaml` and `bytetrack-recall.yaml` are shipped
for comparison.

### Consequences

- Good: GMC out of the box; current Ultralytics default, so best maintained.
- Bad: thresholds are guesses until measured. Step 8 compares all trackers on the annotated clips (ADR-0010): track fragmentation, id switches, FPS and — first criterion — leakage rate after post-processing. This ADR is then updated to `accepted` or superseded.
- Note: a single affine GMC models forward motion (zoom) and equirectangular distortion poorly; post-processing (gap filling, padding) is what guarantees privacy, the tracker only improves continuity.

## Step-8 measurements (2026-10-06)

Details: [benchmarks/results](https://github.com/KylianAUBRY/SGBlur_Video/blob/main/benchmarks/results/README.md).
TrackTrack, TrackTrack without GMC, BoT-SORT and ByteTrack were replayed on the
same YOLO26s detections of three 10 s clips (1080p pedestrians; two 8K 360°
city scenes):

- 1080p: all trackers follow ~70 % of the detections, with 26–30 blurred chains.
- 8K 360°: all trackers follow only **2–12 %** of the detections. Small objects
  are detected on about one frame in two and their merged boxes jump between
  detection passes (median IoU 0.52 between consecutive detections), so new
  tracks are dropped before they are confirmed. TrackTrack follows the most
  (11.9 % and 2.8 % against 10.2–10.7 % and 1.9–2.2 %).
- After offline linking ([ADR-0011](0011-offline-linking.md)) the blur plans
  differ by less than 3 % between trackers, and sign counts by at most 2.
- GMC costs ~15 ms per 1080p frame and ~11 ms per 8K frame (at `TRACK_WIDTH`),
  small next to detection (0.43 s per 8K frame on Apple M4 Pro), for no
  measurable gain on these clips.

Decision: **TrackTrack stays the default**, with GMC, because it is marginally
the best and the choice barely matters once fragments are linked offline. On
high-resolution 360° video the tracker is not what protects privacy: offline
linking, interpolation and padding are. The thresholds of
`tracktrack-recall.yaml` remain provisional: the leakage rate per tracker will
be compared on the annotated clips (`sgblur-video benchmark trackers --dataset`)
and this section updated. Two leads for later work, not needed for privacy:
tracking on the best-pass box instead of the cross-pass union, and a lighter
"confirm on the next frame" rule for flickering detections.
