---
status: accepted (optical flow since 2026-10-07; re-check on the annotated clips)
date: 2026-10-07
---

# Default tracker: optical flow (was TrackTrack with a recall-oriented configuration)

> **Update 2026-10-08.** Since [ADR-0012](0012-independent-frames.md), the tracker only follows signs (one annotation per sign); it no longer affects blurring.

The decision of 2026-10-06 (TrackTrack) is kept below for its measurements; it
was replaced on 2026-10-07 by the optical-flow tracker, see
[the last section](#optical-flow-tracker-2026-10-07).

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

## Optical-flow tracker (2026-10-07)

### Problem

On 8K 360° video filmed from a car, users saw walls of padded plate boxes. On a
38-frame parking-lot sample, the detector found about 7 plates per frame
(54 of SGBlur's 55 detections on the same frames, same model), but TrackTrack
followed 4 of 265 plate detections: a plate parked a few metres away moves
about 70 px between frames at 8K, more than its own width, so consecutive
boxes do not overlap and IoU-based association (with or without GMC) fails.
Every plate became a dozen fragments, each padded 15 frames on both sides:
107 plate chains and 79 blurred plate regions per frame.

### Design

`tracker_type: flow` (`src/sgblur_video/core/flowtrack.py`, `configs/trackers/flow.yaml`),
one tracker per class group, at `TRACK_WIDTH`:

1. each track's box is moved by the median pyramidal Lucas-Kanade flow of a
   4×4 grid over the box enlarged by one box size on each side
   (forward-backward error < 2 px; too few good points: last motion kept);
2. tracks and detections are matched by centre distance, by optimal
   assignment (`lap`), within 2 box sizes for boxes up to 32 px and half a box
   size beyond (a whole box size let face tracks jump to the next pedestrian);
   distances wrap around the 360° seam, areas must be within a factor 4;
3. unmatched detections start tracks; a track survives 1 s without detection,
   its box still following the flow.

The flow follows each object's parallax, which no single camera-motion model
describes in an equirectangular frame.

### Measurements

Same cached YOLO26s detections, re-tracked; default post-processing. Without
annotated clips yet, privacy is checked against **another model**: faces and
plates found by YOLO11l (score ≥ 0.4) on every frame must be covered by a blur
region. Four clips: the parking-lot sample (38 frames), `q360-ville` and
`q360-fin` (8K 360°, 300 frames), `gopro-pietons` (1080p pedestrians, 300 frames).

| Clip | Tracker | Plate chains | Plate area | Plates covered | Face chains | Face area | Faces covered |
|---|---|---|---|---|---|---|---|
| parking | TrackTrack | 107 | 3.25 % | 90/92 | 16 | 0.32 % | 2/3 |
| parking | flow | 45 | 2.66 % | 90/92 | 11 | 0.46 % | 2/3 |
| q360-ville | TrackTrack | 232 | 1.24 % | 558/591 | 463 | 2.75 % | 587/647 |
| q360-ville | flow | 96 | 1.20 % | 560/591 | 178 | 2.02 % | 588/647 |
| q360-fin | TrackTrack | 367 | 2.70 % | 863/881 | 167 | 0.70 % | 116/142 |
| q360-fin | flow | 115 | 2.03 % | 866/881 | 82 | 0.77 % | 120/142 |
| gopro-pietons | TrackTrack | — | — | — | 30 | 20.74 % | 551/644 |
| gopro-pietons | flow | — | — | — | 14 | 19.37 % | 553/644 |

Area = mean share of the frame inside plate (or face) blur regions.

- The flow tracker follows 96–100 % of the detections of the 300-frame clips
  (TrackTrack: 0–16 % on 8K 360°, 71 % on 1080p) and divides the number of
  chains by 2 to 4.
- It covers at least as many reference faces and plates on every clip
  (total 1516 plates and 1263 faces against 1511 and 1256).
- Tracking costs 14–15 ms per 8K frame (TrackTrack: 16–17 ms) and 2.7 ms per
  1080p frame (11.5 ms).
- Sweeps: a gate of 3 box sizes lost 5 references on two clips; a 2 s track
  buffer lowered the area further but lost 1–2 references (and moved the
  pedestrian clip by ±10 faces between 1.5 and 2 s: differences under ~5 are
  noise); `fb_error` 2 px covered more than 1 px. Shortening the 15-frame
  padding to 8 frames lost up to 5 references per clip with either tracker:
  padding stays.
- The synthetic privacy oracle passes with both trackers (`tests/privacy`).

Decision: **the optical-flow tracker is the default**. The blurred area falls
less than the number of chains because every chain end is still padded;
the next lever is to make padding follow the flow instead of a straight-line
extrapolation. The comparison will be repeated on the annotated clips
(`sgblur-video benchmark trackers --dataset`).

