---
status: accepted
date: 2026-10-08
---

# Blur every frame independently, like SGBlur blurs a picture

## Context and Problem Statement

Until now, faces and plates were followed through time: a tracker gave them
ids, fragments were linked offline ([ADR-0011](0011-offline-linking.md)),
gaps between sightings were interpolated, every chain was padded for 12–15
frames before and after, boxes were smoothed, enlarged by a margin, and faces
were blurred with an ellipse circumscribing the box. The goal was to cover the
frames where the detector misses an object.

On real 8K 360° street video the result was far too much blur. On 300 frames of
a city clip (yolo26s), 1.35 % of every frame was blurred, but real detections
accounted for only ~7 % of that area: the rest was padding (~78 %) and
interpolation (~15 %) on frames where nothing had been detected. The tracker
linked faces thousands of pixels apart, padded boxes flew onto the camera
car's roof, and timelapse files made it worse: a Q360 export captured at
~2 frames per second but stored at 30 fps turned "1 s" settings into 15 s of
real time.

The maintainer's intent for the project is simpler: cut the video into frames,
treat each frame as an independent picture exactly as SGBlur does, and
re-assemble the video.

## Decision Drivers

- Predictable, explainable output: what is blurred on a frame is what the
  model detected on that frame.
- Behaviour identical to SGBlur on pictures, which Panoramax already relies on.
- No dependence on the real frame rate (timelapse files).
- Less code, faster analysis (no tracker for faces and plates).

## Considered Options

1. Keep the temporal layer and tune it (shorter padding, jump limits): tried
   on 2026-10-07; padding stayed ~70 % of the blurred area and timelapse files
   remained wrong.
2. Keep the temporal layer but make durations follow a user-supplied capture
   frame rate.
3. Blur every frame on its own detections, with SGBlur's rules.

## Decision Outcome

Chosen option: **3**, with SGBlur's rules:

- detections below `CONF_DETECT` = 0.30 (SGBlur's `MIN_CONF`) are dropped;
- duplicates across passes: IoU > 0.33 or one box inside the other, and the
  **smallest** box is kept;
- each face and plate box is blurred with a **rectangle exactly on the box**;
  boxes smaller than 12 px on a side are skipped;
- no tracking, interpolation, padding, smoothing or margin for faces and plates.

Signs keep a tracker and offline linking ([ADR-0011](0011-offline-linking.md)),
only to produce one Panoramax annotation per physical sign. It never affects
what is blurred.

### Consequences

- Good: on the same 300 frames, the blurred area falls from 1.35 % to 0.011 %
  of the frame (1.55 boxes per frame instead of 19.8), every blurred region has
  a detection behind it, and timelapse files need no special case.
- Good: faces and plates are no longer tracked, so analysis runs one tracker
  (signs) instead of three; `benchmark trackers` and the post-processing
  settings are removed.
- Bad: a face or plate the detector misses on a frame is visible on that frame,
  as on a picture blurred by SGBlur. On real 30 fps video this can show as an
  object appearing unblurred for a few frames. The privacy benchmark on
  annotated clips measures it (leakage, longest exposure); it is the place to
  revisit this decision with numbers.
- Bad: a detector box slightly tighter than the object leaves its edges
  visible (no margin), as with SGBlur.
- Neutral: the synthetic oracle now checks that every detection is blurred on
  its frame and that nothing is blurred on frames without a detection; the
  benchmark counts missed frames as leaks.

## More Information

Supersedes, for faces and plates, the tracking part of
[ADR-0002](0002-own-detection-loop-with-ultralytics-trackers.md) and
[ADR-0003](0003-default-tracker.md), the use of offline linking for blurring
in [ADR-0011](0011-offline-linking.md), and the ellipse shape of
[ADR-0004](0004-irreversible-blur.md). Those records stay in place as history.
