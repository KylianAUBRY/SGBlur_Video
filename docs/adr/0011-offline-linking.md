---
status: accepted
date: 2026-10-06
---

# Offline linking of detections the tracker leaves alone

> **Update 2026-10-08.** Since [ADR-0012](0012-independent-frames.md), offline linking is used for signs and benchmark pre-annotation only; faces and plates are blurred frame by frame.

## Context and Problem Statement

The first run on a real 8K equirectangular video (Q360 camera, 90 frames)
showed that the trackers leave most face and plate detections **untracked**:
833 of 1156 detections were orphans with TrackTrack, and BoT-SORT and
ByteTrack behaved the same when replayed offline on the same detections.

Faces and plates far from the camera are only ~10 px high at tracking
resolution and the detector sees them intermittently (median frame-to-frame
IoU of the best match 0.52, 10th percentile 0). IoU-based association rarely
confirms such tracks. Lowering the tracker thresholds tracked more boxes but
split them into hundreds of tiny tracks.

Orphans were still blurred (with temporal padding around each one), so privacy
held, but the result was inefficient and fragile: ~31 overlapping padded
shapes per object per frame, and no interpolation between two sightings of the
same object.

## Decision Drivers

- Privacy: continuity between intermittent sightings, not only padding around each.
- Cost: the number of blurred shapes per frame drives rendering time.
- Pass 2 has the whole video: association can look at the future.

## Considered Options

1. Tune tracker thresholds until most detections are tracked.
2. Keep orphans independent (padding around each).
3. Link fragments offline in post-processing.

## Decision Outcome

Chosen option: **3**. Every tracker track and every orphan is a *fragment*.
Fragments of the same class group are chained in time order when a fragment
starts at most `LINK_MAX_GAP_S` (1 s) after a chain ends and its first box is
within `LINK_MAX_DISTANCE` (1 box size, +10 % per frame of gap) of where the
chain was heading (constant-velocity extrapolation), with areas within ×4.
Chains then go through selection, gap filling, smoothing and padding as
tracks did before.

On the same 90 frames: 482 fragments → 108 chains; padded shapes 10 422 → 1 982;
312 frames gained an interpolated box between sightings.

### Consequences

- Good: flickering detections become continuous chains; padding only at chain ends; rendering cost drops.
- Good: a low-score detection linked to a confirmed chain is blurred (consistent with the `CONF_BLUR` rule).
- Neutral: wrongly linking two different objects only merges their blur regions (over-blur), it never removes blur.
- The Ultralytics tracker is still used: it resolves crowded scenes where greedy linking would hesitate, and its ids seed chain ids.
- The tracker benchmark (step 8) now compares trackers *after* linking.
