---
status: accepted
date: 2026-10-08
---

# Precision-oriented tracking: fewer false blurs, tighter shapes

## Context and Problem Statement

The tracking pipeline was tuned for recall: detections down to 0.10 fed the
trackers, isolated detections at 0.15 were blurred, gaps of up to 2 s were
interpolated, every chain was padded before and after, boxes were merged into
their union (across passes, on the same frame, and by smoothing), enlarged by
a margin, and faces were blurred with the ellipse **circumscribing** the box
(1.57 times its area).

On 8K 360° street video the result was far too much blur, much of it in the
wrong place: trackers linked objects thousands of pixels apart, padded and
interpolated boxes landed where nothing had been detected (the camera car's
roof), and unions turned several boxes into one large box. The maintainer's
direction: it is better to miss difficult detections than to create false
blurs.

## Decision Drivers

- Every blurred region should correspond to a confident detection, or follow
  one very closely in time and space.
- Shapes should fit the object: no unions, small margin.
- Keep the tracker's benefit (blurring a confirmed object on the few frames the
  detector misses), not its failure modes.

## Considered Options

1. Keep recall-oriented tracking and tune it (done on 2026-10-07: padding and
   jump limits; padding stayed ~70 % of the blurred area).
2. Blur every frame independently (branch `independent-frames`, ADR-0012 there).
3. Keep tracking, oriented towards precision.

## Decision Outcome

Chosen option: **3**.

| Item | Before | Now |
|---|---|---|
| YOLO threshold (`CONF_DETECT`) | 0.10 | **0.30** |
| Blur threshold of a chain (`CONF_BLUR`) | 0.15 | **0.40** |
| Tracker | optical flow | **BoT-SORT** (`configs/trackers/botsort.yaml`): new tracks from 0.5, extended down to 0.3 |
| Lost-track retention (`track_buffer_s`) | 1 s | **0.5 s** |
| Offline linking gap (`LINK_MAX_GAP_S`) | 1 s | **0.5 s** |
| Interpolation (`MAX_INTERPOLATION_GAP_S`, `MAX_INTERPOLATION_JUMP`) | 2 s, 20 box sizes | **0.3 s, 5 box sizes** |
| Temporal padding (`BLUR_TEMPORAL_PADDING_FRAMES`, growth) | 12 frames, 5 %/frame | **3 frames, 2 %/frame** |
| Cross-pass merge | union of duplicates | **most precise box** (tiles, then the largest global pass) |
| Several boxes of a chain on one frame | union | **best box** |
| Smoothing | moving average united with the box | **true centred moving average** |
| Margin (`BLUR_BOX_MARGIN`) | 10 % | **5 %** |
| Face shape | ellipse circumscribing the box | **ellipse inscribed in the box** |
| Plate shape | rectangle | rectangle |

Segmentation masks (faces, plates) and oriented boxes (plates) would fit the
objects better still, but need a segmentation or OBB model: the available
weights (SGBlur YOLO26 / YOLO11) are detection models. The shape is a
per-class choice in `core/postprocess.py`, ready for such a model.

### Consequences

- Good: far less blur, and blur where the detections are (numbers in the
  CHANGELOG, measured on 8K 360° city video).
- Bad: an object only ever detected below `CONF_BLUR` is not blurred; an object
  the detector misses for more than 0.3 s, or before its first / after its last
  detection beyond 3 frames, is visible on those frames; a detector box tighter
  than the object leaves its edges visible.
- The synthetic oracle now checks that every confident detection is blurred,
  that an object with weak detections only is not, and that signs never are; the
  privacy benchmark on annotated clips measures the remaining leaks.
- Durations are converted with the file's frame rate: on a timelapse (captured
  at ~2 fps, stored at 30 fps) they cover 15 times more real time.

## More Information

Supersedes the recall-oriented defaults of
[ADR-0003](0003-default-tracker.md) (tracker) and
[ADR-0011](0011-offline-linking.md) (linking gap), and the circumscribed
ellipse of [ADR-0004](0004-irreversible-blur.md).
