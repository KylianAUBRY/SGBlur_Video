# Annotations and Panoramax

SGBlur-Video never blurs traffic signs. It detects them, follows them over
time and returns **one Panoramax annotation per physical sign**, using exactly
the tags SGBlur emits for pictures.

## 1. One annotation per physical sign

A sign is visible on many frames: a dashcam passing a sign at 30 km/h sees it
for a second or two, i.e. dozens of frames. Sign detections (classes `sign` and
`direction`) go through the same tracking and offline linking as faces and
plates ([Tracking and post-processing](tracking.md)), and each resulting chain
becomes a single annotation if:

- it has at least `SIGN_MIN_TRACK_LENGTH` detections (5): a sign seen on 3
  frames is more likely a false positive than a real sign;
- its best score is at least `CONF_SIGN` (0.6, SGBlur's threshold).

The annotation describes the sign on its **best frame**: the detection with the
largest `score × area` — a sign seen close and clearly rather than far and
uncertain.

## 2. Semantic tags (identical to SGBlur)

```json
{
  "shape": [2201, 1502, 2291, 1591],
  "semantics": [
    {"key": "osm|traffic_sign", "value": "yes"},
    {"key": "detection_model[osm|traffic_sign=yes]", "value": "SGBlur-Video-yolo26s/0.1.0"},
    {"key": "detection_confidence[osm|traffic_sign=yes]", "value": "0.874"}
  ],
  "video": {
    "track_id": "signage:3", "class": "sign",
    "best_frame": 431, "best_timestamp": 14.367,
    "first_frame": 402, "first_timestamp": 13.4,
    "last_frame": 470, "last_timestamp": 15.667,
    "observations": 61, "confidence_max": 0.874, "confidence_mean": 0.731
  }
}
```

- `shape`: integer pixel box `[minx, miny, maxx, maxy]` of the best frame, top-left
  origin, in the frame's **display orientation** (rotation applied), clipped to
  the frame — the format the Panoramax API validates for pictures.
- Values are strings, as the Panoramax API requires.
- `detection_confidence` is the score of the best-frame detection.
- `direction` signs (direction panels) get the same tags as `sign` until SGBlur
  defines specific semantics (maintainer decision).

## 3. The `video` extension

Panoramax annotations describe parts of a *picture*. For video, each annotation
also carries a `video` object (track id, first/best/last frame and timestamp,
number of detections, maximum and mean confidence, and — from step 7 — the GPS
position at the best frame). The current Panoramax backend silently ignores
unknown annotation fields, so the extension is harmless but not stored; it is
meant for the video support being discussed upstream
([panoramax/server/api#369](https://gitlab.com/panoramax/server/api/-/work_items/369)).

## 4. `service_name`, `API_NAME` and re-blurring

`service_name` is `API_NAME` (`SGBlur-Video`) and `detection_model` values are
`{API_NAME}-{model}/{version}`. The Panoramax backend does not read
`service_name`: when a picture is blurred again, it deletes previous detection
tags whose `detection_model` value **starts with `SGBlur-`**. Keeping that
prefix makes video annotations behave like SGBlur's. Never change `API_NAME`
after deployment.

## 5. Best-frame pictures for today's Panoramax

Panoramax does not accept videos yet. With `--frames-dir` (CLI) or `frames=1`
(API, step 6), SGBlur-Video writes one JPEG per frame that holds a sign's best
view, plus `frames.json`:

```json
{"frames": [
  {"n": 0, "file": "0.jpg", "frame": 431, "timestamp": 14.367, "width": 7680, "height": 3840,
   "annotation_indices": [0, 4], "shapes": [[2201, 1502, 2291, 1591], [5120, 1700, 5180, 1760]]}
]}
```

- Each JPEG is the **blurred** output frame: faces and plates on it are blurred.
- Several signs sharing a best frame share one picture (`annotation_indices`).
- EXIF holds the capture date (container `creation_time` + timestamp); GPS is
  added from video telemetry in step 7. Panoramax needs both to accept a
  picture, so frames from videos without GPS cannot be uploaded as they are.
- The pictures can be uploaded with `isBlurred=true` (no second blurring) and
  their annotations (Panoramax API ≥ 2.16).

## 6. Future work: sign types

A second-stage classifier could turn `osm|traffic_sign=yes` into a precise code
(e.g. `osm|traffic_sign=FR:A15b`) from the best-frame crop, as Panoramax's
Prolix tool does for pictures with the `classified_fr_road_signs` dataset. It
would plug in after sign deduplication; it is not implemented.
