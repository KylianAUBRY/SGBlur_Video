# Model card — `yolo26s_panoramax.pt` (SGBlur)

> Compiled from the checkpoint metadata and public Panoramax sources on
> 2026-10-06. The authoritative source is the SGBlur project; this card only
> records what SGBlur-Video relies on.

| Item | Value |
|---|---|
| Publisher | Panoramax / SGBlur (Christian Quest), <https://gitlab.com/panoramax/server/sgblur> |
| File / commit | `models/yolo26s_panoramax.pt`, commit `34c318b` (2026-10-06) |
| SHA-256 | `9efa3df0c719713c79151f55fb5b3b694d168ee793901ef8f11dda753e7152a7` |
| Architecture | Ultralytics YOLO26s (`yolo26s.yaml`) |
| Classes | `direction`, `sign`, `plate`, `face` (in this order) |
| Training | 2026-02-05, Ultralytics 8.4.11, `imgsz=2048`, 1000 epochs, dataset `smartphones.yaml` (not published) |
| Validation (checkpoint, all classes) | precision 0.804, recall 0.682, mAP50 0.786, mAP50-95 0.418 |
| Licence statements | Hugging Face card of `Panoramax/detect_face_plate_sign`: etalab-2.0; checkpoint metadata: AGPL-3.0 |

## Intended use in SGBlur-Video

Detection of faces and plates to blur, and of signs/direction signs to
annotate, on video frames at several scales (1024, 2048 and tiles up to 4096 px).

## Known limits

- Per-class metrics are not published for this checkpoint; for the previous
  YOLO11 model the face class was the weakest (recall 0.62 on 329 validation
  images, Hugging Face card).
- Training data come from smartphone pictures: video-specific conditions
  (motion blur, compression, rolling shutter), 360° distortion, night,
  profile faces, foreign plates and very small objects are expected to be
  harder. The privacy benchmark (`docs/design/testing-strategy.md`) will
  report leakage by class and size.
- Measured in step 8 on an 8K 360° city video (Apple M4 Pro): with the
  `standard` profile YOLO26s reports about 4.5 faces and 5 plates per frame
  above `CONF_BLUR`, twice as many faces as YOLO11s; small objects are detected
  on about one frame in two, which the offline linking and padding compensate
  ([benchmarks/results](../benchmarks/results/README.md)). Leakage against
  annotated real clips is pending.
