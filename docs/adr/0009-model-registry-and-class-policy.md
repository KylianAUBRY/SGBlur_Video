---
status: accepted
date: 2026-10-06
decision-makers: maintainer (Kylian Aubry)
---

# Reuse SGBlur YOLO26 weights; registry and class policy by name

## Context and Problem Statement

The brief planned a training pipeline for a YOLO26 model with classes `sign`,
`plate`, `face`. There is no NVIDIA GPU available, and SGBlur published
`models/yolo26s_panoramax.pt` on 2026-10-06 (commit `34c318b`). Do we train,
and how do we reference models and classes?

## Decision Outcome

- **No training in this project for now.** We use SGBlur's `yolo26s_panoramax.pt` as is (trained 2026-02-05 with Ultralytics 8.4.11, `imgsz=2048`, dataset `smartphones.yaml`, 1000 epochs). `training/` will only contain documentation and an evaluation script, to be used if the privacy benchmark (ADR-0010) shows gaps that require fine-tuning.
- **Registry** `models/registry.yaml`: name, family, version, file name, pinned download URL, SHA-256, expected class names, memory hint. Weights are downloaded to `MODELS_DIR` and verified; they are never committed. Switching to YOLO27 or a new SGBlur model = adding an entry.
- **Classes by name, never by index.** The YOLO26 model has **four** classes in the order `direction, sign, plate, face`; YOLO11 had `sign, plate, face`. A `CLASS_POLICY` maps names to `blur` / `annotate`; a model that lacks a `blur` class is refused (fail closed); unknown extra classes are ignored with a warning.
- **`direction`** (direction signs) is annotated like `sign`, with the same semantic tags, until SGBlur defines specific semantics (maintainer decision).

Initial registry content:

| name | version | sha256 | classes |
|---|---|---|---|
| `yolo26s` | `0.1.0` (to align with SGBlur when it registers the model) | `9efa3df0c719713c79151f55fb5b3b694d168ee793901ef8f11dda753e7152a7` | direction, sign, plate, face |
| `yolo11s` (legacy, comparison) | `0.1.0` | `28222165ec8fd01dd7e377b2e9f4e2a7d937a15e4a3aa545d2431f317532e774` | sign, plate, face |

URL pattern: `https://gitlab.com/panoramax/server/sgblur/-/raw/<commit>/models/<file>`.

### Consequences

- Good: no GPU needed; same model family as SGBlur, so photo and video annotations stay consistent.
- Bad: only the `s` size exists for YOLO26, so VRAM-based size selection (SGBlur's n/s/m) has a single candidate for now; the mechanism stays generic.
- Watch: the checkpoint's overall validation metrics (P 0.804, R 0.682, mAP50 0.786) are lower than YOLO11s' (P 0.857, R 0.800, mAP50 0.862), but they are not comparable (4 classes vs 3, possibly a different validation set). Step 8 compares both on our annotated clips before the default is final.
