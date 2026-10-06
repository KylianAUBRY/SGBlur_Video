# Models and new YOLO versions

SGBlur-Video does not train its own detector: it uses the models published by
[SGBlur](https://gitlab.com/panoramax/server/sgblur) (see
[ADR-0009](../adr/0009-model-registry-and-class-policy.md)). Models are declared
in `models/registry.yaml`; no model name is hard-coded anywhere else.

## Registry entry fields

| Field | Meaning |
|---|---|
| `name` | Short unique name, used by `MODEL_NAME` and in `detection_model` tag values |
| `family` | Family used by automatic selection (`MODEL_FAMILY`), e.g. `yolo26` |
| `version` | Version published in tags: `SGBlur-Video-<name>/<version>` |
| `file` | File name in `MODELS_DIR` |
| `url` | Download URL **pinned to a commit** (never a branch) |
| `sha256` | SHA-256 of the file; a mismatch refuses the model |
| `size_bytes` | Expected size |
| `classes` | Class names provided by the checkpoint, in its own order |
| `train_imgsz` | Training image size (informative) |
| `min_memory_gib` | Free accelerator memory needed by the standard profile (automatic selection) |
| `licence`, `source` | Upstream licence statement and project page |

## Adding a model (for example a YOLO27 release of SGBlur)

1. Download the checkpoint and compute its hash: `shasum -a 256 model.pt`.
2. Read its class names (`model.names` in Ultralytics) and check that `face` and
   `plate` are present: a model without a class configured to be blurred is
   refused at start-up.
3. Add an entry to `models/registry.yaml` with a commit-pinned URL.
4. If the model needs a newer `ultralytics`, bump the exact pin in
   `pyproject.toml`, run `uv lock`, and run the full test suite: the tracker
   configuration test and the adapter contract tests catch API changes.
5. Run the privacy benchmark on the annotated dataset and compare with the
   current default (`sgblur-video benchmark`, step 8).
6. If it becomes the default, change `MODEL_FAMILY`'s default and record the
   decision in a new ADR.

## Fine-tuning

Not part of the project for now. `training/README.md` explains when it would
become necessary (privacy benchmark gaps) and how it would be organised.
