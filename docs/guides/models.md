# Models and new YOLO versions

SGBlur-Video does not train its own detector: it uses the models published by
[SGBlur](https://gitlab.com/panoramax/server/sgblur) (see
[ADR-0009](../adr/0009-model-registry-and-class-policy.md)). Models are declared
in `models/registry.yaml`; no model name is hard-coded anywhere else.

## Choosing a model

Three variables, by precedence (the `--model` option of a command beats them all):

| Variable | Value | Use |
|---|---|---|
| `MODEL_PATH` | Path of a `.pt` checkpoint | Try **any** model without registering it |
| `MODEL_NAME` | A registry name (`sgblur-video models list`), or a `.pt` path | Pick a registered model |
| `MODEL_FAMILY` | `yolo26` (default), `yolo11` | Automatic choice: the largest model of the family that fits the accelerator memory |

```bash
uv run sgblur-video models list                            # registered models; * marks the one in use
MODEL_NAME=yolo11l uv run sgblur-video blur in.mp4 out.mp4  # a registered model (downloaded and verified)
MODEL_PATH=~/models/candidate.pt uv run sgblur-video serve  # the web page and API with a local file
uv run sgblur-video blur in.mp4 out.mp4 --model ~/models/candidate.pt
```

Registered models: `yolo26s` (default, the only one with the `direction` class),
and the older SGBlur `yolo11n`, `yolo11s`, `yolo11m`, `yolo11l`.

A local checkpoint is used in place. Its classes are read from the file and
must include every class configured to be blurred (`face`, `plate`): otherwise
it is refused at start-up. Its Panoramax tag is `SGBlur-Video-<file name>/local-<8 hex digits of its SHA-256>`,
so annotations made by two different files can be told apart. Check a file
before using it:

```bash
uv run sgblur-video models inspect ~/models/candidate.pt   # classes, SHA-256, usable or not, registry entry
```

### Comparing models

Every benchmark takes several `--model` options, names or paths:

```bash
uv run sgblur-video benchmark speed video.mp4 --model yolo26s --model yolo11l --model ~/models/candidate.pt
uv run sgblur-video benchmark privacy --dataset ~/sgblur-video-privacy --model ~/models/candidate.pt
```

The speed benchmark counts detections, which says nothing about missed faces:
only the privacy benchmark on annotated clips measures that. Reports name
models by tag, never by path.

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

1. Try it first with `MODEL_PATH` (above). `sgblur-video models inspect model.pt`
   prints its hash, size and classes, and a registry entry to complete.
2. Read its class names (`model.names` in Ultralytics) and check that `face` and
   `plate` are present: a model without a class configured to be blurred is
   refused at start-up.
3. Add an entry to `models/registry.yaml` with a commit-pinned URL.
4. If the model needs a newer `ultralytics`, bump the exact pin in
   `pyproject.toml`, run `uv lock`, and run the full test suite: the tracker
   configuration test and the adapter contract tests catch API changes.
5. Run the privacy benchmark on the annotated dataset and compare with the
   current default (`sgblur-video benchmark privacy --dataset … --model <name>`,
   see [Benchmarks](benchmarks.md)), and its speed with `sgblur-video benchmark speed`.
6. If it becomes the default, change `MODEL_FAMILY`'s default and record the
   decision in a new ADR.

## Fine-tuning

Not part of the project for now. `training/README.md` explains when it would
become necessary (privacy benchmark gaps) and how it would be organised.
