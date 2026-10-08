# Testing strategy

> Status: **accepted**. Decision record:
> [ADR-0010](../adr/0010-privacy-testing-strategy.md). Maintainer choice
> (2026-10-06): **A** (synthetic oracle) + **D** (manually annotated ground truth).
> A runs on every push since step 4; the tooling of D (`annotate`, `benchmark`)
> exists since step 8, see the [benchmarks guide](../guides/benchmarks.md).

## Why ground truth is needed

The privacy goal is a number: the share of frames where a real face or plate
is still visible after blurring. A miss is, by definition, something our
detector did not see, so it cannot be found by looking at our own outputs. We
need a reference built independently of the model:

- **A — synthetic oracle**: we generate videos where we *know* where the objects are;
- **D — human annotation**: a person marks the real faces and plates on a few real clips.

## Test pyramid

| Level | Runs | GPU | Data | Content |
|---|---|---|---|---|
| Unit (`tests/unit`) | every push (GitHub Actions) | no | in-memory | geometry (wrap-aware IoU, SGBlur-style merge), the per-frame blur plan (score threshold, minimum size, 360° seam), wrap-aware sign linking and dedup, annotation format, config validation, MP4 box editor on tiny generated files, job store state machine. Coverage ≥ 85 % on `core/`, `privacy/`, `semantics/`. |
| Synthetic oracle (A, `tests/privacy`) | every push | no | generated on the fly | see below — the CI privacy gate. |
| Integration (`tests/integration`) | every push (CPU, short clips) | no | small free fixtures, downloaded and SHA-256 checked, cached | real model on CPU on a few seconds: same frame count, same duration (±1 frame), same timestamps, audio present and bit-identical, `gpmd` stream present and parsable, spherical metadata present, rotation preserved, no unknown boxes copied. |
| HTTP API (`tests/integration/test_api.py`) and Docker | every push | no | generated clip | in-process FastAPI client: upload → poll → download → checks, `sync=1`, errors, cleanup; CI also builds the CPU image and checks its health endpoint. A full `docker compose up` + `curl` run was checked manually in step 6. |
| Privacy benchmark (D) | on demand, locally or on a self-hosted runner | recommended | annotated real clips, **outside git** | leakage metrics on real footage, gate on thresholds. |

## A — Synthetic oracle (CI privacy gate)

*Implemented in step 4: `tests/privacy/`, revised for per-frame blurring
([ADR-0012](../adr/0012-independent-frames.md)). It runs on every push, for
H.264 8-bit and HEVC 10-bit, plus a negative control proving that the 360°
scenario leaks without seam handling.*

`tests/privacy/synthetic.py` generates short videos (flat 1920×1080 and
equirect 3840×1920, H.264 and HEVC 10-bit, CFR and VFR) containing moving
"objects": high-frequency textured patches (checkerboards, text-like noise)
whose position is known on every frame, including patches that cross the
360° seam, accelerate, shrink and disappear.

A **scripted fake detector** replaces YOLO and reports those objects with
realistic defects:

- random misses (1 frame in 5), a 10-frame hole, late onset (first 8 frames missed) and early release (last 8 frames missed);
- scores below `CONF_DETECT` on some frames;
- jittered and slightly too small boxes;
- duplicate detections from different passes;
- signs that must never be blurred, one of them overlapping a face.

The real pipeline (merge, sign tracking, blur plan, rendering, encoding) runs
on those inputs, then the oracle checks the **output video**: inside every face
and plate box the fake detector reported (score ≥ `CONF_DETECT`), the
high-frequency energy must have dropped below a threshold (the texture is
destroyed). Frames where an object was not detected must keep its texture:
nothing is carried between frames. This checks the full chain down to the
encoded file, without any personal data and without a GPU. Sign patches must
keep their energy (anti-blur test).

Gate: **0 leaked detections** (a reported face or plate left sharp on its
frame), and 0 blurred sign pixels outside face/plate overlaps. Missed frames
are measured on real footage by the privacy benchmark (D).

## D — Annotated real clips (privacy benchmark)

*Implemented in step 8: `sgblur_video.bench`, commands `annotate export|import`
and `benchmark privacy|trackers|speed`. The metrics below are computed on the
blur plan (what the renderer is asked to blur); the oracle A guarantees that
the renderer blurs every planned region. The same benchmark also runs on the
synthetic scenario in CI (`tests/privacy/test_bench_synthetic.py`) with exact
ground truth.*

### Dataset

- **Source**: the maintainer's own videos (Q360 8K equirect, GoPro HERO12), later possibly clips from contributors. They contain identifiable people: they are personal data and **never enter git** or CI logs.
- **Size (first version)**: 3 to 5 clips of 10–20 s, chosen for difficulty: pedestrians close and far, parked cars, a crossing at the 360° seam, a sign next to a person.
- **Storage**: a local folder (e.g. `~/sgblur-video-privacy/`) with `manifest.yaml` (clip id, source file SHA-256, start/end time, annotator, date) and one annotation file per clip. The benchmark command takes its path: `sgblur-video benchmark privacy --dataset ~/sgblur-video-privacy`.

### Annotation workflow

1. `sgblur-video annotate export <video> --dataset DIR --id ID --start 30 --duration 15` cuts the clip, writes a proxy at ≤ 3840 px wide (8K is too heavy for annotation tools; coordinates are scaled back ×2) and a **pre-annotation** from the model. *Changed in step 8:* the brief suggested a very low threshold, but on 8K 360° clips that gave 600–800 mostly spurious tracks for 10 s; the pre-annotation keeps tracks with a best score ≥ 0.25 detected on ≥ 5 frames, and the annotator draws what is missing.
2. The annotator opens it in **CVAT** (MIT, runs locally with Docker) in track mode: boxes are adjusted on key frames and CVAT interpolates in between, so a 15 s clip takes roughly 15–30 min.
3. Instructions: draw a box on **every** face (any orientation where part of the face is visible, including profile, partially hidden, tiny or blurry) and **every** plate (readable or not); tick `readable` if a human could recognise the person / read the plate. Pay special attention to objects the pre-annotation **missed**: they are exactly what we measure. Do not annotate signs.
4. `sgblur-video annotate import` converts the CVAT export ("CVAT for video 1.1" XML) to the dataset format and records its SHA-256 in the manifest.

Pre-annotation by our own model speeds things up but can bias the annotator
towards the model's misses; the instructions and a second pass by another
person (when available) mitigate it.

### Metrics

For each ground-truth object-frame (box `G`) and the blur mask `M` actually
applied on that frame:

- **coverage** = |G ∩ M| / |G| (not IoU: blurring more than needed is harmless);
- the object-frame is **protected** if coverage ≥ *0.9*;
- **leakage rate** = unprotected object-frames / all object-frames, reported overall, per class, per size bucket (< 16 px, 16–32, 32–96, > 96 px high) and for `readable` objects only;
- **tracks ever leaked** = share of ground-truth tracks with ≥ 1 unprotected frame;
- **longest exposure** = longest run of consecutive unprotected frames;
- **transient exposures** = unprotected frames with protection within ⅓ s before and after (one-off misses of the detector, visible as a flicker);
- **over-blur ratio** = blurred pixels outside any ground-truth box / frame pixels (cost side, not gated).

### Gate

Initial thresholds in `benchmarks/privacy-thresholds.yaml`, to tighten as the
dataset grows: leakage rate of `readable` objects ≤ *1 %*, 0 tracks of
`readable` objects ever leaked for more than *3* consecutive frames. Every
change to a privacy-related default must report these metrics before/after.

## Data handling rules for tests

- No real person's image in the repository, test logs, CI artefacts or issues.
- Free-licence fixtures (`gopro/gpmf-parser` samples, CC0 clips) are downloaded by `tests/data/fetch.py` with pinned URLs and SHA-256, cached in CI, and listed in `THIRD_PARTY_LICENSES.md`.
- Debug videos produced locally (`--debug`) are written outside the repository and the `.gitignore` blocks video and image extensions at the repository root.
