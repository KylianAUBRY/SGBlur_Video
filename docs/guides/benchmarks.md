# Benchmarks and the privacy dataset

Three benchmarks back the default settings (method and metrics:
[testing strategy](../design/testing-strategy.md)):

| Command | Measures | Needs |
|---|---|---|
| `sgblur-video benchmark privacy` | Leakage of faces and plates against human annotation, gated by `benchmarks/privacy-thresholds.yaml` | The annotated dataset |
| `sgblur-video benchmark trackers` | Tracker fragmentation (and leakage on annotated clips), on the same detections | Any video and/or the dataset |
| `sgblur-video benchmark speed` | Detection time per model, device and detection profile | Any video |

Reports print as Markdown and, with `--report-dir`, are also written as JSON
and Markdown. They describe videos only by resolution, projection, codec and
frame count: never by file name, and they never contain pictures.

!!! danger "The dataset contains personal data"
    Annotated clips show real people. Keep the dataset **outside the
    repository**, outside synced folders (iCloud Drive, Dropbox…) if you can,
    never attach clips to issues, and never run the privacy benchmark on public
    CI. Results (numbers only) can be shared.

## Build the dataset

### 1. Cut clips

Choose 3 to 5 clips of 10–20 s that are hard for the detector: pedestrians
close and far, parked cars, an object crossing the 360° seam, a sign next to a
person.

```bash
uv run sgblur-video annotate export ~/Videos/ride.mp4 \
    --dataset ~/sgblur-video-privacy --id q360-market --start 25 --duration 10
```

This writes, in `~/sgblur-video-privacy/clips/q360-market/`:

- `clip.mp4`: the clip at full resolution, re-encoded from the exact frames
  (the benchmark processes this file);
- `proxy.mp4`: the same frames at most 3840 px wide (8K is too heavy for
  annotation tools); coordinates are scaled back on import;
- `preannotation.xml`: face and plate tracks found by the model, in "CVAT for
  video 1.1" format, with one key box every 0.5 s. Only tracks whose best score
  reaches `--conf` (0.25) and that were detected on at least `--min-frames`
  (5) frames are kept: on 8K 360° video the model reports hundreds of
  flickering fragments, and deleting false tracks costs more than drawing the
  missed ones. Fragments are linked more loosely than in post-processing, so
  that one object is usually one CVAT track. `annotate preannotate` rebuilds the
  file with other thresholds without cutting the clip again.

and records the clip in `manifest.yaml` (source SHA-256, start, size; no file
name). It also writes `cvat-labels.json` at the root of the dataset. Rotated
phone videos are not supported in the dataset yet.

### 2. Annotate in CVAT

[CVAT](https://github.com/cvat-ai/cvat) (MIT) runs locally with Docker. Use a
local instance: uploading the clips to a hosted service would share personal
data.

```bash
git clone --depth 1 https://github.com/cvat-ai/cvat && cd cvat
docker compose up -d
docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'
# then open http://localhost:8080
```

1. Create a task from `proxy.mp4`. In the label constructor, open the **Raw**
   tab and paste `cvat-labels.json`: two rectangle labels, `face` and `plate`,
   each with a mutable checkbox attribute `readable` (default `false`).
2. Upload `preannotation.xml` (*Actions → Upload annotations → CVAT 1.1*).
3. Correct it in track mode. Instructions:
    - draw a box on **every** face (any visible part: profile, partly hidden,
      tiny, blurry) and **every** plate, readable or not;
    - tick `readable` on the frames where a person could be recognised or the
      plate read;
    - look hardest for what the pre-annotation **missed**: that is exactly
      what the benchmark measures;
    - delete false detections; do not annotate signs;
    - on 360° clips, an object crossing the left/right edge gets one box on
      each side.
4. Export with *Export task dataset → CVAT for video 1.1* (without images).

A 10 s clip takes roughly 20–40 minutes, depending on the crowd.

### 3. Import

```bash
uv run sgblur-video annotate import export.xml \
    --dataset ~/sgblur-video-privacy --id q360-market --annotator "your name"
```

The import checks that the export matches the clip (frame count, proxy size),
interpolates between key frames like CVAT, scales boxes to full resolution and
writes `ground_truth.json`.

## Run the benchmarks

```bash
# Leakage with the current settings; exit code 1 if the gate fails.
uv run sgblur-video benchmark privacy --dataset ~/sgblur-video-privacy

# Compare settings: every combination is evaluated, detections are computed once.
uv run sgblur-video benchmark privacy --dataset ~/sgblur-video-privacy \
    --sweep CONF_BLUR=0.1,0.15,0.25 --sweep BLUR_TEMPORAL_PADDING_FRAMES=10,15

# Trackers on the same detections (tracking is replayed, detection runs once).
uv run sgblur-video benchmark trackers --dataset ~/sgblur-video-privacy
uv run sgblur-video benchmark trackers --video ride.mp4 --max-frames 300

# Speed of models, devices and profiles.
uv run sgblur-video benchmark speed ride.mp4 --model yolo26s --model yolo11s \
    --device mps --device cpu --profile fast --profile standard --frames 20
```

Detections are cached by what changes them (model checksum, profile,
`CONF_DETECT`, tiling, class policy) and track ids by the tracker
configuration, under `<dataset>/cache/` for clips and
`~/.cache/sgblur-video/bench/` for other videos. Sweeping post-processing
settings (`CONF_BLUR`, padding, margins, linking) is therefore almost instant.

## Reading the privacy report

| Column | Meaning |
|---|---|
| leakage | Unprotected object-frames / all object-frames. *Protected* = at least 90 % of the ground-truth box is blurred. |
| readable leakage | Same, for frames ticked `readable`: the gated number (≤ 1 %). |
| tracks ever leaked | Ground-truth objects with at least one unprotected frame. |
| longest exposure (readable) | Longest run of consecutive unprotected readable frames (gate: ≤ 3). |
| transient | Unprotected frames with protection within ⅓ s before and after: what tracking and padding should remove. |
| chains/track | Blur chains covering one ground-truth object (1 = followed as a single object). |
| over-blur | Share of the frame blurred outside every ground-truth box (cost, not gated). |

The benchmark measures detection, tracking and post-processing on the blur
*plan*. That the renderer really destroys the pixels of every planned region is
checked by the synthetic oracle on every push (`tests/privacy`).

Any change to a privacy-related default must come with the privacy report
before and after (see [CONTRIBUTING](https://github.com/KylianAUBRY/SGBlur_Video/blob/main/CONTRIBUTING.md)).
