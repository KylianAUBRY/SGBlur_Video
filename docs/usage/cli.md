# Command line

Commands are listed by `sgblur-video --help`, and each one documents its
options with `--help`. Errors on bad inputs (not a video, unsupported 360°
format, incomplete `detections.jsonl`…) exit with code 1 and a one-line message.

```bash
uv run sgblur-video models download yolo26s               # fetch and verify the model once
uv run sgblur-video blur input.mp4 output.mp4 --debug     # full pipeline + annotated debug video
uv run sgblur-video detect input.mp4 --out detections.jsonl
uv run sgblur-video render input.mp4 detections.jsonl output.mp4
uv run sgblur-video blur input.mp4 output.mp4 --max-frames 90   # quick try on the first 3 s
```

## Settings

The CLI reads the same settings as the service: environment variables, or a
`.env` file in the current folder ([configuration reference](../reference/configuration.md)).
For example, to blur more aggressively on one run:

```bash
CONF_BLUR=0.1 BLUR_TEMPORAL_PADDING_FRAMES=20 uv run sgblur-video blur input.mp4 output.mp4
```

`sgblur-video config` prints the effective settings (secrets masked, home
folder shown as `~`). Run commands from the repository root, or set
`MODELS_FILE` and `TRACKER_CONFIG` to absolute paths.

## Processing

| Command | Purpose |
|---|---|
| `blur IN OUT [--model NAME] [--tracker YAML] [--debug] [--frames-dir DIR] [--keep-detections FILE] [--max-frames N]` | Full pipeline. Also writes `OUT.metadata.json` (sign annotations) and, with `--debug`, `OUT.debug.mp4`. |
| `detect IN --out detections.jsonl [--model] [--tracker] [--max-frames N]` | Pass 1 only ([format](../design/detections-format.md)). |
| `render IN detections.jsonl OUT [--debug] [--frames-dir DIR] [--allow-partial]` | Post-processing and pass 2 from existing detections. Also writes `OUT.metadata.json`. |
| `signs IN --out signs.json [--model] [--tracker] [--frames-dir DIR] [--max-frames N]` | Sign annotations only, no video written. |

`--frames-dir` writes the best view of each sign as a blurred JPEG (with EXIF
date and GPS when known) and `frames.json`. `--debug` writes an annotated video
(boxes, classes, track ids, interpolated and padded boxes in other colours). It
is **only** available from the CLI, never through the HTTP API.

## Service

| Command | Purpose |
|---|---|
| `serve [--host] [--port 8000] [--workers 1]` | Blur API with worker processes. Jobs run in parallel up to the number of workers. |
| `worker` | One job worker (for deployments where the API and workers run separately). |
| `serve-detect [--host] [--port 8001]` | Detect API, for remote analysis on a GPU machine ([split mode](api.md#remote-detection-split-mode)). |

## Models and configuration

| Command | Purpose |
|---|---|
| `models list` | Models of the registry (`MODELS_FILE`). |
| `models download [NAME]` | Download and verify one model, or every model of the registry. |
| `config` / `version` | Effective configuration / version. |

## Benchmarks and the privacy dataset

See the [benchmarks guide](../guides/benchmarks.md).

| Command | Purpose |
|---|---|
| `annotate export IN --dataset DIR --id ID [--start S] [--duration S] [--proxy-width 3840] [--conf 0.25] [--min-frames 5] [--model]` | Cut a clip, its CVAT proxy and a pre-annotation into the dataset. |
| `annotate preannotate --dataset DIR --id ID [--conf] [--min-frames] [--model]` | Rebuild a clip's pre-annotation with other thresholds. |
| `annotate import XML --dataset DIR --id ID [--annotator NAME]` | Convert a "CVAT for video 1.1" export into ground truth. |
| `benchmark privacy --dataset DIR [--sweep NAME=v1,v2]… [--model] [--tracker] [--thresholds YAML] [--report-dir DIR]` | Leakage against the annotated dataset; exit code 1 if the first run fails the gate. |
| `benchmark trackers [--video F]… [--dataset DIR] [--tracker YAML]… [--model] [--max-frames N] [--cache-dir DIR] [--report-dir DIR]` | Trackers compared on the same detections. |
| `benchmark speed IN [--model]… [--device]… [--profile]… [--frames 20] [--report-dir DIR]` | Detection speed per model, device and profile. |
