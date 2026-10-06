# Command line

Commands are listed by `sgblur-video --help`. Those not implemented yet exit
with code 2 and say which roadmap step brings them. Errors on bad inputs (not a
video, unsupported 360° format, incomplete `detections.jsonl`…) exit with code 1
and a one-line message.

```bash
uv run sgblur-video models download                       # fetch and verify the model once
uv run sgblur-video blur input.mp4 output.mp4 --debug     # full pipeline + annotated debug video
uv run sgblur-video detect input.mp4 --out detections.jsonl
uv run sgblur-video render input.mp4 detections.jsonl output.mp4
uv run sgblur-video blur input.mp4 output.mp4 --max-frames 90   # quick try on the first 3 s
```

| Command | Purpose | Step |
|---|---|---|
| `sgblur-video blur IN OUT [--model] [--tracker] [--debug] [--frames-dir DIR] [--keep-detections F] [--max-frames N]` | Full pipeline; also writes `OUT.metadata.json` | ✅ 4–5 |
| `sgblur-video detect IN --out detections.jsonl` | Pass 1 only | ✅ 4 |
| `sgblur-video render IN detections.jsonl OUT [--debug] [--frames-dir DIR] [--allow-partial]` | Post-processing and pass 2; also writes `OUT.metadata.json` | ✅ 4–5 |
| `sgblur-video signs IN --out signs.json [--frames-dir DIR]` | Sign annotations only (no video written) | ✅ 5 |
| `sgblur-video benchmark --dataset DIR` | Benchmarks | 8 |
| `sgblur-video serve [--host] [--port] [--workers]` / `worker` / `serve-detect` | Blur API with workers / job worker / Detect API | ✅ 6 |
| `sgblur-video models list` / `download` | Model registry | ✅ 3 / 4 |
| `sgblur-video annotate export` / `import` | Privacy dataset tooling | 8 |
| `sgblur-video config` / `version` | Effective configuration / version | ✅ 3 |

`--debug` writes an annotated video (boxes, classes, track ids, interpolated
boxes in another colour). It is **only** available from the CLI, never through
the HTTP API.
