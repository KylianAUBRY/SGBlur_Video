# Command line

!!! note "Planned"
    Commands are listed by `sgblur-video --help`. Those not implemented yet exit
    with code 2 and say which roadmap step brings them.

| Command | Purpose | Step |
|---|---|---|
| `sgblur-video blur IN OUT [--model] [--tracker] [--debug]` | Full pipeline | 4 |
| `sgblur-video detect IN --out detections.jsonl` | Pass 1 only | 4 |
| `sgblur-video render IN detections.jsonl OUT` | Post-processing and pass 2 | 4 |
| `sgblur-video signs IN --out signs.json` | Sign annotations only | 5 |
| `sgblur-video benchmark --dataset DIR` | Benchmarks | 8 |
| `sgblur-video worker` / `serve` | Job worker / API with one worker | 6 |
| `sgblur-video models list` / `download` | Model registry | 3 / 4 |
| `sgblur-video annotate export` / `import` | Privacy dataset tooling | 8 |
| `sgblur-video config` / `version` | Effective configuration / version | 3 |

`--debug` writes an annotated video (boxes, classes, track ids, interpolated
boxes in another colour). It is **only** available from the CLI, never through
the HTTP API.
