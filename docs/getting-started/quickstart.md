# Quick start

## Command line

```bash
uv run sgblur-video models download yolo26s   # once: downloads and verifies the SGBlur model
uv run sgblur-video blur my-video.mp4 blurred.mp4 --debug --frames-dir frames/
```

- `blurred.mp4`: faces and plates blurred on every frame; audio, GoPro GPS
  telemetry and 360° metadata kept;
- `blurred.metadata.json`: one Panoramax annotation per traffic sign
  ([Annotations and Panoramax](../concepts/annotations.md));
- `blurred.debug.mp4`: the same video with every blurred region and every sign
  outlined (see [Tracking and post-processing](../concepts/tracking.md));
- `frames/`: the best view of each sign as a blurred JPEG, with `frames.json`.

Try the first seconds of a long video with `--max-frames 90`. Settings come
from environment variables, e.g. `CONF_BLUR=0.1 uv run sgblur-video blur …`
([configuration](../reference/configuration.md)).

On an Apple M4 Pro (MPS), end-to-end processing (analysis and rendering) runs
at about 10 frames/s for 1080p and 1.5 to 2 frames/s for 8K 360° video.

## HTTP API

```bash
uv run sgblur-video serve                      # API on :8000 with one worker
curl -s -F video=@my-video.mp4 http://localhost:8000/blur/   # → {"job_id": "…", "status": "queued", …}
curl -s http://localhost:8000/jobs/<job_id>                   # progress and ETA
curl -s -o blurred.mp4 http://localhost:8000/jobs/<job_id>/video
curl -s http://localhost:8000/jobs/<job_id>/metadata         # sign annotations
```

Every route, option and error code: [HTTP API](../usage/api.md).
