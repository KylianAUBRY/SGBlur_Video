# Quick start

## Command line

```bash
uv run sgblur-video models download yolo26s   # once: downloads and verifies the SGBlur model
uv run sgblur-video blur my-video.mp4 blurred.mp4 --debug --frames-dir frames/
```

- `blurred.mp4`: faces and plates detected on each frame blurred; audio, GoPro GPS
  telemetry and 360° metadata kept;
- `blurred.metadata.json`: one Panoramax annotation per traffic sign
  ([Annotations and Panoramax](../concepts/annotations.md));
- `blurred.debug.mp4`: the same video with every blurred region and every sign
  outlined (see [Frame-by-frame blurring](../concepts/tracking.md));
- `frames/`: the best view of each sign as a blurred JPEG, with `frames.json`.

Try the first seconds of a long video with `--max-frames 90`. Settings come
from environment variables, e.g. `CONF_DETECT=0.2 uv run sgblur-video blur …`
([configuration](../reference/configuration.md)).

On an Apple M4 Pro (MPS), end-to-end processing (analysis and rendering) runs
at about 10 frames/s for 1080p and about 3 frames/s for 8K 360° video (analysis 3.9 frames/s, rendering 14 frames/s).

## Web page

```bash
uv run sgblur-video serve
```

Open <http://localhost:8000/ui>, drop a video, optionally choose a range of
frames with the bar under the preview (e.g. 20 frames for a quick test), follow
the progress, then watch or download the blurred video, the list of traffic
signs and their best pictures. Tick *Debug video* to also get the video with
every blurred region outlined in the colour of its class (face, plate) and
every sign, to see what the model found. The page only calls the HTTP API below; `WEB_UI=false` disables it.

## HTTP API

```bash
uv run sgblur-video serve                      # API on :8000 with one worker
curl -s -F video=@my-video.mp4 http://localhost:8000/blur/   # → {"job_id": "…", "status": "queued", …}
curl -s http://localhost:8000/jobs/<job_id>                   # progress and ETA
curl -s -o blurred.mp4 http://localhost:8000/jobs/<job_id>/video
curl -s http://localhost:8000/jobs/<job_id>/metadata         # sign annotations
```

Every route, option and error code: [HTTP API](../usage/api.md).
