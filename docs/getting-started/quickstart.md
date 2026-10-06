# Quick start

!!! note "HTTP API"
    The HTTP API arrives in step 6; this page will then add a `curl`
    walkthrough. The command line works today.

```bash
uv run sgblur-video models download          # once: downloads and verifies the SGBlur model
uv run sgblur-video blur my-video.mp4 blurred.mp4 --debug
```

- `blurred.mp4`: faces and plates blurred on every frame, audio and GoPro GPS
  telemetry kept;
- `blurred.debug.mp4`: the same video with every blurred region and every sign
  outlined (see [Tracking and post-processing](../concepts/tracking.md)).

Try the first seconds of a long video with `--max-frames 90`. On an Apple
M4 Pro, 1080p video is analysed at ≈ 11 frames/s and 8K 360° video at
≈ 2.5 frames/s.
