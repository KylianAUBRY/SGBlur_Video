# HTTP API

The contract is described in [HTTP API contract](../design/api.md) and
[`openapi.yaml`](../design/openapi.yaml); a running service also serves its
interactive schema at `/docs`, and a web page to blur a video from a browser at
`/ui` (`WEB_UI=false` disables it; with `API_TOKEN`, type the token in the page).

## Start the service

=== "Native (macOS, Linux)"

    ```bash
    uv run sgblur-video serve            # Blur API on :8000 + one worker
    ```

=== "Docker"

    ```bash
    docker compose -f docker/docker-compose.yml up --build
    ```

## Blur a video (asynchronous)

```bash
# 1. Upload: returns 202 and the job id
curl -s -F video=@my-video.mp4 "http://localhost:8000/blur/?frames=1"
# {"job_id": "3f0d…", "status": "queued", …}

# 2. Follow the progress
curl -s http://localhost:8000/jobs/3f0d…
# {"status": "running", "phase": "analyzing", "progress": {"percent": 41.7, …}, "eta_s": 655, …}

# 3. Download the results
curl -s -o blurred.mp4 http://localhost:8000/jobs/3f0d…/video
curl -s http://localhost:8000/jobs/3f0d…/metadata      # Panoramax annotations of traffic signs
curl -s http://localhost:8000/jobs/3f0d…/frames        # best-frame pictures (frames=1)
curl -s -o debug.mp4 http://localhost:8000/jobs/3f0d…/debug   # outlined debug video (debug=1)

# 4. Delete everything now (otherwise results expire after RESULT_TTL_MINUTES)
curl -s -X DELETE http://localhost:8000/jobs/3f0d…
```

## Short videos (synchronous)

Videos up to `SYNC_MAX_DURATION_S` (30 s) can be processed in one request.
With `Accept: multipart/form-data` (what the Panoramax backend sends) the
response holds a `metadata` JSON part and a `video` part, like SGBlur's
`metadata` + `image` parts:

```bash
curl -s -H "Accept: multipart/form-data" -F video=@clip.mp4 "http://localhost:8000/blur/?sync=1" -o response.multipart
curl -s -F video=@clip.mp4 "http://localhost:8000/blur/?sync=1" -o blurred.mp4   # video only
```

Results of synchronous requests are deleted as soon as they are sent.

## Parameters

| Parameter | Meaning |
|---|---|
| `keep=1` | Keep encrypted originals of low-confidence blurred regions for `KEEP_TTL_HOURS` (needs `KEEP_SECRET_KEY`). |
| `frames=1` | Also produce one blurred JPEG per sign best view. |
| `debug=1` | Also produce a debug video at `/jobs/{id}/debug` (`links.debug` in the status): colour = class (magenta face, yellow plate, blue sign, cyan direction sign); solid outline = detected on that frame, labelled with class, score and track number; dashed outline, without label = blurred without a detection on that frame, from the object's track (interpolated between two detections, or padded before or after them). H.264 at most 1920 px wide, so it plays in every browser. `DEBUG_VIDEOS=false` disables it. |
| `sync=1` | Wait for the result (short videos only). |
| `callback_url=…` | `POST` the final job status there (host must be in `CALLBACK_ALLOWED_HOSTS`). |
| `start_frame=N`, `end_frame=M` | Only process frames `N` to `M - 1` (0-based; either can be omitted). The rest of the upload is deleted before processing; the result holds only those frames, with the audio and telemetry of the same span, and its times count from frame `N` (`video.frame_range` in the metadata). A GoPro telemetry packet covers about 1 s, so a very short range can report a slightly longer duration than its video. |

## Errors

Every error is `{"detail": "…", "code": "…"}`, including invalid query
parameters (e.g. `keep=2` → `422 invalid_parameter`):

| Status | Codes |
|---|---|
| 400 | `invalid_parameter` (malformed multipart body) |
| 401 | `unauthorized` (when `API_TOKEN` is set) |
| 404 | `job_not_found`, `frames_not_requested`, `debug_not_requested`, `frame_not_found` |
| 409 | `job_not_ready`, `job_failed` |
| 410 | `job_expired` |
| 413 | `file_too_large` |
| 415 | `unsupported_media_type`, `unsupported_projection` |
| 422 | `invalid_parameter`, `video_too_long`, `sync_too_long`, `keep_unavailable`, `debug_unavailable`, `callback_not_allowed` |
| 500 | `processing_error`, `worker_crash` (only with `sync=1`, when the job fails) |
| 503 | `queue_full`, `detector_busy` (with `Retry-After`) |

A failed job reports `error.code` in its status: one of the input codes above,
`detection_failed`, `processing_error`, `timeout`, `worker_crash` or `cancelled`. With
`sync=1`, a job that fails on its input answers `422` with that code.

## Remote detection (split mode)

On a GPU machine, run the Detect API:

```bash
uv run sgblur-video serve-detect --host 0.0.0.0     # or the `detect` Compose service
```

and point the worker at it with `DETECT_URL=http://gpu-host:8001`. The worker
streams the video to `POST /detect/` and receives `detections.jsonl` back,
line by line; the detector handles one video at a time and answers
`503 detector_busy` otherwise (the worker retries).
