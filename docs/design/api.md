# HTTP API contract

> Status: **draft for validation (step 2)**. The machine-readable contract is
> [`openapi.yaml`](openapi.yaml) (OpenAPI 3.1). In step 6 the FastAPI app
> generates its own schema from Pydantic models, and a test checks it against
> this draft so the two cannot drift silently.

Two applications, as in SGBlur:

| App | Default port | Routes |
|---|---|---|
| Blur API | 8000 | `GET /`, `POST /blur/`, `GET /jobs/{job_id}`, `GET /jobs/{job_id}/video`, `GET /jobs/{job_id}/metadata`, `GET /jobs/{job_id}/frames`, `GET /jobs/{job_id}/frames/{n}.jpg`, `DELETE /jobs/{job_id}`, `GET /metrics` |
| Detect API | 8001 | `GET /`, `POST /detect/`, `GET /metrics` |

## Conventions

- **Uploads** are `multipart/form-data` with the file in the field `video` (SGBlur uses `picture` for images). Files are streamed to disk in chunks; the size limit is enforced while streaming.
- **Errors** follow FastAPI's shape with an added machine-readable code: `{"detail": "human message", "code": "video_too_long"}`.
- **Job ids** are random UUID4 values; knowing the id is what grants access to a job (capability URL). An optional static bearer token (`API_TOKEN`) can be required on every route except `GET /` and `GET /metrics`.
- **No public debug switch**: unlike SGBlur's `debug` parameter, annotated debug videos are only available from the CLI.

## `POST /blur/`

| Parameter | In | Type | Default | Meaning |
|---|---|---|---|---|
| `video` | form | file | required | MP4 or MOV, ≤ `MAX_UPLOAD_BYTES` |
| `keep` | query | `0` \| `1` | `0` | Keep encrypted originals of low-confidence blurred regions for `KEEP_TTL_HOURS` (Panoramax `PICTURE_PROCESS_KEEP_UNBLURRED_PARTS`). `422 keep_unavailable` if `KEEP_SECRET_KEY` is not configured. |
| `sync` | query | `0` \| `1` | `0` | Wait and return the result in the response. Only for videos ≤ `SYNC_MAX_DURATION_S`, otherwise `422 sync_too_long`. |
| `frames` | query | `0` \| `1` | `0` | Also produce one blurred JPEG per sign annotation (best frame). |
| `callback_url` | query | URL | none | Called with `POST` and the job status JSON when the job ends. Host must match `CALLBACK_ALLOWED_HOSTS` (empty = callbacks disabled → `422 callback_not_allowed`). |

Responses:

- `202 Accepted` (asynchronous): job status body (see below) and `Location: /jobs/{job_id}`.
- `200 OK` (`sync=1`):
  - if `Accept` contains `multipart/form-data` (what the Panoramax backend sends): a multipart body with a `metadata` part (`application/json`) and a `video` part (`video/mp4`), mirroring SGBlur's `metadata` + `image` parts;
  - otherwise the video (`video/mp4`) with header `X-SGBlur-Video-Job: {job_id}`; metadata stays available at `/jobs/{job_id}/metadata` until the result TTL.
- `413 file_too_large`, `415 unsupported_media_type`, `415 unsupported_projection` (GoPro `.360`, Insta360 `.insv`, cubemap…), `422 video_too_long`, `422 invalid_parameter`, `503 queue_full` (with `Retry-After`).

## `GET /jobs/{job_id}`

```json
{
  "job_id": "3f0d2c3e-8a51-4c1f-9a4e-0d6f8f4f5b21",
  "status": "running",
  "phase": "analyzing",
  "progress": {"percent": 41.7, "frames_done": 1205, "frames_total": 2893},
  "eta_s": 655,
  "created_at": "2026-10-06T14:00:02Z",
  "started_at": "2026-10-06T14:00:05Z",
  "finished_at": null,
  "expires_at": null,
  "error": null,
  "links": {
    "self": "/jobs/3f0d2c3e-8a51-4c1f-9a4e-0d6f8f4f5b21",
    "video": "/jobs/3f0d2c3e-8a51-4c1f-9a4e-0d6f8f4f5b21/video",
    "metadata": "/jobs/3f0d2c3e-8a51-4c1f-9a4e-0d6f8f4f5b21/metadata"
  }
}
```

- `status`: `queued`, `running`, `succeeded`, `failed`, `cancelled`, `expired`.
- `phase` (while running): `analyzing`, `postprocessing`, `rendering`, `finalizing`.
- `progress.percent` weights analysis and rendering by their measured share of the job duration (initially *70 / 30*); `eta_s` uses the moving frame rate of the current phase.
- `error`: `{"code": "decode_error", "message": "…"}` without paths or user data.
- `404 job_not_found`; `410 job_expired` once files are gone.

## `GET /jobs/{job_id}/video`

`200 video/mp4` (with `Content-Disposition: attachment; filename="blurred.mp4"`
and HTTP range support), `409 job_not_ready`, `404`, `410`.

## `GET /jobs/{job_id}/metadata` {#metadata}

Same structure as SGBlur's `metadata` part, with documented extensions:

```json
{
  "blurring_id": "56e83230-4e88-4700-9311-18a8197088bb",
  "service_name": "SGBlur-Video",
  "annotations": [
    {
      "shape": [2201, 1502, 2291, 1591],
      "semantics": [
        {"key": "osm|traffic_sign", "value": "yes"},
        {"key": "detection_model[osm|traffic_sign=yes]", "value": "SGBlur-Video-yolo26s/0.1.0"},
        {"key": "detection_confidence[osm|traffic_sign=yes]", "value": "0.874"}
      ],
      "video": {
        "track_id": "signage:3",
        "class": "sign",
        "best_frame": 431,
        "best_timestamp": 14.367,
        "first_frame": 402,
        "first_timestamp": 13.4,
        "last_frame": 470,
        "last_timestamp": 15.667,
        "observations": 61,
        "confidence_max": 0.874,
        "confidence_mean": 0.731,
        "position": {"lat": 45.18834, "lon": 5.72452, "alt": 212.4, "source": "gpmf"}
      }
    }
  ],
  "video": {
    "width": 7680, "height": 3840, "duration_s": 96.43, "frame_count": 2893,
    "projection": "equirectangular", "telemetry": "none"
  },
  "stats": {
    "tracks": {"face": 14, "plate": 37, "signage": 22},
    "blurred_boxes": {"detected": 4021, "interpolated": 311, "padded": 2290, "orphan": 344},
    "frames_with_blur": 2410,
    "dropped_streams": [],
    "processing_s": 1212.5,
    "model": "yolo26s/0.1.0",
    "tracker": "tracktrack-recall"
  }
}
```

Compatibility with the Panoramax picture contract
([blur API spec](https://docs.panoramax.fr/backend/install/deep_dive/blur_api/)):

| Field | Status |
|---|---|
| `blurring_id`, `service_name`, `annotations[].shape`, `annotations[].semantics` | **Same as SGBlur**: integer pixel bbox `[minx, miny, maxx, maxy]` (display orientation of the video frame), string values, same tag keys. |
| `service_name` = `API_NAME` = `SGBlur-Video` | Model tag values start with `SGBlur-`, which is the prefix the current backend uses to clean previous detection tags (it ignores `service_name` itself). |
| `annotations[].video` | **Extension.** The current backend ignores unknown annotation fields (Pydantic default), so it is safe but not stored. |
| `video`, `stats` | **Extension**, top-level keys are tolerated by the backend. |
| `direction` signs | Annotated with the same tags as `sign` (maintainer decision 2026-10-06) until SGBlur defines specific semantics. |

Faces and plates never produce annotations (same as SGBlur).

## `GET /jobs/{job_id}/frames` and `/frames/{n}.jpg`

Only when the job was created with `frames=1`.

```json
{"frames": [
  {"n": 0, "url": "/jobs/3f0d…/frames/0.jpg", "annotation_index": 0, "frame": 431,
   "timestamp": 14.367, "width": 7680, "height": 3840, "shape": [2201, 1502, 2291, 1591],
   "position": null}
]}
```

Each JPEG is a frame of the **blurred** output (faces and plates in it are
blurred), in display orientation, with EXIF `DateTimeOriginal` and GPS when
known, and the same annotation shape in its own pixel coordinates.

## `DELETE /jobs/{job_id}`

`204`: cancels a queued or running job (the job process is killed) and deletes
every file of the job immediately. Idempotent: deleting an expired job also
returns `204`.

## `GET /`

```json
{"name": "SGBlur-Video", "version": "0.1.0", "status": "ok",
 "model": {"name": "yolo26s", "version": "0.1.0"}, "tracker": "tracktrack-recall",
 "device": "mps", "queue": {"queued": 0, "running": 1}}
```

## `GET /metrics`

Prometheus text format, computed from the job store (no multi-process
registry): `sgblur_video_jobs_total{status}`, `sgblur_video_jobs_in_progress`,
`sgblur_video_queue_length`, `sgblur_video_job_duration_seconds` (histogram),
`sgblur_video_frames_processed_total{phase}`, `sgblur_video_processing_fps{phase}`,
`sgblur_video_accelerator_memory_bytes{device}` (reported by the worker
heartbeat), `sgblur_video_blurred_boxes_total{source}`.

## Detect API — `POST /detect/`

Multipart field `video`; returns `200 application/x-ndjson` streaming the
[`detections.jsonl`](detections-format.md) lines as they are produced.
Processes one video at a time per process: `503 detector_busy` with
`Retry-After` otherwise; the worker retries with backoff. The received file is
deleted as soon as the response ends, including on client disconnect.
