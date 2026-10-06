# Configuration reference (design)

!!! note "Superseded"
    This was the step-2 design table. The authoritative, always up-to-date
    reference is generated from the code: [Configuration reference](../reference/configuration.md).

> Status: **draft for validation (step 2)**. Values in *italics* are provisional
> and will be tuned in step 8. Settings are read with `pydantic-settings` from
> environment variables and an optional `.env` file, **without prefix** to stay
> consistent with SGBlur (`API_NAME`, `DETECT_URL`, `MODEL_NAME`…).

Privacy impact legend: **↑ risk** = changing the value in that direction can
leave faces/plates visible or keep personal data longer.

## Service identity

| Name | Type | Default | Description | Privacy impact |
|---|---|---|---|---|
| `API_NAME` | str | `SGBlur-Video` | `service_name` in metadata and prefix of `detection_model` tag values. **Never change after deployment**: Panoramax uses the prefix to clean old tags. | none |
| `API_TOKEN` | secret str | empty | If set, required as bearer token on every route except `/` and `/metrics`. | none |

## Model and device

| Name | Type | Default | Description | Privacy impact |
|---|---|---|---|---|
| `MODEL_NAME` | str | empty (auto) | Registry entry to use; auto picks the largest model of `MODEL_FAMILY` that fits the accelerator memory. | Smaller models miss more |
| `MODEL_FAMILY` | str | `yolo26` | Family preferred by auto-selection. | — |
| `MODELS_FILE` | path | `models/registry.yaml` | Model registry (name, version, URL, SHA-256, classes). | — |
| `MODELS_DIR` | path | `~/.cache/sgblur-video/models` (`/models` in Docker) | Where weights are downloaded and verified. | — |
| `DEVICE` | `auto`\|`cpu`\|`cuda`\|`cuda:N`\|`mps` | `auto` | `auto` = CUDA, else MPS, else CPU. | — |
| `HALF` | bool | `auto` | FP16 inference (mapped to Ultralytics `quantize=16`); `auto` = on for CUDA only. | Negligible |
| `CLASS_POLICY` | JSON | `{"face":"blur","plate":"blur","sign":"annotate","direction":"annotate"}` | What to do with each class name. A model lacking a `blur` class is refused. | **↑ risk** if a blur class is removed |

## Detection

| Name | Type | Default | Description | Privacy impact |
|---|---|---|---|---|
| `DETECT_PROFILE` | `fast`\|`standard`\|`thorough` | `standard` | Passes per frame (see pipeline §2.2). | `fast` ↑ risk on small faces/plates |
| `TILE_TRIGGER_WIDTH` | int | *5760* | Long side from which the tile pass runs. | Higher ↑ risk |
| `EQUIRECT_PAD_RATIO` | float | *0.0625* | Circular padding on each side for 360° video (fraction of width). | Lower ↑ risk at the seam |
| `PROJECTION` | `auto`\|`flat`\|`equirectangular` | `auto` | Force the projection when metadata is missing or wrong. | Wrong value ↑ risk at the seam |
| `CONF_DETECT` | float | *0.10* | Minimum detector score kept (also fed to trackers). | Higher ↑ risk |
| `CONF_BLUR` | float | *0.15* | Minimum score for a `blur`-class detection to be blurred on its own (orphans included). Tracked detections below it are blurred when their track contains a detection ≥ `CONF_BLUR`. | Higher ↑ risk |
| `CONF_SIGN` | float | *0.6* | Minimum max-score of a sign track to produce an annotation (SGBlur value). | none |
| `DETECT_URL` | URL | empty | Remote Detect API; empty = in-process detection (SGBlur convention). | Video sent over the network: use a private network |

## Tracking

| Name | Type | Default | Description | Privacy impact |
|---|---|---|---|---|
| `TRACKER_CONFIG` | path | `configs/trackers/tracktrack-recall.yaml` | Ultralytics tracker YAML (+ our `track_buffer_s` extension). | Indirect (gap filling relies on ids) |
| `TRACK_WIDTH` | int | *1920* | Width of the frame used for tracking and camera-motion compensation. | — |

## Post-processing and blur

| Name | Type | Default | Description | Privacy impact |
|---|---|---|---|---|
| `BLUR_METHOD` | `pixelate_blur`\|`gaussian_strong`\|`solid` | `pixelate_blur` | Irreversible blur operation. | `gaussian_strong` weaker |
| `PIXELATE_CELLS` | int | *6* | Max mosaic cells on the long side of a shape. | Higher ↑ risk |
| `BLUR_BOX_MARGIN` | float | *0.15* | Enlargement of each box on each side (fraction of its size). | Lower ↑ risk |
| `BLUR_TEMPORAL_PADDING_FRAMES` | int | *15* | Frames blurred before the first and after the last observation of a track / around an orphan. | Lower ↑ risk |
| `BLUR_PADDING_GROWTH` | float | *0.05* | Per-frame growth of padded boxes. | Lower ↑ risk |
| `MAX_INTERPOLATION_GAP_S` | float | *2.0* | Longest gap filled by interpolation inside a track. | Lower ↑ risk |
| `SIGN_MIN_TRACK_LENGTH` | int | *5* | Minimum observations for a sign annotation. | none |
| `SEAM_MAX_GAP_S` | float | *0.5* | Max time between two tracks stitched across the 360° seam. | none |

## Encoding

| Name | Type | Default | Description | Privacy impact |
|---|---|---|---|---|
| `ENCODER` | str | `auto` | `auto`, `libx264`, `libx265`, `h264_videotoolbox`, `hevc_videotoolbox`, `h264_nvenc`, `hevc_nvenc`. | none |
| `ENCODE_BITRATE_FACTOR` | float | *1.0* | Output bit rate relative to the source. | none |

## Jobs, storage and limits

| Name | Type | Default | Description | Privacy impact |
|---|---|---|---|---|
| `DATA_DIR` | path | `./data` (`/data` in Docker) | Job database and per-job folders. Must be on a local disk, not shared with other apps. | Contains originals while jobs run |
| `TMP_DIR` | path | `DATA_DIR/tmp` | Scratch files. | Same |
| `RESULT_TTL_MINUTES` | int | *60* | How long results stay downloadable. | Higher ↑ retention |
| `KEEP_DIR` | path | `DATA_DIR/keep` | Encrypted `keep=1` regions. | — |
| `KEEP_TTL_HOURS` | int | *48* | Lifetime of `keep=1` regions. | Higher ↑ retention |
| `KEEP_SECRET_KEY` | secret str | empty | Server secret mixed with `blurring_id` (HKDF) to encrypt `keep=1` regions; empty disables `keep=1`. | Leak + store access = originals readable |
| `KEEP_MAX_CONFIDENCE` | float | *0.5* | Only regions of tracks whose max score is below this are kept (potential false positives, as in SGBlur). | Higher ↑ retention |
| `MAX_UPLOAD_BYTES` | int | *8 GiB* | Upload size limit. | none |
| `MAX_VIDEO_DURATION_S` | int | *1800* | Duration limit. | none |
| `SYNC_MAX_DURATION_S` | int | *30* | Max duration for `sync=1`. | none |
| `ACCEPTED_CONTAINERS` | list | `mp4,mov` | Accepted containers. | none |
| `WORKER_CONCURRENCY` | int | *1* | Jobs processed in parallel by one worker (one accelerator ⇒ 1). | none |
| `QUEUE_MAX` | int | *20* | Queued jobs before `503 queue_full`. | none |
| `JOB_TIMEOUT_S` | int | *21600* | Hard limit per job; the job process is killed and files deleted. | none |
| `CALLBACK_ALLOWED_HOSTS` | list | empty | Hosts allowed in `callback_url`; empty disables callbacks (SSRF protection). | none |
| `LOG_LEVEL` | str | `INFO` | Logs never contain file names, coordinates or images. | none |
