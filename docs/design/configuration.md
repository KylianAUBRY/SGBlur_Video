# Configuration reference (design)

!!! note "Superseded"
    This was the step-2 design table. The authoritative, always up-to-date
    reference is generated from the code: [Configuration reference](../reference/configuration.md).

> Historical: values in *italics* were the step-2 proposals. Settings are read with `pydantic-settings` from
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
| `CONF_DETECT` | float | *0.30* | Minimum detector score kept (SGBlur's `MIN_CONF`); every face and plate detection kept is blurred on its frame. | Higher ↑ risk |
| `DETECT_URL` | URL | empty | Remote Detect API; empty = in-process detection (SGBlur convention). | Video sent over the network: use a private network |

## Signs

Only signs are tracked (one annotation per physical sign); tracking never
changes what is blurred ([ADR-0012](../adr/0012-independent-frames.md)).

| Name | Type | Default | Description | Privacy impact |
|---|---|---|---|---|
| `TRACKER_CONFIG` | path | `configs/trackers/flow.yaml` | Sign tracker: optical flow, or an Ultralytics tracker YAML (+ our `track_buffer_s` extension). | none |
| `TRACK_WIDTH` | int | *1920* | Width of the frame used for sign tracking and camera-motion compensation. | none |
| `CONF_SIGN` | float | *0.6* | Minimum max-score of a sign track to produce an annotation (SGBlur value). | none |
| `LINK_MAX_GAP_S` | float | *1.0* | Longest interruption across which detections of one sign are linked. | none |
| `LINK_MAX_DISTANCE` | float | *1.0* | Box sizes between where a sign was heading and where it reappears. | none |
| `SIGN_MIN_TRACK_LENGTH` | int | *5* | Minimum observations for a sign annotation. | none |

## Blur

| Name | Type | Default | Description | Privacy impact |
|---|---|---|---|---|
| `BLUR_METHOD` | `pixelate_blur`\|`gaussian_strong`\|`solid` | `pixelate_blur` | Irreversible blur operation. | `gaussian_strong` weaker |
| `PIXELATE_CELLS` | int | *6* | Max mosaic cells on the long side of a blurred box. | Higher ↑ risk |

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
| `KEEP_MAX_CONFIDENCE` | float | *0.5* | Only regions detected with a score below this are kept (potential false positives, as in SGBlur). | Higher ↑ retention |
| `MAX_UPLOAD_BYTES` | int | *8 GiB* | Upload size limit. | none |
| `MAX_VIDEO_DURATION_S` | int | *1800* | Duration limit. | none |
| `SYNC_MAX_DURATION_S` | int | *30* | Max duration for `sync=1`. | none |
| `ACCEPTED_CONTAINERS` | list | `mp4,mov` | Accepted containers. | none |
| `QUEUE_MAX` | int | *20* | Queued jobs before `503 queue_full`. | none |
| `JOB_TIMEOUT_S` | int | *21600* | Hard limit per job; the job process is killed and files deleted. | none |
| `CALLBACK_ALLOWED_HOSTS` | list | empty | Hosts allowed in `callback_url`; empty disables callbacks (SSRF protection). | none |
| `LOG_LEVEL` | str | `INFO` | Logs never contain file names, coordinates or images. | none |
