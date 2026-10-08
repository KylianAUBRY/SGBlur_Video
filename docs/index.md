# SGBlur-Video

**Privacy blurring of street-level videos for [Panoramax](https://panoramax.fr).**

SGBlur-Video is the video counterpart of
[SGBlur](https://gitlab.com/panoramax/server/sgblur). It finds faces, licence
plates and traffic signs in dashcam, bike, pedestrian and 360° videos,
**irreversibly blurs faces and plates on every frame where they appear**, and
returns **one Panoramax annotation per physical traffic sign**.

!!! warning "Alpha"
    The pipeline, the HTTP API, 360° support and the benchmarks are
    implemented. The privacy defaults are checked on synthetic videos on every
    push; their validation on annotated real footage is in progress (see
    [Benchmarks](guides/benchmarks.md)). Do not rely on it for publication
    without reviewing the output.

```mermaid
flowchart LR
    v[/Video/] --> a["Pass 1: decode → YOLO26 (multi-scale) → tracking"]
    a --> j[/detections.jsonl/]
    j --> p["Post-processing: gap filling, padding, margins, sign dedup"]
    p --> r["Pass 2: decode → blur → encode → remux audio, GPS, 360° metadata"]
    r --> o[/Blurred video + annotations/]
```

## Where to start

| You want to… | Read |
|---|---|
| Install it and blur a first video | [Installation](getting-started/installation.md), [Quick start](getting-started/quickstart.md) |
| Use it from a terminal or a script | [Command line](usage/cli.md) |
| Run it as a service | [HTTP API](usage/api.md), [Installation § Docker](getting-started/installation.md#docker) |
| Understand what is blurred and why | [Frame-by-frame blurring](concepts/tracking.md) |
| Understand the design | [Architecture](design/architecture.md), [Pipeline](design/pipeline.md), [ADRs](adr/README.md) |
| Configure the service | [Configuration reference](reference/configuration.md) |
| Integrate with Panoramax | [HTTP API contract](design/api.md), [Annotations](concepts/annotations.md) |
| Measure privacy and speed | [Benchmarks](guides/benchmarks.md) |
| Know what happens to personal data | [Privacy and GDPR](concepts/privacy.md) |
| Contribute | [CONTRIBUTING.md](https://github.com/KylianAUBRY/SGBlur_Video/blob/main/CONTRIBUTING.md) |
