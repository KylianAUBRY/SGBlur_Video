# Models

This folder holds the **model registry** (`registry.yaml`) only. Weights are
never committed: they are downloaded from the pinned URL of each entry into
`MODELS_DIR` (default `~/.cache/sgblur-video/models`, `/models` in Docker) and
verified against their SHA-256.

```bash
uv run sgblur-video models list
uv run sgblur-video models download yolo26s   # or without a name: every model of the registry
```

How to add a model or move to a new YOLO version: see
[docs/guides/models.md](../docs/guides/models.md).
