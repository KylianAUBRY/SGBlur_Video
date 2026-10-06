"""Video pipeline (see ``docs/design/pipeline.md``).

* ``probe``: container/stream properties, projection, rotation, copyable streams;
* ``decode``: frame iterator with exact timestamps;
* ``device``: CUDA / MPS / CPU selection;
* ``detect``: detection plan (global passes, tiles), YOLO adapter, cross-pass merge;
* ``track``: Ultralytics tracker adapter (one tracker per class group);
* ``detections_io``: ``detections.jsonl`` reader and writer;
* ``analyze``: pass 1;
* ``postprocess``: linking, gap filling, smoothing, padding, margins → blur plan;
* ``encode`` / ``render``: pass 2 (blur, encode and mux in one loop);
* ``debug``: annotated debug video (CLI only);
* ``frames``: best-frame JPEGs of signs (blurred, EXIF);
* ``pipeline``: orchestration used by the CLI (and later the worker and the Detect API).

Planned: ``mp4boxes`` (spherical metadata and udta transplant, step 7).
"""
