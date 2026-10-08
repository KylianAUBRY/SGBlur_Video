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
* ``debug``: annotated debug video (CLI ``--debug``, API ``debug=1``);
* ``frames``: best-frame JPEGs of signs (blurred, EXIF);
* ``mp4boxes``: allow-listed transplant of spherical metadata, rotation and camera boxes;
* ``geometry``: box arithmetic shared by every step;
* ``pipeline``: orchestration used by the CLI, the job worker and the Detect API.
"""
