"""Video pipeline: probe, two-pass processing and remux (steps 4 and 7).

Planned modules (see ``docs/design/pipeline.md``):

* ``probe``: read container/stream properties, projection, rotation, telemetry;
* ``decode``: frame iterator with exact timestamps;
* ``detect``: detection plan (global passes, tiles, 360° padding) and cross-pass merge;
* ``track``: Ultralytics tracker adapter (one tracker per class group);
* ``postprocess``: gap filling, padding, margins, blur plan, sign tracks;
* ``render`` / ``encode`` / ``remux``: blur, encode and mux in one loop;
* ``mp4boxes``: allow-listed MP4 box transplant (spherical metadata, udta);
* ``pipeline``: orchestration used by the CLI, the worker and the Detect API.
"""
