"""HTTP applications (FastAPI).

Two separate applications, as in SGBlur:

* ``blur_api``: uploads, jobs, results and metrics (port 8000);
* ``detect_api``: remote analysis, video in, ``detections.jsonl`` out (port 8001);
* ``upload``: streamed multipart upload to disk with a size limit;
* ``errors``: ``{"detail", "code"}`` error bodies;
* ``metrics``: Prometheus metrics computed from the job store.

The contract is ``docs/design/openapi.yaml``.
"""
