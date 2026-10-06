"""HTTP applications (FastAPI), implemented in step 6.

Two separate applications, as in SGBlur:

* ``blur_api``: uploads, jobs, results and metrics (port 8000);
* ``detect_api``: remote analysis, video in, ``detections.jsonl`` out (port 8001);
* ``schemas``: Pydantic request/response models shared by both, matching
  ``docs/design/openapi.yaml``.
"""
