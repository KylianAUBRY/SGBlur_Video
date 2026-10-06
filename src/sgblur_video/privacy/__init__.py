"""Privacy primitives.

* ``blur``: irreversible blur methods and shapes applied on YUV planes;
* ``keep``: AES-256-GCM storage of ``keep=1`` regions with a time-to-live.

Job files are deleted by ``jobs.runner`` (input, as soon as processing ends)
and ``jobs.worker`` (results after ``RESULT_TTL_MINUTES``, failed and cancelled jobs).
"""
