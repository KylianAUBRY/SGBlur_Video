"""Privacy primitives (steps 4 and 6).

* ``blur``: irreversible blur methods and shapes applied on YUV planes;
* ``keep``: AES-256-GCM storage of ``keep=1`` regions with a time-to-live;
* ``cleanup``: deletion of every job file, on success, failure and cancel.
"""
