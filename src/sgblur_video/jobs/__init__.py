"""Asynchronous jobs.

* ``store``: SQLite job store (state machine, progress, retention queries);
* ``runner``: one job, from the uploaded file to the results (run in a child process);
* ``worker``: claims jobs, isolates each one in a process, enforces timeouts,
  cancellation and the retention janitor.

See ``docs/adr/0005-job-queue.md``.
"""
