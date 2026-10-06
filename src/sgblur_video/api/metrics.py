"""Prometheus metrics computed from the job store (no multi-process registry needed)."""

from sgblur_video.jobs.store import JobStore

_DURATION_BUCKETS = (60, 300, 900, 1800, 3600, 7200, 21600)


def render_metrics(store: JobStore) -> str:
    """Prometheus text exposition of the service state.

    Args:
        store: Job store.

    Returns:
        Metrics in the Prometheus text format (version 0.0.4).
    """
    counts = store.counts()
    finished = store.finished_jobs()
    durations = [float(j.stats["elapsed_s"]) for j in finished if "elapsed_s" in j.stats]
    frames = sum(int(j.stats.get("frames", 0)) for j in finished)
    last_fps = next(
        (
            float(j.stats["fps"])
            for j in sorted(finished, key=lambda j: j.finished_at or j.created_at, reverse=True)
            if "fps" in j.stats
        ),
        0.0,
    )
    lines = [
        "# HELP sgblur_video_jobs Jobs currently known by the store, per status.",
        "# TYPE sgblur_video_jobs gauge",
        *(
            f'sgblur_video_jobs{{status="{status}"}} {counts.get(status, 0)}'
            for status in ("queued", "running", "succeeded", "failed", "cancelled", "expired")
        ),
        "# HELP sgblur_video_jobs_in_progress Jobs being processed.",
        "# TYPE sgblur_video_jobs_in_progress gauge",
        f"sgblur_video_jobs_in_progress {counts.get('running', 0)}",
        "# HELP sgblur_video_queue_length Jobs waiting for a worker.",
        "# TYPE sgblur_video_queue_length gauge",
        f"sgblur_video_queue_length {counts.get('queued', 0)}",
        "# HELP sgblur_video_job_duration_seconds Processing time of finished jobs.",
        "# TYPE sgblur_video_job_duration_seconds histogram",
        *(
            f'sgblur_video_job_duration_seconds_bucket{{le="{bound}"}} {sum(d <= bound for d in durations)}'
            for bound in _DURATION_BUCKETS
        ),
        f'sgblur_video_job_duration_seconds_bucket{{le="+Inf"}} {len(durations)}',
        f"sgblur_video_job_duration_seconds_sum {sum(durations):.1f}",
        f"sgblur_video_job_duration_seconds_count {len(durations)}",
        "# HELP sgblur_video_frames_processed_total Video frames processed by finished jobs.",
        "# TYPE sgblur_video_frames_processed_total counter",
        f"sgblur_video_frames_processed_total {frames}",
        "# HELP sgblur_video_processing_fps Overall frames per second of the last finished job.",
        "# TYPE sgblur_video_processing_fps gauge",
        f"sgblur_video_processing_fps {last_fps}",
    ]
    return "\n".join(lines) + "\n"
