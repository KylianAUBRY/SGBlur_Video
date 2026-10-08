"""Pass 1 of a job in a process of its own.

The detector (PyTorch, tiles up to 4096 px) leaves gigabytes allocated in the
process even once it is released, and rendering 8K video with a software
encoder needs about as much again: run in one process, both phases exceeded
the 7.7 GB of a default Docker Desktop VM. Running the analysis in a child
process returns all of its memory to the system before rendering starts.

The child is started with ``python -m sgblur_video.jobs.analysis <args file>``
(not ``multiprocessing``: job processes are daemonic and cannot have
children). It reports progress to the job store itself, and stops if the job
process that started it disappears (job cancelled, timed out or killed).
"""

import logging
import os
import pickle
import signal
import subprocess
import sys
from pathlib import Path

from sgblur_video.config import Settings
from sgblur_video.core.probe import VideoInfo

logger = logging.getLogger(__name__)

#: Exit code of a child that stopped because its parent (the job process) disappeared.
ORPHANED = 3


class AnalysisKilledError(RuntimeError):
    """The analysis process was killed by the system (most likely for lack of memory)."""


def run_analysis(job_id: str, info: VideoInfo, settings: Settings, output: Path) -> None:
    """Run pass 1 of a job in a child process and wait for it.

    Args:
        job_id: Job (the child reports progress in the job store).
        info: Probed video to analyse.
        settings: Settings.
        output: ``detections.jsonl`` to write.

    Raises:
        AnalysisKilledError: If the child was killed by a signal (out of memory…).
        RuntimeError: If the child failed.
    """
    arguments = output.with_name("analysis-args.pickle")
    arguments.write_bytes(pickle.dumps((job_id, info, settings, output)))
    try:
        completed = subprocess.run(  # noqa: S603 - fixed command, arguments are a file path
            [sys.executable, "-m", "sgblur_video.jobs.analysis", str(arguments)], check=False
        )
    finally:
        arguments.unlink(missing_ok=True)
    if completed.returncode == -signal.SIGKILL:
        msg = "the analysis process was killed by the system"
        raise AnalysisKilledError(msg)
    if completed.returncode != 0:
        msg = f"the analysis process failed (exit code {completed.returncode})"
        raise RuntimeError(msg)


def _main(arguments: Path) -> None:
    from sgblur_video.core.analyze import analyze
    from sgblur_video.core.pipeline import load_model
    from sgblur_video.jobs.runner import Progress
    from sgblur_video.jobs.store import JobStore

    job_id, info, settings, output = pickle.loads(arguments.read_bytes())  # noqa: S301 - written by the job process, in its folder
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    progress = Progress(JobStore(settings.data_dir), job_id)
    parent = os.getppid()

    def report(done: int, total: int | None) -> None:
        if os.getppid() != parent:  # the job process is gone: nobody will use the result
            os._exit(ORPHANED)
        progress(done, total)

    model = load_model(settings)
    analyze(
        info,
        model.detector,
        settings,
        output,
        model=model.header(settings),
        device=model.device,
        progress=report,
    )


if __name__ == "__main__":
    _main(Path(sys.argv[1]))
