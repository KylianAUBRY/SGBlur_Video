"""Command-line interface (``sgblur-video``).

The CLI runs the same pipeline as the HTTP service, without any server. Debug
outputs (annotated videos) are **only** available here, never through the API.

Commands marked "planned" print a message and exit with code 2 until the step
of the roadmap that implements them (see ``docs/research/step-1-analysis.md``).

Example:
    ```console
    $ sgblur-video blur input.mp4 output.mp4 --model yolo26s --debug
    ```
"""

import json
import logging
import sys
import tempfile
import time
from pathlib import Path
from typing import Annotated, NoReturn

import typer

from sgblur_video import __version__
from sgblur_video.config import Settings, get_settings
from sgblur_video.models import ClassPolicyError, RegistryError, WeightsError, ensure_weights, load_registry

app = typer.Typer(
    name="sgblur-video",
    help="Blur faces and licence plates in street-level videos and annotate traffic signs (Panoramax).",
    no_args_is_help=True,
    add_completion=False,
)
models_app = typer.Typer(help="Inspect and download detection models.", no_args_is_help=True)
annotate_app = typer.Typer(help="Build the manually annotated privacy dataset.", no_args_is_help=True)
app.add_typer(models_app, name="models")
app.add_typer(annotate_app, name="annotate")

InputVideo = Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Input video (MP4 or MOV).")]
OutputVideo = Annotated[Path, typer.Argument(dir_okay=False, help="Output video path.")]
ModelOption = Annotated[
    str | None, typer.Option("--model", help="Registry model name (default: automatic selection).")
]
TrackerOption = Annotated[
    Path | None, typer.Option("--tracker", exists=True, dir_okay=False, help="Tracker YAML configuration.")
]


def _planned(step: str) -> NoReturn:
    """Exit with a clear message for a command that is not implemented yet.

    Args:
        step: Roadmap step that will implement the command.

    Raises:
        typer.Exit: Always, with code 2.
    """
    typer.echo(f"Not implemented yet: planned for {step} of the roadmap.", err=True)
    raise typer.Exit(code=2)


FramesDirOption = Annotated[
    Path | None,
    typer.Option(
        "--frames-dir",
        file_okay=False,
        help="Write one blurred JPEG per sign best view here (Panoramax upload).",
    ),
]
MaxFramesOption = Annotated[
    int | None, typer.Option("--max-frames", min=1, help="Process only the first N frames (development).")
]


def _setup(tracker: Path | None = None) -> Settings:
    """Configure logging and return the settings, with CLI overrides applied."""
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr
    )
    if tracker is not None:
        settings = settings.model_copy(update={"tracker_config": tracker})
    return settings


class _Progress:
    """Minimal progress line on stderr (one update per percent at most)."""

    def __init__(self, label: str) -> None:
        self._label = label
        self._last = -1
        self._started = time.monotonic()

    def __call__(self, done: int, total: int | None) -> None:
        percent = int(done * 100 / total) if total else -1
        if percent == self._last and done % 100:
            return
        self._last = percent
        rate = done / max(1e-6, time.monotonic() - self._started)
        shown = f"{percent:3d} %" if percent >= 0 else f"{done} frames"
        typer.echo(f"\r{self._label}: {shown} ({rate:.1f} fps)", err=True, nl=False)
        if total and done >= total:
            typer.echo("", err=True)
            self._started = time.monotonic()


def _fail(exc: Exception) -> NoReturn:
    """Print an expected error (bad input, missing model…) without a traceback and exit with code 1."""
    typer.echo(f"error: {exc}", err=True)
    raise typer.Exit(code=1) from exc


def _expected_errors() -> tuple[type[Exception], ...]:
    """Errors reported to the user as messages rather than tracebacks."""
    from sgblur_video.core.detections_io import DetectionsFormatError
    from sgblur_video.core.probe import UnsupportedVideoError

    return (UnsupportedVideoError, DetectionsFormatError, RegistryError, WeightsError, ClassPolicyError)


@app.command()
def version() -> None:
    """Print the sgblur-video version."""
    typer.echo(f"sgblur-video {__version__}")


@app.command(name="config")
def show_config() -> None:
    """Print the effective configuration as JSON (secrets are masked)."""
    settings = get_settings()
    typer.echo(json.dumps(settings.model_dump(mode="json"), indent=2, sort_keys=True))


@models_app.command(name="list")
def models_list() -> None:
    """List the models of the registry (``MODELS_FILE``)."""
    settings = get_settings()
    try:
        registry = load_registry(settings.models_file)
    except RegistryError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    for entry in registry.models:
        typer.echo(f"{entry.name:12s} {entry.family:8s} {entry.version:8s} classes={','.join(entry.classes)}")


@models_app.command(name="download")
def models_download(
    name: Annotated[str | None, typer.Argument(help="Model name (default: all).")] = None,
) -> None:
    """Download model weights into ``MODELS_DIR`` and verify their SHA-256."""
    settings = _setup()
    try:
        registry = load_registry(settings.models_file)
        entries = [registry.get(name)] if name else list(registry.models)
        for entry in entries:
            path = ensure_weights(entry, settings.models_dir)
            typer.echo(f"{entry.name}: {path} (verified)")
    except (RegistryError, WeightsError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc


@app.command()
def blur(
    input_video: InputVideo,
    output_video: OutputVideo,
    model: ModelOption = None,
    tracker: TrackerOption = None,
    debug: Annotated[
        bool, typer.Option("--debug", help="Also write an annotated debug video next to the output.")
    ] = False,
    keep_detections: Annotated[
        Path | None,
        typer.Option("--keep-detections", dir_okay=False, help="Also save detections.jsonl here."),
    ] = None,
    frames_dir: FramesDirOption = None,
    max_frames: MaxFramesOption = None,
) -> None:
    """Detect, track and blur a video in one command (passes 1 and 2).

    Writes the blurred video, ``<output>.metadata.json`` (Panoramax annotations of
    traffic signs) and, with ``--debug``, ``<output>.debug.mp4``.
    """
    from sgblur_video.core.pipeline import run_blur

    settings = _setup(tracker)
    debug_output = output_video.with_name(f"{output_video.stem}.debug.mp4") if debug else None
    metadata_path = output_video.with_name(f"{output_video.stem}.metadata.json")
    with tempfile.TemporaryDirectory(prefix="sgblur-video-") as scratch:
        detections_path = keep_detections or Path(scratch) / "detections.jsonl"
        try:
            summary = run_blur(
                input_video,
                output_video,
                settings,
                detections_path=detections_path,
                metadata_path=metadata_path,
                frames_dir=frames_dir,
                model_name=model,
                debug_output=debug_output,
                max_frames=max_frames,
                progress=_Progress("progress"),
            )
        except _expected_errors() as exc:
            _fail(exc)
    typer.echo(json.dumps(summary, indent=2))


@app.command()
def detect(
    input_video: InputVideo,
    out: Annotated[Path, typer.Option("--out", dir_okay=False, help="detections.jsonl to write.")],
    model: ModelOption = None,
    tracker: TrackerOption = None,
    max_frames: MaxFramesOption = None,
) -> None:
    """Run pass 1 only and write ``detections.jsonl``."""
    from sgblur_video.core.pipeline import run_detect

    settings = _setup(tracker)
    try:
        _info, footer = run_detect(
            input_video,
            out,
            settings,
            model_name=model,
            max_frames=max_frames,
            progress=_Progress("analysis"),
        )
    except _expected_errors() as exc:
        _fail(exc)
    typer.echo(footer.model_dump_json(indent=2))


@app.command()
def render(
    input_video: InputVideo,
    detections: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="detections.jsonl.")],
    output_video: OutputVideo,
    debug: Annotated[bool, typer.Option("--debug", help="Also write an annotated debug video.")] = False,
    frames_dir: FramesDirOption = None,
    allow_partial: Annotated[
        bool, typer.Option("--allow-partial", help="Render even if detections.jsonl is incomplete.")
    ] = False,
) -> None:
    """Run post-processing and pass 2 from an existing ``detections.jsonl``.

    Writes the blurred video and ``<output>.metadata.json`` (sign annotations).
    """
    from sgblur_video.core.pipeline import run_render, write_metadata

    settings = _setup()
    debug_output = output_video.with_name(f"{output_video.stem}.debug.mp4") if debug else None
    try:
        result = run_render(
            input_video,
            detections,
            output_video,
            settings,
            debug_output=debug_output,
            frames_dir=frames_dir,
            allow_partial=allow_partial,
            progress=_Progress("rendering"),
        )
    except _expected_errors() as exc:
        _fail(exc)
    write_metadata(result.metadata, output_video.with_name(f"{output_video.stem}.metadata.json"))
    summary = {
        "blur": result.plan.stats,
        "frames": result.render.frames,
        "signs": len(result.metadata.annotations),
        "best_frames": len(result.frames),
    }
    typer.echo(json.dumps(summary, indent=2))


@app.command()
def signs(
    input_video: InputVideo,
    out: Annotated[Path, typer.Option("--out", dir_okay=False, help="Annotations JSON to write.")],
    model: ModelOption = None,
    tracker: TrackerOption = None,
    frames_dir: FramesDirOption = None,
    max_frames: MaxFramesOption = None,
) -> None:
    """Detect, track and deduplicate traffic signs; write one Panoramax annotation per sign.

    No video is written. With ``--frames-dir``, the best view of each sign is
    saved as a JPEG in which faces and plates are blurred.
    """
    from sgblur_video.core.pipeline import run_signs

    settings = _setup(tracker)
    with tempfile.TemporaryDirectory(prefix="sgblur-video-") as scratch:
        try:
            metadata = run_signs(
                input_video,
                out,
                settings,
                detections_path=Path(scratch) / "detections.jsonl",
                frames_dir=frames_dir,
                model_name=model,
                max_frames=max_frames,
                progress=_Progress("analysis"),
            )
        except _expected_errors() as exc:
            _fail(exc)
    typer.echo(f"{len(metadata.annotations)} signs written to {out}")


@app.command()
def benchmark(
    dataset: Annotated[Path, typer.Option("--dataset", exists=True, file_okay=False, help="Dataset folder.")],
) -> None:
    """Run the tracker / model / privacy benchmarks on a dataset."""
    _planned("step 8")


HostOption = Annotated[str, typer.Option(help="Interface to listen on (0.0.0.0 in containers).")]


def _worker_process(settings: Settings) -> None:
    """Entry point of a worker process started by ``serve``."""
    from sgblur_video.jobs.worker import Worker

    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    Worker(settings).run_forever()


@app.command()
def worker() -> None:
    """Run the job worker: claims queued jobs from the job store and processes them."""
    from sgblur_video.jobs.worker import Worker

    Worker(_setup()).run_forever()


@app.command()
def serve(
    host: HostOption = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port of the Blur API.")] = 8000,
    workers: Annotated[int, typer.Option(min=0, help="Worker processes to start alongside the API.")] = 1,
) -> None:
    """Run the Blur API with worker processes (native single-host mode)."""
    import multiprocessing

    import uvicorn

    from sgblur_video.api.blur_api import create_app

    settings = _setup()
    context = multiprocessing.get_context("spawn")
    # Not daemonic: a worker starts one child process per job.
    processes = [
        context.Process(target=_worker_process, args=(settings,), name=f"worker-{i}") for i in range(workers)
    ]
    for process in processes:
        process.start()
    try:
        uvicorn.run(create_app(settings), host=host, port=port, log_level=settings.log_level.lower())
    finally:
        for process in processes:
            process.terminate()
        for process in processes:
            process.join(timeout=10)


@app.command(name="serve-detect")
def serve_detect(
    host: HostOption = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port of the Detect API.")] = 8001,
) -> None:
    """Run the Detect API (remote analysis for a Blur API configured with ``DETECT_URL``)."""
    import uvicorn

    from sgblur_video.api.detect_api import create_app

    settings = _setup()
    uvicorn.run(create_app(settings), host=host, port=port, log_level=settings.log_level.lower())


@annotate_app.command(name="export")
def annotate_export(
    input_video: InputVideo,
    start: Annotated[float, typer.Option(help="Clip start, in seconds.")] = 0.0,
    duration: Annotated[float, typer.Option(help="Clip duration, in seconds.")] = 15.0,
) -> None:
    """Cut a clip, build an annotation proxy and a pre-annotation for CVAT."""
    _planned("step 8")


@annotate_app.command(name="import")
def annotate_import(
    cvat_export: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="CVAT for video 1.1 XML.")],
) -> None:
    """Convert a CVAT export into the privacy dataset format."""
    _planned("step 8")
