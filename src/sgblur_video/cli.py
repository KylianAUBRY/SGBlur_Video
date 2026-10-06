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
from pathlib import Path
from typing import Annotated, NoReturn

import typer

from sgblur_video import __version__
from sgblur_video.config import get_settings
from sgblur_video.models import RegistryError, load_registry

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
    _planned("step 4")


@app.command()
def blur(
    input_video: InputVideo,
    output_video: OutputVideo,
    model: ModelOption = None,
    tracker: TrackerOption = None,
    debug: Annotated[
        bool, typer.Option("--debug", help="Also write an annotated debug video next to the output.")
    ] = False,
) -> None:
    """Detect, track and blur a video in one command (passes 1 and 2)."""
    _planned("step 4")


@app.command()
def detect(
    input_video: InputVideo,
    out: Annotated[Path, typer.Option("--out", dir_okay=False, help="detections.jsonl to write.")],
    model: ModelOption = None,
    tracker: TrackerOption = None,
) -> None:
    """Run pass 1 only and write ``detections.jsonl``."""
    _planned("step 4")


@app.command()
def render(
    input_video: InputVideo,
    detections: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="detections.jsonl.")],
    output_video: OutputVideo,
    debug: Annotated[bool, typer.Option("--debug", help="Also write an annotated debug video.")] = False,
) -> None:
    """Run post-processing and pass 2 from an existing ``detections.jsonl``."""
    _planned("step 4")


@app.command()
def signs(
    input_video: InputVideo,
    out: Annotated[Path, typer.Option("--out", dir_okay=False, help="Annotations JSON to write.")],
    model: ModelOption = None,
) -> None:
    """Detect, track and deduplicate traffic signs; write Panoramax annotations."""
    _planned("step 5")


@app.command()
def benchmark(
    dataset: Annotated[Path, typer.Option("--dataset", exists=True, file_okay=False, help="Dataset folder.")],
) -> None:
    """Run the tracker / model / privacy benchmarks on a dataset."""
    _planned("step 8")


@app.command()
def worker() -> None:
    """Run the job worker (claims jobs from the job store)."""
    _planned("step 6")


@app.command()
def serve() -> None:
    """Run the Blur API with one worker (development and native single-host mode)."""
    _planned("step 6")


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
