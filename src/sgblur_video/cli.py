"""Command-line interface (`sgblur-video`).

The CLI runs the same pipeline as the HTTP service, without any server. Debug
outputs (annotated videos) are **only** available here, never through the API.

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
from typing import Annotated, Any, NoReturn

import typer

from sgblur_video import __version__
from sgblur_video.config import Settings, get_settings
from sgblur_video.models import (
    ClassPolicyError,
    RegistryError,
    WeightsError,
    ensure_weights,
    load_registry,
    resolve_model,
)

app = typer.Typer(
    name="sgblur-video",
    help="Blur faces and licence plates in street-level videos and annotate traffic signs (Panoramax).",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="markdown",
)
models_app = typer.Typer(help="Inspect and download detection models.", no_args_is_help=True)
annotate_app = typer.Typer(help="Build the manually annotated privacy dataset.", no_args_is_help=True)
benchmark_app = typer.Typer(help="Measure privacy and speed.", no_args_is_help=True)
app.add_typer(models_app, name="models")
app.add_typer(annotate_app, name="annotate")
app.add_typer(benchmark_app, name="benchmark")

InputVideo = Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Input video (MP4 or MOV).")]
OutputVideo = Annotated[Path, typer.Argument(dir_okay=False, help="Output video path.")]
ModelOption = Annotated[
    str | None,
    typer.Option(
        "--model", help="Registry model name, or path of a .pt checkpoint (default: MODEL_PATH, MODEL_NAME)."
    ),
]
TrackerOption = Annotated[
    Path | None,
    typer.Option("--tracker", exists=True, dir_okay=False, help="Sign tracker YAML configuration."),
]


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


def _expected_errors(*extra: type[Exception]) -> tuple[type[Exception], ...]:
    """Errors reported to the user as messages rather than tracebacks (plus `extra`)."""
    from sgblur_video.bench.clips import ClipError
    from sgblur_video.bench.dataset import DatasetError
    from sgblur_video.core.detections_io import DetectionsFormatError
    from sgblur_video.core.probe import UnsupportedVideoError

    return (
        UnsupportedVideoError,
        DetectionsFormatError,
        RegistryError,
        WeightsError,
        ClassPolicyError,
        DatasetError,
        ClipError,
        *extra,
    )


@app.command()
def version() -> None:
    """Print the sgblur-video version."""
    typer.echo(f"sgblur-video {__version__}")


@app.command(name="config")
def show_config() -> None:
    """Print the effective configuration as JSON (secrets are masked, the home folder is shown as `~`).

    The output is meant to be pasted in public issues: it must not reveal secrets or the user name.
    """
    settings = get_settings()
    text = json.dumps(settings.model_dump(mode="json"), indent=2, sort_keys=True)
    typer.echo(text.replace(json.dumps(str(Path.home()))[1:-1], "~"))


@models_app.command(name="list")
def models_list() -> None:
    """List the models of the registry (`MODELS_FILE`); `*` marks the one that would be used."""
    settings = get_settings()
    try:
        registry = load_registry(settings.models_file)
    except RegistryError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    try:
        selected = resolve_model(settings).name
    except RegistryError, WeightsError:
        selected = None
    for entry in registry.models:
        mark = "*" if entry.name == selected else " "
        typer.echo(
            f"{mark} {entry.name:12s} {entry.family:8s} {entry.version:8s} classes={','.join(entry.classes)}"
        )
    if settings.model_path is not None:
        typer.echo(
            f"* MODEL_PATH: {settings.model_path.name} (local checkpoint, used instead of the registry)"
        )


@models_app.command(name="inspect")
def models_inspect(
    checkpoint: Annotated[Path, typer.Argument(dir_okay=False, help="A .pt checkpoint.")],
) -> None:
    """Show what a checkpoint contains, whether it can be used, and a registry entry to start from.

    Try it without registering it: `MODEL_PATH=<file> sgblur-video blur …` or `--model <file>`.
    """
    from sgblur_video.models import check_class_policy, local_entry

    settings = _setup()
    try:
        entry = local_entry(checkpoint)
    except (RegistryError, WeightsError) as exc:
        _fail(exc)
    typer.echo(f"file:        {checkpoint.name} ({entry.size_bytes} bytes)")
    typer.echo(f"sha256:      {entry.sha256}")
    typer.echo(f"classes:     {', '.join(entry.classes)}")
    typer.echo(f"train imgsz: {entry.train_imgsz}")
    typer.echo(f"tag:         {settings.api_name}-{entry.tag} (Panoramax detection_model when used as is)")
    try:
        warnings = check_class_policy(entry.classes, settings.class_policy)
    except ClassPolicyError as exc:
        typer.echo(f"usable:      NO, {exc}")
        raise typer.Exit(code=1) from exc
    typer.echo("usable:      yes" + "".join(f"\n  warning: {w}" for w in warnings))
    typer.echo(
        "\nRegistry entry (models/registry.yaml), once the file is published at a pinned URL:\n"
        f"  - name: {entry.name}\n"
        "    family: <family>\n"
        '    version: "<version>"\n'
        f"    file: {checkpoint.name}\n"
        "    url: <commit-pinned URL>\n"
        f"    sha256: {entry.sha256}\n"
        f"    size_bytes: {entry.size_bytes}\n"
        f"    classes: [{', '.join(entry.classes)}]\n"
        f"    train_imgsz: {entry.train_imgsz}\n"
        "    min_memory_gib: 2\n"
        '    licence: "<licence>"\n'
        "    source: <project URL>"
    )


@models_app.command(name="download")
def models_download(
    name: Annotated[str | None, typer.Argument(help="Model name (default: all).")] = None,
) -> None:
    """Download model weights into `MODELS_DIR` and verify their SHA-256."""
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

    Writes the blurred video, `<output>.metadata.json` (Panoramax annotations of
    traffic signs) and, with `--debug`, `<output>.debug.mp4`.
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
    """Run pass 1 only and write `detections.jsonl`."""
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
    """Run post-processing and pass 2 from an existing `detections.jsonl`.

    Writes the blurred video and `<output>.metadata.json` (sign annotations).
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

    No video is written. With `--frames-dir`, the best view of each sign is
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


HostOption = Annotated[str, typer.Option(help="Interface to listen on (0.0.0.0 in containers).")]
_LOCAL_HOSTS = frozenset({"127.0.0.1", "0.0.0.0"})  # noqa: S104 - only compared, never bound


def _worker_process(settings: Settings) -> None:
    """Entry point of a worker process started by `serve`."""
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
    if settings.web_ui:
        # 0.0.0.0 (containers) is reachable as localhost from the same machine.
        shown = "localhost" if host in _LOCAL_HOSTS else host
        typer.echo(f"Web interface: http://{shown}:{port}/ui")
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
    """Run the Detect API (remote analysis for a Blur API configured with `DETECT_URL`)."""
    import uvicorn

    from sgblur_video.api.detect_api import create_app

    settings = _setup()
    uvicorn.run(create_app(settings), host=host, port=port, log_level=settings.log_level.lower())


DatasetOption = Annotated[
    Path, typer.Option("--dataset", file_okay=False, help="Privacy dataset folder (outside the repository).")
]
ClipIdOption = Annotated[
    str, typer.Option("--id", help="Clip identifier (lower-case letters, digits, - and _).")
]
ThresholdsOption = Annotated[
    Path,
    typer.Option("--thresholds", exists=True, dir_okay=False, help="Privacy gate thresholds."),
]
DEFAULT_THRESHOLDS = Path("benchmarks/privacy-thresholds.yaml")


PreannotationConf = Annotated[
    float, typer.Option("--conf", help="Minimum best score of a pre-annotated track.")
]
PreannotationMinFrames = Annotated[
    int,
    typer.Option("--min-frames", min=1, help="Minimum frames on which a pre-annotated track was detected."),
]


@annotate_app.command(name="export")
def annotate_export(
    input_video: InputVideo,
    dataset: DatasetOption,
    clip_id: ClipIdOption,
    start: Annotated[float, typer.Option(min=0, help="Clip start, in seconds.")] = 0.0,
    duration: Annotated[float, typer.Option(min=0.1, help="Clip duration, in seconds.")] = 15.0,
    proxy_width: Annotated[int, typer.Option(min=320, help="Maximum width of the CVAT proxy.")] = 3840,
    conf: PreannotationConf = 0.25,
    min_frames: PreannotationMinFrames = 5,
    model: ModelOption = None,
) -> None:
    """Cut a clip, build an annotation proxy and a pre-annotation for CVAT.

    Writes `clips/<id>/clip.mp4`, `proxy.mp4` and `preannotation.xml` in the
    dataset folder and adds the clip to `manifest.yaml`.
    """
    from sgblur_video.bench.annotate import export_clip, preannotate_clip
    from sgblur_video.bench.dataset import Dataset

    settings = _setup()
    data = Dataset(dataset)
    try:
        entry = export_clip(
            input_video,
            data,
            clip_id,
            settings,
            start_s=start,
            duration_s=duration,
            proxy_max_width=proxy_width,
            progress=_Progress(f"{clip_id} cut"),
        )
        tracks = preannotate_clip(
            data,
            clip_id,
            settings,
            conf=conf,
            min_frames=min_frames,
            model_name=model,
            progress=_Progress(f"{clip_id} analysis"),
        )
    except _expected_errors() as exc:
        _fail(exc)
    typer.echo(
        f"{entry.id}: {entry.frames} frames, proxy {entry.proxy_width}x{entry.proxy_height}, "
        f"{tracks} pre-annotated tracks. Next: create a CVAT task from proxy.mp4, upload "
        "preannotation.xml (CVAT 1.1), correct it, export 'CVAT for video 1.1', then run "
        "'sgblur-video annotate import'."
    )


@annotate_app.command(name="preannotate")
def annotate_preannotate(
    dataset: DatasetOption,
    clip_id: ClipIdOption,
    conf: PreannotationConf = 0.25,
    min_frames: PreannotationMinFrames = 5,
    model: ModelOption = None,
) -> None:
    """Rebuild the pre-annotation of an exported clip (other thresholds; detections are cached)."""
    from sgblur_video.bench.annotate import preannotate_clip
    from sgblur_video.bench.dataset import Dataset

    settings = _setup()
    try:
        tracks = preannotate_clip(
            Dataset(dataset),
            clip_id,
            settings,
            conf=conf,
            min_frames=min_frames,
            model_name=model,
            progress=_Progress(f"{clip_id} analysis"),
        )
    except _expected_errors() as exc:
        _fail(exc)
    typer.echo(f"{clip_id}: {tracks} pre-annotated tracks")


@annotate_app.command(name="import")
def annotate_import(
    cvat_export: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="CVAT for video 1.1 XML.")],
    dataset: DatasetOption,
    clip_id: ClipIdOption,
    annotator: Annotated[str, typer.Option(help="Who annotated the clip (recorded in the manifest).")] = "",
) -> None:
    """Convert a CVAT export into the privacy dataset format (`ground_truth.json`)."""
    from sgblur_video.bench.annotate import import_annotation
    from sgblur_video.bench.dataset import Dataset

    _setup()
    try:
        truth = import_annotation(Dataset(dataset), clip_id, cvat_export, annotator=annotator)
    except _expected_errors() as exc:
        _fail(exc)
    boxes = sum(len(t.boxes) for t in truth.tracks)
    readable = sum(1 for t in truth.tracks for b in t.boxes if b.readable)
    typer.echo(f"{clip_id}: {len(truth.tracks)} tracks, {boxes} object-frames ({readable} readable)")


def _emit(report: dict[str, Any], report_dir: Path | None) -> None:
    from sgblur_video.bench.report import to_markdown, write_report

    typer.echo(to_markdown(report))
    if report_dir is not None:
        json_path, md_path = write_report(report, report_dir)
        typer.echo(f"report written to {json_path} and {md_path}", err=True)


ReportDirOption = Annotated[
    Path | None,
    typer.Option("--report-dir", file_okay=False, help="Also write JSON and Markdown reports here."),
]


@benchmark_app.command(name="privacy")
def benchmark_privacy(
    dataset: DatasetOption,
    sweep: Annotated[
        list[str] | None,
        typer.Option("--sweep", help="Settings to compare, e.g. CONF_DETECT=0.2,0.3 (repeatable)."),
    ] = None,
    model: ModelOption = None,
    tracker: TrackerOption = None,
    thresholds: ThresholdsOption = DEFAULT_THRESHOLDS,
    report_dir: ReportDirOption = None,
) -> None:
    """Leakage of faces and plates on the annotated clips, gated by the thresholds.

    Exits with code 1 when the default settings (first run) fail the gate.
    """
    from sgblur_video.bench.dataset import Dataset
    from sgblur_video.bench.runs import load_thresholds, parse_sweeps, run_privacy

    settings = _setup(tracker)
    try:
        report = run_privacy(
            Dataset(dataset),
            settings,
            sweeps=parse_sweeps(sweep or []),
            thresholds=load_thresholds(thresholds),
            model_name=model,
            progress=_Progress,
        )
    except _expected_errors(ValueError) as exc:
        _fail(exc)
    _emit(report, report_dir or dataset / "reports")
    if not report["runs"][0]["gate"]["passed"]:
        raise typer.Exit(code=1)


@benchmark_app.command(name="speed")
def benchmark_speed(
    input_video: InputVideo,
    model: Annotated[list[str] | None, typer.Option("--model", help="Registry model (repeatable).")] = None,
    device: Annotated[list[str] | None, typer.Option("--device", help="cpu, mps, cuda (repeatable).")] = None,
    profile: Annotated[
        list[str] | None, typer.Option("--profile", help="fast, standard, thorough (repeatable).")
    ] = None,
    frames: Annotated[int, typer.Option(min=1, help="Timed frames per combination.")] = 20,
    report_dir: ReportDirOption = None,
) -> None:
    """Detection speed per model, device and detection profile."""
    from sgblur_video.bench.runs import run_speed

    settings = _setup()
    try:
        report = run_speed(
            input_video,
            settings,
            models=model or [settings.model_name or "yolo26s"],
            devices=device or [settings.device],
            profiles=profile or [settings.detect_profile.value],
            frames=frames,
            progress=_Progress,
        )
    except _expected_errors(ValueError) as exc:
        _fail(exc)
    _emit(report, report_dir)
