"""Service configuration.

Every setting is read from an environment variable of the same name (case
insensitive, **no prefix**, as in SGBlur: ``API_NAME``, ``DETECT_URL``…) or from
a ``.env`` file in the working directory. The reference table in
``docs/reference/configuration.md`` is generated from this module by
``scripts/gen_config_reference.py``; a unit test fails if it is out of date.

Each field carries two documentation hints in ``json_schema_extra``:

* ``group``: section of the reference table;
* ``privacy``: what happens to privacy when the value changes (empty when
  irrelevant). Settings marked with a risk must not be loosened without
  running the privacy benchmark (see ``docs/design/testing-strategy.md``).

Example:
    >>> from sgblur_video.config import Settings
    >>> Settings(conf_blur=0.2).conf_blur
    0.2
"""

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import Field, HttpUrl, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

GiB = 1024**3


class ClassAction(StrEnum):
    """What the pipeline does with a detected class."""

    BLUR = "blur"
    """Irreversibly blurred on every frame where the object is present; never annotated."""
    ANNOTATE = "annotate"
    """Tracked, deduplicated and returned as one Panoramax annotation; never blurred."""


class BlurMethod(StrEnum):
    """Irreversible blur operations (see ``docs/adr/0004-irreversible-blur.md``)."""

    PIXELATE_BLUR = "pixelate_blur"
    GAUSSIAN_STRONG = "gaussian_strong"
    SOLID = "solid"


class DetectProfile(StrEnum):
    """How many detection passes run on each frame."""

    FAST = "fast"
    """Global passes only (no tiles)."""
    STANDARD = "standard"
    """Global passes, plus tiles on very large frames (SGBlur-equivalent)."""
    THOROUGH = "thorough"
    """Like ``standard`` but tiles cover the full frame height (360° nadir/zenith)."""


class Projection(StrEnum):
    """Video projection handling."""

    AUTO = "auto"
    FLAT = "flat"
    EQUIRECTANGULAR = "equirectangular"


Encoder = Literal[
    "auto",
    "libx264",
    "libx265",
    "h264_videotoolbox",
    "hevc_videotoolbox",
    "h264_nvenc",
    "hevc_nvenc",
]

DEFAULT_CLASS_POLICY: dict[str, ClassAction] = {
    "face": ClassAction.BLUR,
    "plate": ClassAction.BLUR,
    "sign": ClassAction.ANNOTATE,
    "direction": ClassAction.ANNOTATE,
}


def _doc(group: str, privacy: str = "", doc_default: str | None = None) -> dict[str, Any]:
    """Build the ``json_schema_extra`` documentation hints of a field.

    Args:
        group: Section of the generated reference table.
        privacy: Privacy impact of changing the value; empty if irrelevant.
        doc_default: Default value as displayed in the documentation, when the
            real default is machine-dependent or unreadable.

    Returns:
        The mapping stored in ``json_schema_extra``.
    """
    hints: dict[str, Any] = {"group": group, "privacy": privacy}
    if doc_default is not None:
        hints["doc_default"] = doc_default
    return hints


class Settings(BaseSettings):
    """All runtime settings of the service, the worker and the CLI.

    Instances are immutable. Use :func:`get_settings` for the process-wide
    instance read from the environment; tests build their own instances.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        frozen=True,
    )

    # --- Service identity -------------------------------------------------------------------
    api_name: str = Field(
        "SGBlur-Video",
        min_length=1,
        description=(
            "`service_name` in metadata and prefix of `detection_model` tag values. Never change it "
            "after deployment: Panoramax removes previous detection tags by this prefix."
        ),
        json_schema_extra=_doc("Service identity"),
    )
    api_token: SecretStr | None = Field(
        None,
        description="If set, required as a bearer token on every route except `/`, `/ui` and `/metrics`.",
        json_schema_extra=_doc("Service identity"),
    )
    web_ui: bool = Field(
        True,
        description="Serve a web page at `/ui` to upload a video and download the blurred result.",
        json_schema_extra=_doc("Service identity"),
    )
    debug_videos: bool = Field(
        True,
        description=(
            "Accept `debug=1` jobs: an extra video (at most 1920 px wide) outlining every blurred "
            "region and sign with its class."
        ),
        json_schema_extra=_doc("Service identity", "Each debug job also encodes a second video."),
    )

    # --- Model and device -------------------------------------------------------------------
    model_name: str | None = Field(
        None,
        description=(
            "Registry entry to use (`sgblur-video models list`). Empty: the largest model of "
            "`MODEL_FAMILY` that fits the accelerator memory."
        ),
        json_schema_extra=_doc("Model and device", "Smaller models miss more objects."),
    )
    model_path: Path | None = Field(
        None,
        description=(
            "Local checkpoint (`.pt`) to use instead of the registry, to try any model: classes are "
            "read from the file and checked against `CLASS_POLICY`. Overrides `MODEL_NAME`."
        ),
        json_schema_extra=_doc("Model and device", "Check its blur results before using it for real."),
    )
    model_family: str = Field(
        "yolo26",
        description="Model family preferred by automatic selection.",
        json_schema_extra=_doc("Model and device"),
    )
    models_file: Path = Field(
        Path("models/registry.yaml"),
        description="Model registry (name, version, URL, SHA-256, expected classes).",
        json_schema_extra=_doc("Model and device"),
    )
    models_dir: Path = Field(
        Path.home() / ".cache" / "sgblur-video" / "models",
        description="Where weights are downloaded and verified (`/models` in Docker).",
        json_schema_extra=_doc("Model and device", doc_default="~/.cache/sgblur-video/models"),
    )
    device: str = Field(
        "auto",
        pattern=r"^(auto|cpu|mps|cuda(:\d+)?)$",
        description="`auto` picks CUDA, then MPS (Apple Silicon), then CPU.",
        json_schema_extra=_doc("Model and device"),
    )
    half: Literal["auto"] | bool = Field(
        "auto",
        description="FP16 inference (Ultralytics `quantize=16`). `auto`: on CUDA and Apple GPUs (MPS).",
        json_schema_extra=_doc("Model and device"),
    )
    class_policy: dict[str, ClassAction] = Field(
        default_factory=lambda: dict(DEFAULT_CLASS_POLICY),
        description=(
            "JSON object mapping class names to `blur` or `annotate`. A model lacking a `blur` class "
            "is refused at start-up."
        ),
        json_schema_extra=_doc("Model and device", "Removing a `blur` class leaves it visible."),
    )

    # --- Detection --------------------------------------------------------------------------
    detect_profile: DetectProfile = Field(
        DetectProfile.STANDARD,
        description="Detection passes per frame: `fast`, `standard` or `thorough`.",
        json_schema_extra=_doc("Detection", "`fast` found 3× fewer faces than `standard` on 8K video."),
    )
    tile_trigger_width: int = Field(
        5760,
        ge=1024,
        description="Frame long side (px) from which the tile pass runs.",
        json_schema_extra=_doc("Detection", "Higher values miss more small objects."),
    )
    equirect_pad_ratio: float = Field(
        0.0625,
        ge=0.0,
        le=0.25,
        description="Circular padding added on each side of 360° frames, as a fraction of the width.",
        json_schema_extra=_doc("Detection", "Lower values miss more objects at the 0°/360° seam."),
    )
    projection: Projection = Field(
        Projection.AUTO,
        description="Force the projection when metadata is missing or wrong.",
        json_schema_extra=_doc("Detection", "A wrong value breaks seam handling on 360° videos."),
    )
    conf_detect: float = Field(
        0.30,
        ge=0.0,
        le=1.0,
        description="Minimum detector score kept (and fed to the trackers).",
        json_schema_extra=_doc("Detection", "Higher values miss more objects."),
    )
    conf_blur: float = Field(
        0.40,
        ge=0.0,
        le=1.0,
        description=(
            "Score from which a face/plate detection is blurred on its own; lower-score detections are "
            "blurred when their track contains a detection at or above it."
        ),
        json_schema_extra=_doc("Detection", "Higher values leave more faces and plates visible."),
    )
    conf_sign: float = Field(
        0.6,
        ge=0.0,
        le=1.0,
        description="Minimum best score of a sign track to produce an annotation (SGBlur value).",
        json_schema_extra=_doc("Detection"),
    )
    detect_url: HttpUrl | None = Field(
        None,
        description="Remote Detect API. Empty: detection runs in the worker process (SGBlur convention).",
        json_schema_extra=_doc("Detection", "Videos are sent over the network: use a private network."),
    )

    # --- Tracking ---------------------------------------------------------------------------
    tracker_config: Path = Field(
        Path("configs/trackers/botsort.yaml"),
        description=(
            "Tracker YAML: BoT-SORT (`botsort.yaml`, default), another Ultralytics tracker or the "
            "optical-flow tracker (`flow.yaml`), plus the `track_buffer_s` extension."
        ),
        json_schema_extra=_doc("Tracking", "Gap filling relies on track continuity."),
    )
    track_width: int = Field(
        1920,
        ge=320,
        description="Width of the frame used for tracking and camera-motion compensation.",
        json_schema_extra=_doc("Tracking"),
    )

    # --- Post-processing and blur -----------------------------------------------------------
    blur_method: BlurMethod = Field(
        BlurMethod.PIXELATE_BLUR,
        description="Irreversible blur operation: `pixelate_blur`, `gaussian_strong` or `solid`.",
        json_schema_extra=_doc("Post-processing and blur", "`gaussian_strong` is the weakest option."),
    )
    pixelate_cells: int = Field(
        6,
        ge=2,
        le=32,
        description="Maximum number of mosaic cells on the long side of a blurred shape.",
        json_schema_extra=_doc("Post-processing and blur", "More cells keep more identity information."),
    )
    blur_box_margin: float = Field(
        0.05,
        ge=0.0,
        le=1.0,
        description="Enlargement of each box on each side, as a fraction of its width/height.",
        json_schema_extra=_doc("Post-processing and blur", "Lower values may leave edges visible."),
    )
    blur_temporal_padding_frames: int = Field(
        3,
        ge=0,
        description="Frames blurred before the first and after the last detection of a track or orphan.",
        json_schema_extra=_doc("Post-processing and blur", "Lower values expose objects at track ends."),
    )
    blur_padding_growth: float = Field(
        0.02,
        ge=0.0,
        le=1.0,
        description="Per-frame growth of padded boxes, to absorb motion uncertainty.",
        json_schema_extra=_doc("Post-processing and blur", "Lower values may miss moving objects."),
    )
    max_interpolation_gap_s: float = Field(
        0.3,
        ge=0.0,
        description="Longest gap inside a track that is filled by interpolation, in seconds.",
        json_schema_extra=_doc("Post-processing and blur", "Lower values leave gaps unblurred."),
    )
    max_interpolation_jump: float = Field(
        5.0,
        gt=0.0,
        description=(
            "Largest move, in box sizes, between two detections of a track that is filled by "
            "interpolation (larger jumps are two objects tracked as one)."
        ),
        json_schema_extra=_doc(
            "Post-processing and blur",
            "Lower values leave fast objects unblurred between detections; "
            "higher values sweep blur across the frame.",
        ),
    )
    link_max_gap_s: float = Field(
        0.5,
        ge=0.0,
        description=(
            "Longest interruption, in seconds, across which detections of one object are linked offline "
            "(flickering small objects)."
        ),
        json_schema_extra=_doc(
            "Post-processing and blur",
            "Lower values break objects into more pieces (more padding, not less blur).",
        ),
    )
    link_max_distance: float = Field(
        1.0,
        gt=0.0,
        description=(
            "Maximum distance, in box sizes, between where an object was heading and where it reappears."
        ),
        json_schema_extra=_doc("Post-processing and blur", "Lower values break objects into more pieces."),
    )
    sign_min_track_length: int = Field(
        5,
        ge=1,
        description="Minimum number of detections for a sign track to produce an annotation.",
        json_schema_extra=_doc("Post-processing and blur"),
    )
    # --- Encoding ---------------------------------------------------------------------------
    encoder: Encoder = Field(
        "auto",
        description="Video encoder; `auto` keeps the source codec and prefers hardware encoders.",
        json_schema_extra=_doc("Encoding"),
    )
    encode_bitrate_factor: float = Field(
        1.0,
        gt=0.0,
        le=4.0,
        description="Output video bit rate relative to the source.",
        json_schema_extra=_doc("Encoding"),
    )

    # --- Jobs, storage and limits -----------------------------------------------------------
    data_dir: Path = Field(
        Path("data"),
        description="Job database and per-job folders (`/data` in Docker). Use a local, dedicated disk.",
        json_schema_extra=_doc("Jobs, storage and limits", "Holds original videos while jobs run."),
    )
    tmp_dir: Path | None = Field(
        None,
        description="Scratch files. Empty: `DATA_DIR/tmp`.",
        json_schema_extra=_doc("Jobs, storage and limits", "Holds temporary personal data."),
    )
    result_ttl_minutes: int = Field(
        60,
        ge=1,
        description="How long results stay downloadable after a job succeeds.",
        json_schema_extra=_doc("Jobs, storage and limits", "Higher values keep results longer."),
    )
    keep_dir: Path | None = Field(
        None,
        description="Encrypted `keep=1` regions. Empty: `DATA_DIR/keep`.",
        json_schema_extra=_doc("Jobs, storage and limits"),
    )
    keep_ttl_hours: int = Field(
        48,
        ge=1,
        description="Lifetime of `keep=1` regions.",
        json_schema_extra=_doc("Jobs, storage and limits", "Higher values keep original pixels longer."),
    )
    keep_secret_key: SecretStr | None = Field(
        None,
        description=(
            "Server secret mixed with `blurring_id` to encrypt `keep=1` regions; empty disables `keep=1`."
        ),
        json_schema_extra=_doc(
            "Jobs, storage and limits", "Whoever holds this key and the store can read kept originals."
        ),
    )
    keep_max_confidence: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Only regions of tracks whose best score is below this are kept (likely false positives)."
        ),
        json_schema_extra=_doc("Jobs, storage and limits", "Higher values keep more original pixels."),
    )
    max_upload_bytes: int = Field(
        8 * GiB,
        ge=1,
        description="Upload size limit in bytes.",
        json_schema_extra=_doc("Jobs, storage and limits", doc_default="8589934592 (8 GiB)"),
    )
    max_video_duration_s: int = Field(
        1800,
        ge=1,
        description="Video duration limit in seconds.",
        json_schema_extra=_doc("Jobs, storage and limits"),
    )
    sync_max_duration_s: int = Field(
        30,
        ge=0,
        description="Maximum duration accepted with `sync=1`, in seconds (0 disables `sync=1`).",
        json_schema_extra=_doc("Jobs, storage and limits"),
    )
    accepted_containers: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["mp4", "mov"],
        description="Accepted containers, comma-separated.",
        json_schema_extra=_doc("Jobs, storage and limits"),
    )
    queue_max: int = Field(
        20,
        ge=1,
        description="Queued jobs before new submissions get `503 queue_full`.",
        json_schema_extra=_doc("Jobs, storage and limits"),
    )
    job_timeout_s: int = Field(
        6 * 3600,
        ge=60,
        description="Hard time limit per job; the job process is killed and its files deleted.",
        json_schema_extra=_doc("Jobs, storage and limits"),
    )
    callback_allowed_hosts: Annotated[list[str], NoDecode] = Field(
        default_factory=list,
        description="Hosts allowed in `callback_url`, comma-separated. Empty disables callbacks.",
        json_schema_extra=_doc("Jobs, storage and limits"),
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        "INFO",
        description="Log level. Logs never contain file names, coordinates or images.",
        json_schema_extra=_doc("Jobs, storage and limits"),
    )

    @field_validator(
        "model_name",
        "model_path",
        "detect_url",
        "api_token",
        "keep_secret_key",
        "tmp_dir",
        "keep_dir",
        mode="before",
    )
    @classmethod
    def _empty_string_is_none(cls, value: object) -> object:
        """Treat ``VAR=`` (empty) like an unset optional variable."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("model_path", mode="after")
    @classmethod
    def _expand_home(cls, value: Path | None) -> Path | None:
        """Accept ``~/models/x.pt`` (a ``.env`` file does not expand ``~``)."""
        return value.expanduser() if value is not None else None

    @field_validator("accepted_containers", "callback_allowed_hosts", mode="before")
    @classmethod
    def _split_comma_list(cls, value: object) -> object:
        """Accept ``a,b,c`` strings for list settings (environment-friendly)."""
        if isinstance(value, str):
            return [item.strip().lower() for item in value.split(",") if item.strip()]
        return value

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        """Reject combinations that would silently weaken privacy."""
        if self.conf_detect > self.conf_blur:
            msg = "CONF_DETECT must be lower than or equal to CONF_BLUR, otherwise blur candidates are lost."
            raise ValueError(msg)
        if ClassAction.BLUR not in self.class_policy.values():
            msg = "CLASS_POLICY must contain at least one class with action 'blur'."
            raise ValueError(msg)
        return self

    @property
    def effective_tmp_dir(self) -> Path:
        """Scratch directory: ``TMP_DIR`` or ``DATA_DIR/tmp``."""
        return self.tmp_dir if self.tmp_dir is not None else self.data_dir / "tmp"

    @property
    def effective_keep_dir(self) -> Path:
        """Directory of encrypted ``keep=1`` regions: ``KEEP_DIR`` or ``DATA_DIR/keep``."""
        return self.keep_dir if self.keep_dir is not None else self.data_dir / "keep"

    @property
    def keep_enabled(self) -> bool:
        """Whether ``keep=1`` requests can be honoured (a secret key is configured)."""
        return self.keep_secret_key is not None

    def classes_with(self, action: ClassAction) -> frozenset[str]:
        """Return the class names configured with a given action.

        Args:
            action: The action to look for.

        Returns:
            Class names whose policy is ``action``.

        Example:
            >>> sorted(Settings().classes_with(ClassAction.BLUR))
            ['face', 'plate']
        """
        return frozenset(name for name, act in self.class_policy.items() if act is action)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, read once from the environment and ``.env``.

    Returns:
        The cached :class:`Settings` instance.
    """
    return Settings()
