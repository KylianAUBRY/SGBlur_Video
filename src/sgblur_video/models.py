"""Model registry: which detection models exist, where to get them, how to pick one.

The registry is a YAML file (``MODELS_FILE``, default ``models/registry.yaml``)
listing every usable checkpoint with a pinned download URL and SHA-256. Code
never hard-codes a model name: switching to a new model (for example a future
YOLO27 release of SGBlur) means adding an entry. See
``docs/adr/0009-model-registry-and-class-policy.md``.

Classes are always referenced by **name**. :func:`check_class_policy` makes
sure a checkpoint provides every class the configuration wants to blur, so a
model trained with different classes can never silently disable blurring.

Example:
    >>> registry = load_registry(Path("models/registry.yaml"))  # doctest: +SKIP
    >>> select_model(registry, family="yolo26", available_memory_gib=None).name  # doctest: +SKIP
    'yolo26s'
"""

import hashlib
import logging
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, Self, cast

import httpx
import yaml
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

from sgblur_video.config import ClassAction, Settings

logger = logging.getLogger(__name__)


class RegistryError(ValueError):
    """The registry file is invalid or does not contain the requested model."""


class ClassPolicyError(ValueError):
    """A model does not provide a class that the configuration wants to blur."""


class ModelEntry(BaseModel):
    """One checkpoint of the registry.

    Attributes:
        name: Unique short name, used in ``MODEL_NAME`` and in semantic tags (``yolo26s``).
        family: Model family used by automatic selection (``yolo26``).
        version: Version published in ``detection_model`` tags (``<name>/<version>``).
        file: File name inside ``MODELS_DIR``.
        url: Pinned download URL (a commit-addressed URL, never a branch).
        local_path: Checkpoint used in place (``MODEL_PATH``), for entries made by :func:`local_entry`.
        sha256: Expected SHA-256 of the file, lowercase hex.
        size_bytes: Expected file size, used for download progress and sanity checks.
        classes: Class names provided by the checkpoint, in the model's order.
        train_imgsz: Image size used for training (informative).
        min_memory_gib: Free accelerator memory needed for the standard detection profile.
        licence: Licence statement of the upstream publisher, reproduced as is.
        source: Upstream project page.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    family: str = Field(min_length=1)
    version: str = Field(min_length=1)
    file: str = Field(pattern=r"^[^/\\]+$")
    url: HttpUrl | None = None
    local_path: Path | None = None
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int | None = Field(None, gt=0)
    classes: tuple[str, ...] = Field(min_length=1)
    train_imgsz: int = Field(gt=0)
    min_memory_gib: float = Field(ge=0)
    licence: str
    source: HttpUrl | None = None

    @model_validator(mode="after")
    def _unique_classes(self) -> Self:
        """Reject duplicated class names, and entries that can be neither downloaded nor found."""
        if len(set(self.classes)) != len(self.classes):
            msg = f"model {self.name!r}: duplicated class names in {list(self.classes)}"
            raise ValueError(msg)
        if self.url is None and self.local_path is None:
            msg = f"model {self.name!r}: a registry entry needs a url"
            raise ValueError(msg)
        return self

    @property
    def tag(self) -> str:
        """``<name>/<version>``, the model part of Panoramax ``detection_model`` values."""
        return f"{self.name}/{self.version}"


class ModelRegistry(BaseModel):
    """Content of the registry file."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1]
    models: tuple[ModelEntry, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_names(self) -> Self:
        """Reject duplicated model names."""
        names = [entry.name for entry in self.models]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            msg = f"duplicated model names: {duplicates}"
            raise ValueError(msg)
        local = [entry.name for entry in self.models if entry.url is None or entry.local_path is not None]
        if local:
            msg = f"registry entries need a pinned url and no local_path (use MODEL_PATH): {local}"
            raise ValueError(msg)
        return self

    def get(self, name: str) -> ModelEntry:
        """Return the entry called ``name``.

        Args:
            name: Model name, as in ``MODEL_NAME``.

        Returns:
            The matching entry.

        Raises:
            RegistryError: If no entry has this name.
        """
        for entry in self.models:
            if entry.name == name:
                return entry
        known = ", ".join(entry.name for entry in self.models)
        msg = f"unknown model {name!r}; known models: {known}"
        raise RegistryError(msg)


def load_registry(path: Path) -> ModelRegistry:
    """Load and validate a registry YAML file.

    Args:
        path: Path of the registry file.

    Returns:
        The validated registry.

    Raises:
        RegistryError: If the file cannot be read or is invalid.
    """
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return ModelRegistry.model_validate(data)
    except (OSError, yaml.YAMLError, ValueError) as exc:
        msg = f"invalid model registry {path}: {exc}"
        raise RegistryError(msg) from exc


def select_model(
    registry: ModelRegistry,
    *,
    family: str,
    available_memory_gib: float | None,
    name: str | None = None,
) -> ModelEntry:
    """Pick the model to load, mirroring SGBlur's automatic size selection.

    An explicit ``name`` always wins. Otherwise the candidates are the entries
    of ``family``: on CPU (``available_memory_gib is None``) the lightest one,
    on an accelerator the heaviest one whose ``min_memory_gib`` fits, falling
    back to the lightest one if none fits.

    Args:
        registry: The loaded registry.
        family: Preferred family (``MODEL_FAMILY``).
        available_memory_gib: Free accelerator memory, or ``None`` on CPU.
        name: Explicit model name (``MODEL_NAME``), if any.

    Returns:
        The selected entry.

    Raises:
        RegistryError: If ``name`` is unknown or the family has no entry.
    """
    if name:
        return registry.get(name)
    candidates = sorted(
        (entry for entry in registry.models if entry.family == family),
        key=lambda entry: entry.min_memory_gib,
    )
    if not candidates:
        msg = f"no model of family {family!r} in the registry"
        raise RegistryError(msg)
    if available_memory_gib is None:
        return candidates[0]
    fitting = [entry for entry in candidates if entry.min_memory_gib <= available_memory_gib]
    return fitting[-1] if fitting else candidates[0]


def check_class_policy(model_classes: Sequence[str], policy: Mapping[str, ClassAction]) -> list[str]:
    """Check that a model provides every class the policy wants to blur.

    Fails closed: a missing ``blur`` class is an error, because running anyway
    would leave those objects visible. Missing ``annotate`` classes and model
    classes without a policy are only reported.

    Args:
        model_classes: Class names provided by the checkpoint.
        policy: ``CLASS_POLICY`` mapping class names to actions.

    Returns:
        Human-readable warnings (empty when everything matches).

    Raises:
        ClassPolicyError: If a class with action ``blur`` is not provided by the model.

    Example:
        >>> warnings = check_class_policy(["sign", "face"], {"face": ClassAction.BLUR})
        >>> warnings
        ["model class 'sign' has no policy and will be ignored"]
    """
    provided = set(model_classes)
    missing_blur = sorted(
        name for name, action in policy.items() if action is ClassAction.BLUR and name not in provided
    )
    if missing_blur:
        msg = f"the model does not provide classes configured to be blurred: {missing_blur}"
        raise ClassPolicyError(msg)
    warnings = [
        f"class {name!r} is configured to be annotated but the model does not provide it"
        for name, action in policy.items()
        if action is ClassAction.ANNOTATE and name not in provided
    ]
    warnings += [
        f"model class {name!r} has no policy and will be ignored"
        for name in model_classes
        if name not in policy
    ]
    return warnings


class WeightsError(RuntimeError):
    """Model weights could not be downloaded or failed verification."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_weights(entry: ModelEntry, models_dir: Path, *, client: httpx.Client | None = None) -> Path:
    """Return the verified local path of a model, downloading it if needed.

    The file is downloaded to a temporary name, verified (size and SHA-256),
    then atomically renamed. An existing file with a wrong hash is replaced.

    Args:
        entry: Registry entry.
        models_dir: Destination folder (``MODELS_DIR``).
        client: HTTP client (tests inject a mock transport).

    Returns:
        Path of the verified weights.

    Raises:
        WeightsError: On download failure or verification mismatch.
    """
    if entry.local_path is not None:
        if not entry.local_path.is_file():
            msg = f"model file not found: {entry.local_path}"
            raise WeightsError(msg)
        return entry.local_path
    models_dir.mkdir(parents=True, exist_ok=True)
    target = models_dir / entry.file
    if target.exists() and _sha256(target) == entry.sha256:
        return target
    temporary = target.with_name(target.name + ".part")
    http = client or httpx.Client(follow_redirects=True, timeout=httpx.Timeout(30.0, read=300.0))
    logger.info("downloading model %s from %s", entry.name, entry.url)
    try:
        with http.stream("GET", str(entry.url)) as response:
            response.raise_for_status()
            with temporary.open("wb") as handle:
                for chunk in response.iter_bytes(1 << 20):
                    handle.write(chunk)
    except httpx.HTTPError as exc:
        temporary.unlink(missing_ok=True)
        msg = f"cannot download model {entry.name}: {exc}"
        raise WeightsError(msg) from exc
    finally:
        if client is None:
            http.close()
    size = temporary.stat().st_size
    digest = _sha256(temporary)
    if (entry.size_bytes is not None and size != entry.size_bytes) or digest != entry.sha256:
        temporary.unlink(missing_ok=True)
        msg = f"model {entry.name}: verification failed (size {size}, sha256 {digest})"
        raise WeightsError(msg)
    temporary.replace(target)
    return target


#: Suffixes that make ``MODEL_NAME`` / ``--model`` a checkpoint path rather than a registry name.
CHECKPOINT_SUFFIXES = (".pt", ".pth")


def is_model_path(value: str) -> bool:
    """Whether a ``--model`` / ``MODEL_NAME`` value designates a file rather than a registry entry."""
    return value.lower().endswith(CHECKPOINT_SUFFIXES) or "/" in value or "\\" in value


def checkpoint_info(path: Path) -> dict[str, object]:
    """What an Ultralytics checkpoint says about itself, without building the model.

    Args:
        path: ``.pt`` file.

    Returns:
        ``classes`` (names in model order), ``train_imgsz``, ``date`` and ``ultralytics``
        version when present, ``sha256`` and ``size_bytes``.

    Raises:
        WeightsError: If the file is missing or is not a readable checkpoint.
    """
    from ultralytics.nn.tasks import torch_safe_load

    if not path.is_file():
        msg = f"model file not found: {path}"
        raise WeightsError(msg)
    try:
        # Ultralytics' own loader (it maps the module names of older releases). Checkpoints are
        # pickles of model objects: like Ultralytics itself, this only loads files the operator
        # chose to run.
        checkpoint, _ = torch_safe_load(str(path))
    except Exception as exc:
        msg = f"{path.name}: not a readable PyTorch checkpoint ({type(exc).__name__}: {exc})"
        raise WeightsError(msg) from exc
    if not isinstance(checkpoint, dict):
        msg = f"{path.name}: not an Ultralytics checkpoint"
        raise WeightsError(msg)
    model = checkpoint.get("ema") or checkpoint.get("model")
    names = getattr(model, "names", None) or checkpoint.get("names") or {}
    classes = [str(names[k]) for k in sorted(names)] if isinstance(names, dict) else [str(n) for n in names]
    train_args = checkpoint.get("train_args") or {}
    return {
        "classes": classes,
        "train_imgsz": train_args.get("imgsz") if isinstance(train_args, dict) else None,
        "date": str(checkpoint.get("date") or "")[:10] or None,
        "ultralytics": checkpoint.get("version"),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def local_entry(path: Path) -> ModelEntry:
    """An entry for a checkpoint outside the registry (``MODEL_PATH`` or ``--model path``).

    Its name is the file name and its version ``local-<first 8 hex digits of the SHA-256>``,
    so Panoramax tags (``detection_model``) tell two local files apart.

    Raises:
        WeightsError: If the file is missing or unreadable.
        RegistryError: If the checkpoint names no class.
    """
    path = path.expanduser().resolve()
    info = checkpoint_info(path)
    classes = tuple(cast(list[str], info["classes"]))
    if not classes:
        msg = f"{path.name}: the checkpoint names no class"
        raise RegistryError(msg)
    name = re.sub(r"[^a-z0-9._-]+", "-", path.stem.lower()).strip("-._") or "local"
    imgsz = info["train_imgsz"]
    try:
        return ModelEntry(
            name=name,
            family="local",
            version=f"local-{str(info['sha256'])[:8]}",
            file=path.name,
            local_path=path,
            sha256=str(info["sha256"]),
            size_bytes=int(cast(int, info["size_bytes"])),
            classes=classes,
            train_imgsz=imgsz if isinstance(imgsz, int) and imgsz > 0 else 640,
            min_memory_gib=0,
            licence="unknown (local file)",
        )
    except ValueError as exc:
        msg = f"{path.name}: {exc}"
        raise RegistryError(msg) from exc


def resolve_model(
    settings: Settings, model: str | None = None, *, available_memory_gib: float | None = None
) -> ModelEntry:
    """The model to run, by precedence: ``model`` (CLI), ``MODEL_PATH``, ``MODEL_NAME``, automatic choice.

    ``model`` and ``MODEL_NAME`` may also be checkpoint paths (``*.pt`` or containing ``/``).

    Args:
        settings: Model settings.
        model: Registry name or checkpoint path given on the command line.
        available_memory_gib: Free accelerator memory for automatic selection (``None`` on CPU).

    Returns:
        A registry entry, or a local entry for a checkpoint file.

    Raises:
        RegistryError: On an unknown name or an invalid registry.
        WeightsError: If a checkpoint file is missing or unreadable.
    """
    choice = model or (str(settings.model_path) if settings.model_path else None) or settings.model_name
    if choice and is_model_path(choice):
        return local_entry(Path(choice))
    registry = load_registry(settings.models_file)
    return select_model(
        registry, family=settings.model_family, available_memory_gib=available_memory_gib, name=choice
    )
