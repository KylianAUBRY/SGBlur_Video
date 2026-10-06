"""Read and write ``detections.jsonl`` (format version 1).

The file is the contract between pass 1 (analysis) and everything after it;
see ``docs/design/detections-format.md``. One JSON object per line: a
``header``, one ``frame`` per decoded frame, a ``footer``. Lines are written
and flushed incrementally so the file can be streamed by the Detect API.
"""

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import IO, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field

SCHEMA: Literal["sgblur-video/detections"] = "sgblur-video/detections"
VERSION: Literal[1] = 1


class DetectionsFormatError(ValueError):
    """The file is not a valid ``detections.jsonl``."""


class DetectionRecord(BaseModel):
    """One merged detection of a frame."""

    model_config = ConfigDict(extra="ignore")

    class_: str = Field(alias="class")
    score: float
    box: tuple[float, float, float, float]
    track_id: str | None = None
    passes: list[str] = Field(default_factory=list)


class FrameRecord(BaseModel):
    """Detections of one decoded frame."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["frame"] = "frame"
    index: int
    pts: int
    time: float
    detections: list[DetectionRecord] = Field(default_factory=list)


class Header(BaseModel):
    """First line: what was analysed, how, with which software."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["header"] = "header"
    schema_: Literal["sgblur-video/detections"] = Field(SCHEMA, alias="schema")
    version: Literal[1] = VERSION
    created_at: str
    video: dict[str, Any]
    model: dict[str, Any]
    detection: dict[str, Any]
    tracking: dict[str, Any]
    software: dict[str, Any]


class Footer(BaseModel):
    """Last line: completion status and counters."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["footer"] = "footer"
    frames: int
    complete: bool
    elapsed_s: float
    counts: dict[str, int] = Field(default_factory=dict)


@dataclass(frozen=True)
class Detections:
    """A whole ``detections.jsonl`` loaded in memory (boxes only, a few MB at most)."""

    header: Header
    frames: list[FrameRecord]
    footer: Footer | None

    @property
    def complete(self) -> bool:
        """Whether analysis covered the whole video."""
        return self.footer is not None and self.footer.complete


def _dump(model: BaseModel) -> str:
    return json.dumps(
        model.model_dump(mode="json", by_alias=True), separators=(", ", ": "), ensure_ascii=False
    )


class DetectionsWriter:
    """Incremental writer, used as a context manager.

    Example:
        >>> with DetectionsWriter(path, header) as writer:  # doctest: +SKIP
        ...     writer.write_frame(frame_record)
        ...     writer.close_with(footer)
    """

    def __init__(self, path: Path, header: Header) -> None:
        self._path = path
        self._file: IO[str] = path.open("w", encoding="utf-8")
        self._write(header)

    def _write(self, model: BaseModel) -> None:
        self._file.write(_dump(model) + "\n")
        self._file.flush()

    def write_frame(self, frame: FrameRecord) -> None:
        """Append one frame line."""
        self._write(frame)

    def close_with(self, footer: Footer) -> None:
        """Write the footer and close the file."""
        self._write(footer)
        self._file.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        if not self._file.closed:
            self._file.close()


def iter_lines(path: Path) -> Iterator[dict[str, Any]]:
    """Yield the JSON objects of a file, with clear errors on malformed lines."""
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                msg = f"{path.name}:{number}: invalid JSON"
                raise DetectionsFormatError(msg) from exc
            if not isinstance(value, dict) or "type" not in value:
                msg = f"{path.name}:{number}: expected an object with a 'type'"
                raise DetectionsFormatError(msg)
            yield value


def read_detections(path: Path) -> Detections:
    """Load and validate a ``detections.jsonl`` file.

    Args:
        path: File to read.

    Returns:
        Header, frames and footer (``None`` if the file is truncated).

    Raises:
        DetectionsFormatError: On a missing header, unknown version, frames out of order, or malformed lines.
    """
    lines = iter_lines(path)
    first = next(lines, None)
    if first is None or first.get("type") != "header":
        msg = f"{path.name}: the first line must be a header"
        raise DetectionsFormatError(msg)
    if first.get("schema") != SCHEMA or first.get("version") != VERSION:
        msg = f"{path.name}: unsupported schema/version {first.get('schema')!r}/{first.get('version')!r}"
        raise DetectionsFormatError(msg)
    header = Header.model_validate(first)
    frames: list[FrameRecord] = []
    footer: Footer | None = None
    for value in lines:
        if footer is not None:
            msg = f"{path.name}: content after the footer"
            raise DetectionsFormatError(msg)
        if value["type"] == "frame":
            frame = FrameRecord.model_validate(value)
            if frame.index != len(frames):
                msg = f"{path.name}: frame {frame.index} found where {len(frames)} was expected"
                raise DetectionsFormatError(msg)
            frames.append(frame)
        elif value["type"] == "footer":
            footer = Footer.model_validate(value)
    return Detections(header=header, frames=frames, footer=footer)
