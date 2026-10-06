"""The manually annotated privacy dataset (``docs/design/testing-strategy.md``, method D).

The dataset is a local folder, **never committed**: its clips show real people.

```text
<dataset>/
├── manifest.yaml              # clip ids, source SHA-256, cut, annotation status
├── clips/<clip id>/
│   ├── clip.mp4               # the clip at full resolution (what the benchmark processes)
│   ├── proxy.mp4              # ≤ 3840 px wide copy, opened in CVAT
│   ├── preannotation.xml      # model pre-annotation, "CVAT for video 1.1"
│   └── ground_truth.json      # human annotation, full-resolution pixels
└── cache/<clip id>/*.jsonl    # detections.jsonl per model/tracker/profile (reused by sweeps)
```

Frame ``i`` of ``proxy.mp4`` is frame ``i`` of ``clip.mp4``: both are encoded
from the same decoded frames.
"""

import hashlib
import re
from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

MANIFEST = "manifest.yaml"
CLIP_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
GtClass = Literal["face", "plate"]


class DatasetError(ValueError):
    """The dataset folder or one of its files is invalid."""


class ClipEntry(BaseModel):
    """One clip of the dataset.

    Attributes:
        id: Short identifier chosen by the maintainer (no file name, no place name).
        source_sha256: SHA-256 of the original video the clip was cut from.
        start_s: Start of the clip in the original video, in seconds.
        frames: Number of frames.
        fps: Average frame rate.
        width: Clip width (pixels, coded frame).
        height: Clip height.
        proxy_width: Width of ``proxy.mp4``.
        proxy_height: Height of ``proxy.mp4``.
        projection: ``flat`` or ``equirectangular``.
        annotator: Who annotated it (free text, may be empty).
        annotated_on: Date of the import of the annotation.
        ground_truth_sha256: SHA-256 of the imported CVAT export.
    """

    id: str
    source_sha256: str
    start_s: float
    frames: int
    fps: float
    width: int
    height: int
    proxy_width: int
    proxy_height: int
    projection: str
    annotator: str = ""
    annotated_on: date | None = None
    ground_truth_sha256: str | None = None


class Manifest(BaseModel):
    """``manifest.yaml``."""

    version: Literal[1] = 1
    clips: list[ClipEntry] = Field(default_factory=list)

    def get(self, clip_id: str) -> ClipEntry:
        """The clip with this id.

        Raises:
            DatasetError: If there is none.
        """
        for clip in self.clips:
            if clip.id == clip_id:
                return clip
        msg = f"no clip {clip_id!r} in the dataset"
        raise DatasetError(msg)

    def put(self, entry: ClipEntry) -> None:
        """Add or replace a clip."""
        self.clips = [c for c in self.clips if c.id != entry.id] + [entry]
        self.clips.sort(key=lambda c: c.id)


class GtBox(BaseModel):
    """A ground-truth box on one frame (full-resolution pixels)."""

    frame: int
    box: tuple[float, float, float, float]
    readable: bool = False


class GtTrack(BaseModel):
    """One annotated face or plate, on the frames where it is visible."""

    id: str
    cls: GtClass
    boxes: list[GtBox]


class GroundTruth(BaseModel):
    """``ground_truth.json``."""

    version: Literal[1] = 1
    clip_id: str
    frames: int
    width: int
    height: int
    tracks: list[GtTrack]


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    """SHA-256 of a file, read by chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def check_clip_id(clip_id: str) -> str:
    """Validate a clip id (lower-case letters, digits, ``-`` and ``_``).

    Raises:
        DatasetError: If the id is not valid.
    """
    if not CLIP_ID.match(clip_id):
        msg = f"invalid clip id {clip_id!r}: use 1-64 lower-case letters, digits, '-' or '_'"
        raise DatasetError(msg)
    return clip_id


class Dataset:
    """Access to a dataset folder.

    Args:
        root: Dataset folder (created by ``annotate export`` if missing).
    """

    def __init__(self, root: Path) -> None:
        self.root = root

    def clip_dir(self, clip_id: str) -> Path:
        """Folder of a clip."""
        return self.root / "clips" / check_clip_id(clip_id)

    def clip_video(self, clip_id: str) -> Path:
        """Full-resolution clip."""
        return self.clip_dir(clip_id) / "clip.mp4"

    def proxy_video(self, clip_id: str) -> Path:
        """Annotation proxy."""
        return self.clip_dir(clip_id) / "proxy.mp4"

    def preannotation(self, clip_id: str) -> Path:
        """CVAT pre-annotation."""
        return self.clip_dir(clip_id) / "preannotation.xml"

    def ground_truth_path(self, clip_id: str) -> Path:
        """Imported ground truth."""
        return self.clip_dir(clip_id) / "ground_truth.json"

    def cache_dir(self, clip_id: str) -> Path:
        """Cached ``detections.jsonl`` files of a clip."""
        return self.root / "cache" / check_clip_id(clip_id)

    def load_manifest(self) -> Manifest:
        """Read ``manifest.yaml`` (an empty manifest if the dataset is new).

        Raises:
            DatasetError: If the manifest is not valid.
        """
        path = self.root / MANIFEST
        if not path.exists():
            return Manifest()
        try:
            return Manifest.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
        except ValueError as exc:
            msg = f"{path}: {exc}"
            raise DatasetError(msg) from exc

    def save_manifest(self, manifest: Manifest) -> None:
        """Write ``manifest.yaml``."""
        self.root.mkdir(parents=True, exist_ok=True)
        data = manifest.model_dump(mode="json")
        (self.root / MANIFEST).write_text(
            "# sgblur-video privacy dataset. Contains no media; clips live next to it.\n"
            + yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

    def load_ground_truth(self, clip_id: str) -> GroundTruth | None:
        """Ground truth of a clip, or ``None`` if it has not been annotated yet."""
        path = self.ground_truth_path(clip_id)
        if not path.exists():
            return None
        return GroundTruth.model_validate_json(path.read_text(encoding="utf-8"))

    def save_ground_truth(self, truth: GroundTruth) -> None:
        """Write ``ground_truth.json``."""
        path = self.ground_truth_path(truth.clip_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(truth.model_dump_json(indent=1), encoding="utf-8")
