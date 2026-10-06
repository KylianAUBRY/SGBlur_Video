"""``keep=1``: encrypted, short-lived copies of regions that may have been blurred by mistake.

Panoramax can ask the blurring service to keep the unblurred parts of a
picture so that a false positive (a blurred sign mistaken for a plate, say) can
be un-blurred later (``PICTURE_PROCESS_KEEP_UNBLURRED_PARTS``). Like SGBlur,
only **low-confidence** regions are kept: chains whose best score is below
``KEEP_MAX_CONFIDENCE``. Unlike SGBlur, they are encrypted and expire:

* crops of the original frames are collected during rendering, then packed in
  a ZIP with a manifest (frame, box, chain, class, score);
* the archive is encrypted with AES-256-GCM, with a key derived (HKDF-SHA256)
  from the server secret ``KEEP_SECRET_KEY`` and the job's ``blurring_id``: the
  store alone, or the id alone, is not enough to read it;
* the file name is the SHA-256 of the ``blurring_id`` (not guessable, not linkable);
* files older than ``KEEP_TTL_HOURS`` are deleted by the worker's janitor.

The un-blur route that would use these archives is planned for v2.
"""

import hashlib
import io
import json
import os
import time
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import av
import numpy as np
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from PIL import Image

from sgblur_video.core.geometry import clip, round_out
from sgblur_video.core.postprocess import BlurShape

_INFO = b"sgblur-video keep v1"
_NONCE_BYTES = 12


def _key(secret: str, blurring_id: str) -> bytes:
    kdf = HKDF(algorithm=hashes.SHA256(), length=32, salt=blurring_id.encode(), info=_INFO)
    return kdf.derive(secret.encode())


def archive_path(keep_dir: Path, blurring_id: str) -> Path:
    """Location of the encrypted archive of a job."""
    return keep_dir / f"{hashlib.sha256(blurring_id.encode()).hexdigest()}.bin"


def encrypt(data: bytes, *, secret: str, blurring_id: str) -> bytes:
    """Encrypt with AES-256-GCM (random nonce prepended, ``blurring_id`` as associated data)."""
    nonce = os.urandom(_NONCE_BYTES)
    return nonce + AESGCM(_key(secret, blurring_id)).encrypt(nonce, data, blurring_id.encode())


def decrypt(blob: bytes, *, secret: str, blurring_id: str) -> bytes:
    """Decrypt an archive.

    Raises:
        cryptography.exceptions.InvalidTag: With a wrong secret or ``blurring_id``, or if tampered with.
    """
    nonce, payload = blob[:_NONCE_BYTES], blob[_NONCE_BYTES:]
    return AESGCM(_key(secret, blurring_id)).decrypt(nonce, payload, blurring_id.encode())


class KeepRecorder:
    """Collect original crops of low-confidence regions during rendering (a render region sink).

    Args:
        chain_scores: Best score of each chain (from the blur plan).
        max_confidence: ``KEEP_MAX_CONFIDENCE``.
    """

    def __init__(self, chain_scores: Mapping[str, float], max_confidence: float) -> None:
        self._scores = dict(chain_scores)
        self._max = max_confidence
        self._buffer = io.BytesIO()
        self._zip = zipfile.ZipFile(self._buffer, "w", compression=zipfile.ZIP_STORED)
        self._manifest: list[dict[str, object]] = []

    def __call__(self, index: int, frame: av.VideoFrame, shapes: Sequence[BlurShape]) -> None:
        """Store the original pixels of the kept shapes of frame ``index``."""
        kept = [s for s in shapes if self._scores.get(s.track_id, 1.0) < self._max]
        if not kept:
            return
        image = np.asarray(frame.to_ndarray(format="rgb24"), dtype=np.uint8)
        height, width = image.shape[:2]
        for number, shape in enumerate(kept):
            x1, y1, x2, y2 = round_out(clip(shape.box, width, height))
            if x2 <= x1 or y2 <= y1:
                continue
            name = f"{index:07d}_{number}.png"
            crop = io.BytesIO()
            Image.fromarray(image[y1:y2, x1:x2]).save(crop, format="PNG")
            self._zip.writestr(name, crop.getvalue())
            self._manifest.append(
                {
                    "file": name,
                    "frame": index,
                    "box": [x1, y1, x2, y2],
                    "chain": shape.track_id,
                    "class": shape.cls,
                    "score": round(self._scores.get(shape.track_id, 0.0), 3),
                }
            )

    @property
    def regions(self) -> int:
        """Number of kept regions so far."""
        return len(self._manifest)

    def save(self, keep_dir: Path, *, secret: str, blurring_id: str) -> Path | None:
        """Encrypt and write the archive; nothing is written when no region was kept."""
        manifest = {"blurring_id": blurring_id, "regions": self._manifest}
        self._zip.writestr("manifest.json", json.dumps(manifest))
        self._zip.close()
        if not self._manifest:
            return None
        keep_dir.mkdir(parents=True, exist_ok=True)
        path = archive_path(keep_dir, blurring_id)
        path.write_bytes(encrypt(self._buffer.getvalue(), secret=secret, blurring_id=blurring_id))
        return path


def purge_expired(keep_dir: Path, ttl_hours: float, *, now: float | None = None) -> int:
    """Delete archives older than ``ttl_hours``; returns how many were deleted."""
    if not keep_dir.exists():
        return 0
    limit = (now if now is not None else time.time()) - ttl_hours * 3600
    deleted = 0
    for path in keep_dir.glob("*.bin"):
        if path.stat().st_mtime < limit:
            path.unlink(missing_ok=True)
            deleted += 1
    return deleted
