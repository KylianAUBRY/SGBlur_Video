"""Streaming multipart upload straight to the job folder.

Starlette's ``UploadFile`` spools uploads to the system temporary directory and
only lets us check their size once they are complete. Videos are large and are
personal data, so the request body is parsed here with ``python-multipart``
and the ``video`` part is written directly to its final location, chunk by
chunk; the size limit is enforced while streaming and a partial file is
deleted on any error.
"""

import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from python_multipart.multipart import MultipartParser, parse_options_header

if TYPE_CHECKING:
    from python_multipart.multipart import MultipartCallbacks


@dataclass(frozen=True)
class ReceivedFile:
    """What was received.

    Attributes:
        size: Bytes written.
        suffix: Lower-case extension of the client's file name (``.mp4``…), or empty.
            The file name itself is never kept or logged.
    """

    size: int
    suffix: str


def safe_suffix(suffix: str) -> str:
    """Extension to store the upload under: the client's one if it is a plain extension, else ``.mp4``.

    The extension is kept (and not replaced by ``.mp4``) so that the probe can
    reject unsupported containers and raw 360° formats (``.insv``, ``.360``).
    """
    return suffix if re.fullmatch(r"\.[a-z0-9]{1,5}", suffix) else ".mp4"


class UploadError(ValueError):
    """The upload is invalid.

    Attributes:
        status: HTTP status to return (400, 413 or 415).
        code: Machine-readable error code.
    """

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code


async def receive_file(
    content_type: str | None,
    body: AsyncIterator[bytes],
    *,
    field_name: str,
    destination: Path,
    max_bytes: int,
) -> ReceivedFile:
    """Write the file part ``field_name`` of a multipart body to ``destination``.

    Args:
        content_type: Request ``Content-Type`` header.
        body: Request body chunks (``request.stream()``).
        field_name: Name of the file field (``video``).
        destination: Where to write the file (its folder must exist).
        max_bytes: Maximum file size; exceeding it raises a 413 error.

    Returns:
        Size and extension of the received file.

    Raises:
        UploadError: Not multipart, missing field, empty file, or too large.
    """
    kind, options = parse_options_header(content_type or "")
    boundary = options.get(b"boundary")
    if kind != b"multipart/form-data" or not boundary:
        raise UploadError(415, "unsupported_media_type", "Send the video as multipart/form-data.")

    collector = _FileCollector(field_name.encode(), destination, max_bytes)
    parser = MultipartParser(boundary, collector.callbacks())
    try:
        async for chunk in body:
            parser.write(chunk)
        parser.finalize()
    except UploadError:
        collector.discard()
        raise
    except Exception as exc:
        collector.discard()
        raise UploadError(400, "invalid_parameter", "Malformed multipart body.") from exc
    collector.close()
    if collector.written == 0:
        collector.discard()
        raise UploadError(422, "invalid_parameter", f"Missing or empty file field '{field_name}'.")
    return ReceivedFile(size=collector.written, suffix=collector.suffix)


class _FileCollector:
    """Multipart parser callbacks writing one named file part to disk."""

    def __init__(self, field_name: bytes, destination: Path, max_bytes: int) -> None:
        self._wanted = field_name
        self._destination = destination
        self._max = max_bytes
        self._handle = destination.open("wb")
        self._field = b""
        self._header_field = b""
        self._header_value = b""
        self._suffix = ""
        self.written = 0

    @property
    def suffix(self) -> str:
        """Extension of the wanted part's file name."""
        return self._suffix

    def callbacks(self) -> MultipartCallbacks:
        """Callbacks for ``python_multipart.multipart.MultipartParser``."""
        return {
            "on_part_begin": self._on_part_begin,
            "on_header_field": self._on_header_field,
            "on_header_value": self._on_header_value,
            "on_header_end": self._on_header_end,
            "on_part_data": self._on_part_data,
        }

    def _on_part_begin(self) -> None:
        self._field = b""

    def _on_header_field(self, data: bytes, start: int, end: int) -> None:
        self._header_field += data[start:end]

    def _on_header_value(self, data: bytes, start: int, end: int) -> None:
        self._header_value += data[start:end]

    def _on_header_end(self) -> None:
        if self._header_field.lower() == b"content-disposition":
            _disposition, params = parse_options_header(self._header_value)
            self._field = params.get(b"name", b"")
            if self._field == self._wanted:
                filename = params.get(b"filename", b"").decode("utf-8", "replace")
                self._suffix = PurePosixPath(filename.replace("\\", "/")).suffix.lower()[:8]
        self._header_field, self._header_value = b"", b""

    def _on_part_data(self, data: bytes, start: int, end: int) -> None:
        if self._field != self._wanted:
            return
        self.written += end - start
        if self.written > self._max:
            raise UploadError(413, "file_too_large", f"File larger than {self._max} bytes.")
        self._handle.write(data[start:end])

    def close(self) -> None:
        """Close the file."""
        self._handle.close()

    def discard(self) -> None:
        """Close and delete the partial file."""
        self._handle.close()
        self._destination.unlink(missing_ok=True)
