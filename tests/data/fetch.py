"""Download free-licence media fixtures on demand (pinned URL + SHA-256, cached outside the repo).

No media is committed to the repository (see ``tests/data/README.md``).
Fixtures are cached in ``~/.cache/sgblur-video/fixtures`` (``SGBLUR_FIXTURES_DIR``
overrides it) and verified before use; tests skip when they cannot be fetched.
"""

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest

GPMF_PARSER_COMMIT = "9a7150632892c7356c91145c889016f07b0ed48d"


@dataclass(frozen=True)
class Fixture:
    """A downloadable test file."""

    name: str
    url: str
    sha256: str
    licence: str


FIXTURES = {
    "gopro-hero6-gps": Fixture(
        name="hero6.mp4",
        url=f"https://raw.githubusercontent.com/gopro/gpmf-parser/{GPMF_PARSER_COMMIT}/samples/hero6.mp4",
        sha256="84aebc4e370ef9081f9015bf310d7a858d431258f6b0f2160d731c4308249c67",
        licence="Apache-2.0 OR MIT (gopro/gpmf-parser samples)",
    ),
}


def _cache_dir() -> Path:
    return Path(os.environ.get("SGBLUR_FIXTURES_DIR", Path.home() / ".cache" / "sgblur-video" / "fixtures"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture_path(key: str) -> Path:
    """Path of a verified fixture, downloading it if needed; skips the test when offline."""
    fixture = FIXTURES[key]
    path = _cache_dir() / fixture.name
    if path.exists() and _sha256(path) == fixture.sha256:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        response = httpx.get(fixture.url, follow_redirects=True, timeout=120)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        pytest.skip(f"fixture {fixture.name} unavailable: {exc}")
    path.write_bytes(response.content)
    if _sha256(path) != fixture.sha256:
        path.unlink()
        pytest.fail(f"fixture {fixture.name} has an unexpected SHA-256")
    return path
