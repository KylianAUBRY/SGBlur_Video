"""The HTTP routes of both applications match the documented contract (docs/design/openapi.yaml)."""

from pathlib import Path

import yaml

from sgblur_video.api import blur_api, detect_api
from sgblur_video.config import Settings

METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE"}
GENERATED = ("/docs", "/openapi", "/redoc")


def test_routes_match_openapi_contract(tmp_path: Path, repo_root: Path) -> None:
    spec = yaml.safe_load((repo_root / "docs" / "design" / "openapi.yaml").read_text(encoding="utf-8"))
    documented = {
        (method.upper(), path) for path, operations in spec["paths"].items() for method in operations
    } & {(m, p) for m in METHODS for p in spec["paths"]}
    settings = Settings(data_dir=tmp_path)
    served = {
        (method, route.path)
        for app in (blur_api.create_app(settings), detect_api.create_app(settings))
        for route in app.routes
        if not route.path.startswith(GENERATED)
        for method in getattr(route, "methods", None) or ()
        if method in METHODS
    }
    assert served == documented
