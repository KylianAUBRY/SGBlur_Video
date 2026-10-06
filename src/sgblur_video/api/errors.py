"""API errors: FastAPI's ``{"detail": …}`` shape plus a machine-readable ``code``."""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

#: HTTP status used for each input rejection code of :class:`~sgblur_video.core.probe.UnsupportedVideoError`.
PROBE_STATUS = {"unsupported_media_type": 415, "unsupported_projection": 415, "video_too_long": 422}


class ApiError(Exception):
    """An error returned as ``{"detail": message, "code": code}``.

    Attributes:
        status: HTTP status.
        code: Machine-readable code (see ``docs/design/openapi.yaml``).
        headers: Extra response headers (``Retry-After``…).
    """

    def __init__(self, status: int, code: str, message: str, headers: dict[str, str] | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.headers = headers or {}


def install_error_handler(app: FastAPI) -> None:
    """Register the error handlers of an application: every error body is ``{"detail", "code"}``.

    Query parameters rejected by FastAPI's validation (e.g. ``keep=2``) become
    422 ``invalid_parameter`` with a readable message instead of FastAPI's list of errors.
    """

    @app.exception_handler(ApiError)
    async def _handle(_request: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(
            {"detail": str(exc), "code": exc.code}, status_code=exc.status, headers=exc.headers
        )

    @app.exception_handler(RequestValidationError)
    async def _handle_validation(_request: Request, exc: RequestValidationError) -> JSONResponse:
        problems = "; ".join(f"{_location(error)}: {error.get('msg', 'invalid')}" for error in exc.errors())
        return JSONResponse({"detail": problems, "code": "invalid_parameter"}, status_code=422)


def _location(error: dict[str, object]) -> str:
    """``keep`` for ``("query", "keep")``: the parameter name without its origin."""
    location = error.get("loc", ())
    parts = list(location)[1:] if isinstance(location, (list, tuple)) else []
    return ".".join(str(part) for part in parts) or "request"
