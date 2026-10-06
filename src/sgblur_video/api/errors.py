"""API errors: FastAPI's ``{"detail": …}`` shape plus a machine-readable ``code``."""

from fastapi import FastAPI, Request
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
    """Register the :class:`ApiError` handler on an application."""

    @app.exception_handler(ApiError)
    async def _handle(_request: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(
            {"detail": str(exc), "code": exc.code}, status_code=exc.status, headers=exc.headers
        )
