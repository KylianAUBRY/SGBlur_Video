"""GPS track of a video: read from its telemetry, queried by timestamp."""

import logging
from dataclasses import dataclass
from fractions import Fraction

import av
import numpy as np
import numpy.typing as npt

from sgblur_video.core.decode import to_fraction
from sgblur_video.core.probe import VideoInfo
from sgblur_video.telemetry.gpmf import GpsSample, gps_samples

logger = logging.getLogger(__name__)

GOPRO_HANDLER = "GoPro MET"
#: Positions are not extrapolated further than this from the first/last GPS sample.
MAX_EXTRAPOLATION_S = 1.0


@dataclass(frozen=True)
class Fix:
    """A position."""

    lat: float
    lon: float
    alt: float


class GpsTrack:
    """Time-ordered GPS samples with linear interpolation.

    Args:
        samples: GPS samples (any order).
        source: Telemetry format, reported in annotations (``gpmf``).
    """

    def __init__(self, samples: list[GpsSample], source: str) -> None:
        ordered = sorted(samples, key=lambda s: s.time)
        self.source = source
        self._t: npt.NDArray[np.float64] = np.array([s.time for s in ordered], dtype=np.float64)
        self._lat = np.array([s.lat for s in ordered], dtype=np.float64)
        self._lon = np.array([s.lon for s in ordered], dtype=np.float64)
        self._alt = np.array([s.alt for s in ordered], dtype=np.float64)

    def __len__(self) -> int:
        return len(self._t)

    def position_at(self, time: float) -> Fix | None:
        """Interpolated position at ``time`` seconds, or None outside the recorded track."""
        if (
            not len(self._t)
            or time < self._t[0] - MAX_EXTRAPOLATION_S
            or time > self._t[-1] + MAX_EXTRAPOLATION_S
        ):
            return None
        return Fix(
            lat=float(np.interp(time, self._t, self._lat)),
            lon=float(np.interp(time, self._t, self._lon)),
            alt=float(np.interp(time, self._t, self._alt)),
        )


def read_gps(info: VideoInfo) -> GpsTrack | None:
    """Read the GPS track of a video, if its telemetry is supported (GoPro GPMF).

    Args:
        info: Probed video.

    Returns:
        The GPS track, or None without supported telemetry or without any GPS fix
        (e.g. GoPro HERO12, which has no GPS receiver).
    """
    stream_info = next((s for s in info.streams if s.handler_name == GOPRO_HANDLER), None)
    if stream_info is None:
        return None
    samples: list[GpsSample] = []
    with av.open(str(info.path)) as container:
        stream = container.streams[stream_info.index]
        time_base = to_fraction(stream.time_base) if stream.time_base else Fraction(1, 1000)
        # Times count from the first video frame, like frame timestamps (decode.iter_frames):
        # the telemetry track may start later (e.g. in a frame range cut from a longer video).
        video = container.streams.video[0]
        origin = (video.start_time or 0) * (to_fraction(video.time_base) if video.time_base else Fraction(0))
        for packet in container.demux(stream):
            if packet.pts is None or not packet.size:
                continue
            start = float(packet.pts * time_base - origin)
            duration = float((packet.duration or 0) * time_base) or 1.0
            samples += gps_samples(bytes(packet), start, duration)
    if not samples:
        logger.info("telemetry present but no GPS fix")
        return None
    logger.info("GPS track: %d samples", len(samples))
    return GpsTrack(samples, source="gpmf")
