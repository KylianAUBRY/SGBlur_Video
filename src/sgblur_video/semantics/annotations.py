"""Traffic-sign tracks → Panoramax annotations (one annotation per physical sign).

Signs and direction signs are never blurred. Their detections are grouped into
fragments (tracker tracks and orphans), linked offline exactly like faces and
plates (``docs/adr/0011-offline-linking.md``), filtered, and each remaining
chain becomes **one** annotation:

* ``shape``: the box of the chain's *best frame* (highest ``score × area``,
  real detections only), integer pixels in display orientation, clipped to the
  frame, top-left origin — the Panoramax picture format;
* ``semantics``: exactly SGBlur's tags (``osm|traffic_sign=yes``,
  ``detection_model[osm|traffic_sign=yes]``,
  ``detection_confidence[osm|traffic_sign=yes]``), string values;
* ``video``: an extension with the track id, first/best/last frame and
  timestamp, observation count and confidences. The current Panoramax backend
  ignores it (unknown annotation fields are dropped), see ``docs/design/api.md``.

The model string is ``{API_NAME}-{model}/{version}`` (e.g.
``SGBlur-Video-yolo26s/0.1.0``): its ``SGBlur-`` prefix is what the backend uses
to clean previous detection tags when a picture is re-blurred.

A second-stage classifier of sign types (e.g. Panoramax ``classified_fr_road_signs``,
tags such as ``osm|traffic_sign=FR:A15b``) would plug in after
:func:`find_sign_tracks`; it is documented future work.
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from sgblur_video.config import ClassAction, Settings
from sgblur_video.core.detect import class_groups, rotate_box
from sgblur_video.core.detections_io import Detections
from sgblur_video.core.geometry import area, clip
from sgblur_video.core.postprocess import Observation, fragments_from_detections, link_fragments
from sgblur_video.telemetry.gps import GpsTrack
from sgblur_video.video360.wrap import split

TRAFFIC_SIGN_KEY = "osm|traffic_sign"
TRAFFIC_SIGN_VALUE = "yes"
_QUALIFIED = f"[{TRAFFIC_SIGN_KEY}={TRAFFIC_SIGN_VALUE}]"


class SemanticTag(BaseModel):
    """A Panoramax semantic tag (``key=value``, both strings)."""

    model_config = ConfigDict(frozen=True)

    key: str = Field(max_length=256)
    value: str = Field(max_length=2048)


class Position(BaseModel):
    """Geographic position of a sign at its best frame (from video telemetry)."""

    lat: float
    lon: float
    alt: float | None = None
    source: str


class VideoTrackInfo(BaseModel):
    """Video extension of an annotation (not part of the Panoramax picture format)."""

    track_id: str
    class_: str = Field(alias="class")
    best_frame: int
    best_timestamp: float
    first_frame: int
    first_timestamp: float
    last_frame: int
    last_timestamp: float
    observations: int
    confidence_max: float
    confidence_mean: float
    position: Position | None = None

    model_config = ConfigDict(populate_by_name=True)


class Annotation(BaseModel):
    """A Panoramax annotation of a part of a picture/frame."""

    shape: tuple[int, int, int, int]
    semantics: list[SemanticTag]
    video: VideoTrackInfo


class Metadata(BaseModel):
    """The ``metadata`` object returned with a blurred video (SGBlur format + extensions)."""

    blurring_id: str | None = None
    service_name: str
    annotations: list[Annotation]
    video: dict[str, Any] = Field(default_factory=dict)
    stats: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class SignTrack:
    """One physical sign followed over time.

    Attributes:
        id: Chain identifier (``signage:3`` or ``signage:~5``).
        cls: Majority class (``sign`` or ``direction``).
        observations: Real detections, sorted by frame.
        best: Detection with the highest ``score × area``.
    """

    id: str
    cls: str
    observations: tuple[Observation, ...]
    best: Observation

    @property
    def confidence_max(self) -> float:
        """Best score of the track."""
        return max(o.score for o in self.observations)

    @property
    def confidence_mean(self) -> float:
        """Mean score of the track."""
        return sum(o.score for o in self.observations) / len(self.observations)


def find_sign_tracks(
    detections: Detections, settings: Settings, *, fps: float, wrap_width: int | None = None
) -> list[SignTrack]:
    """Deduplicate sign detections into physical signs.

    Args:
        detections: Content of ``detections.jsonl``.
        settings: ``CONF_SIGN``, ``SIGN_MIN_TRACK_LENGTH`` and linking settings.
        fps: Average frame rate (converts ``LINK_MAX_GAP_S`` to frames).
        wrap_width: Frame width of a 360° video (a sign crossing the seam stays one sign).

    Returns:
        Kept sign tracks, ordered by first appearance.
    """
    annotate = settings.classes_with(ClassAction.ANNOTATE)
    groups = {name: group for name, group in class_groups(settings.class_policy).items() if name in annotate}
    chains = link_fragments(
        fragments_from_detections(detections, groups, wrap_width=wrap_width),
        max_gap=max(1, round(settings.link_max_gap_s * fps)),
        max_distance=settings.link_max_distance,
        wrap_width=wrap_width,
    )
    tracks = []
    for chain in chains:
        observations = tuple(chain.observations)
        if len(observations) < settings.sign_min_track_length:
            continue
        if max(o.score for o in observations) < settings.conf_sign:
            continue
        best = max(observations, key=lambda o: (o.score * area(o.box), -o.frame))
        cls = Counter(o.cls for o in observations).most_common(1)[0][0]
        tracks.append(SignTrack(id=chain.id, cls=cls, observations=observations, best=best))
    tracks.sort(key=lambda t: t.observations[0].frame)
    return tracks


def display_shape(
    track: SignTrack, *, frame_size: tuple[int, int], rotation: int, wrap: bool = False
) -> tuple[int, int, int, int]:
    """Integer bbox of the best detection in display orientation, inside the frame.

    Args:
        track: Sign track.
        frame_size: Coded frame ``(width, height)``.
        rotation: Display rotation of the video in degrees.
        wrap: 360° video: a box crossing the 0°/360° seam is reduced to its larger visible part
            (Panoramax shapes cannot wrap around).

    Returns:
        ``(minx, miny, maxx, maxy)`` with ``minx < maxx`` and ``miny < maxy`` when possible.
    """
    width, height = frame_size
    best = track.best.box
    if wrap:
        best = max(split(best, width), key=lambda part: part[2] - part[0])
    box = rotate_box(best, rotation, width, height)
    turns = (round(rotation / 90) % 2) if rotation else 0
    display_w, display_h = (height, width) if turns else (width, height)
    x1, y1, x2, y2 = clip(box, display_w, display_h)
    return (round(x1), round(y1), max(round(x1) + 1, round(x2)), max(round(y1) + 1, round(y2)))


def model_tag(settings: Settings, model: dict[str, Any]) -> str:
    """``{API_NAME}-{model}/{version}``, the value of ``detection_model`` tags."""
    return f"{settings.api_name}-{model.get('name', 'unknown')}/{model.get('version', 'unknown')}"


def to_annotation(
    track: SignTrack,
    detections: Detections,
    settings: Settings,
    *,
    frame_size: tuple[int, int],
    rotation: int,
    wrap: bool = False,
    gps: GpsTrack | None = None,
) -> Annotation:
    """Build the Panoramax annotation of one sign track (with its GPS position when known)."""
    times = detections.frames
    tag = model_tag(settings, detections.header.model)
    semantics = [
        SemanticTag(key=TRAFFIC_SIGN_KEY, value=TRAFFIC_SIGN_VALUE),
        SemanticTag(key=f"detection_model{_QUALIFIED}", value=tag),
        SemanticTag(key=f"detection_confidence{_QUALIFIED}", value=f"{track.best.score:.3f}"),
    ]
    first, last = track.observations[0], track.observations[-1]
    video = VideoTrackInfo(
        track_id=track.id,
        class_=track.cls,
        best_frame=track.best.frame,
        best_timestamp=round(times[track.best.frame].time, 3),
        first_frame=first.frame,
        first_timestamp=round(times[first.frame].time, 3),
        last_frame=last.frame,
        last_timestamp=round(times[last.frame].time, 3),
        observations=len(track.observations),
        confidence_max=round(track.confidence_max, 3),
        confidence_mean=round(track.confidence_mean, 3),
        position=_position(gps, times[track.best.frame].time),
    )
    return Annotation(
        shape=display_shape(track, frame_size=frame_size, rotation=rotation, wrap=wrap),
        semantics=semantics,
        video=video,
    )


def _position(gps: GpsTrack | None, timestamp: float) -> Position | None:
    """GPS position at a timestamp, if the video has a usable GPS track."""
    if gps is None:
        return None
    fix = gps.position_at(timestamp)
    if fix is None:
        return None
    return Position(lat=round(fix.lat, 7), lon=round(fix.lon, 7), alt=round(fix.alt, 1), source=gps.source)


def build_annotations(
    tracks: Sequence[SignTrack],
    detections: Detections,
    settings: Settings,
    *,
    frame_size: tuple[int, int],
    rotation: int,
    wrap: bool = False,
    gps: GpsTrack | None = None,
) -> list[Annotation]:
    """Annotations of every kept sign track, in order of appearance."""
    return [
        to_annotation(t, detections, settings, frame_size=frame_size, rotation=rotation, wrap=wrap, gps=gps)
        for t in tracks
    ]


def dump_metadata(metadata: Metadata) -> dict[str, Any]:
    """JSON-ready metadata, with ``class`` (not ``class_``) in the video extension."""
    data: dict[str, Any] = metadata.model_dump(mode="json", by_alias=True, exclude_none=True)
    return data
