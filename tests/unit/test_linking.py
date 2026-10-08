"""Tests of offline linking (signs and benchmark pre-annotation only; never used for blurring)."""

from sgblur_video.core.geometry import Box
from sgblur_video.core.linking import Fragment, Observation, link_fragments


def test_linking_respects_groups_distance_and_time() -> None:
    def fragment(group: str, frame: int, box: Box) -> Fragment:
        return Fragment(None, group, [Observation(frame, box, 0.5, group)])

    fragments = [
        fragment("signage", 0, (0, 0, 10, 10)),
        fragment("plate", 1, (0, 0, 10, 10)),  # other group
        fragment("signage", 2, (500, 500, 510, 510)),  # too far
        fragment("signage", 3, (2, 0, 12, 10)),  # same sign
        fragment("signage", 200, (4, 0, 14, 10)),  # too late
    ]
    chains = link_fragments(fragments, max_gap=30, max_distance=1.0)
    sizes = sorted(len(chain.observations) for chain in chains)
    assert sizes == [1, 1, 1, 2]
