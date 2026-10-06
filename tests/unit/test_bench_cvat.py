from pathlib import Path

import pytest

from sgblur_video.bench.cvat import KeyBox, PreTrack, read_cvat_video, write_preannotation
from sgblur_video.bench.dataset import DatasetError

SIZE = {"frames": 30, "width": 200, "height": 100, "proxy_width": 100, "proxy_height": 50}


def _export(path: Path, body: str, *, size: int = 30, original: tuple[int, int] = (100, 50)) -> Path:
    path.write_text(
        f"""<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <meta><job><size>{size}</size>
    <original_size><width>{original[0]}</width><height>{original[1]}</height></original_size>
  </job></meta>
  {body}
</annotations>""",
        encoding="utf-8",
    )
    return path


def _box(frame: int, x: float, *, outside: int = 0, keyframe: int = 1, readable: str = "false") -> str:
    return (
        f'<box frame="{frame}" outside="{outside}" occluded="0" keyframe="{keyframe}" '
        f'xtl="{x}" ytl="10" xbr="{x + 10}" ybr="20" z_order="0">'
        f'<attribute name="readable">{readable}</attribute></box>'
    )


def test_preannotation_round_trip(tmp_path: Path) -> None:
    track = PreTrack("face", (KeyBox(0, (0.0, 10.0, 10.0, 20.0)), KeyBox(10, (20.0, 10.0, 30.0, 20.0))), 20)
    path = tmp_path / "pre.xml"
    write_preannotation(path, [track], frames=30, width=100, height=50)
    tracks, skipped = read_cvat_video(path, **SIZE)
    assert not skipped
    assert len(tracks) == 1
    boxes = tracks[0].boxes
    # Visible from frame 0 to 19 (outside at 20), interpolated between key frames, scaled ×2.
    assert [b.frame for b in boxes] == list(range(20))
    assert boxes[5].box == pytest.approx((20.0, 20.0, 40.0, 40.0))
    assert boxes[15].box == pytest.approx((40.0, 20.0, 60.0, 40.0))
    assert tracks[0].cls == "face"
    assert not any(b.readable for b in boxes)


def test_track_without_outside_lasts_until_the_end(tmp_path: Path) -> None:
    path = _export(tmp_path / "a.xml", f'<track id="3" label="plate">{_box(25, 0)}</track>')
    tracks, _ = read_cvat_video(path, **SIZE)
    assert [b.frame for b in tracks[0].boxes] == [25, 26, 27, 28, 29]
    assert tracks[0].id == "plate:3"


def test_full_export_with_readable_flag_and_reappearance(tmp_path: Path) -> None:
    boxes = (
        _box(0, 0, readable="true")
        + _box(1, 2, keyframe=0, readable="true")
        + _box(2, 4, outside=1)
        + _box(6, 4)
        + _box(7, 4, outside=1)
    )
    path = _export(tmp_path / "b.xml", f'<track id="0" label="face">{boxes}</track>')
    tracks, _ = read_cvat_video(path, **SIZE)
    result = tracks[0].boxes
    assert [b.frame for b in result] == [0, 1, 6]
    assert [b.readable for b in result] == [True, True, False]


def test_boxes_are_clipped_to_the_frame(tmp_path: Path) -> None:
    path = _export(
        tmp_path / "c.xml",
        '<track id="0" label="face"><box frame="0" outside="0" keyframe="1" '
        'xtl="-20" ytl="-5" xbr="5" ybr="5"/><box frame="1" outside="1" keyframe="1" '
        'xtl="-20" ytl="-5" xbr="5" ybr="5"/></track>',
    )
    tracks, _ = read_cvat_video(path, **SIZE)
    assert tracks[0].boxes[0].box == (0.0, 0.0, 10.0, 10.0)


def test_other_labels_and_shapes_are_skipped(tmp_path: Path) -> None:
    path = _export(
        tmp_path / "d.xml",
        f'<track id="0" label="sign">{_box(0, 0)}</track>'
        '<track id="1" label="face"><polygon frame="0" points="0,0;1,1;2,0"/></track>',
    )
    tracks, skipped = read_cvat_video(path, **SIZE)
    assert tracks == []
    assert skipped == {"label:sign": 1, "shape:polygon": 1}


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("not xml", "not an XML file"),
        ("<root/>", "not a CVAT export"),
        ("<annotations><image id='0'/></annotations>", "CVAT for images"),
    ],
)
def test_invalid_exports_are_rejected(tmp_path: Path, content: str, message: str) -> None:
    path = tmp_path / "bad.xml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(DatasetError, match=message):
        read_cvat_video(path, **SIZE)


def test_export_of_another_video_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="31 frames annotated"):
        read_cvat_video(_export(tmp_path / "e.xml", "", size=31), **SIZE)
    with pytest.raises(DatasetError, match="annotated a 640x360 video"):
        read_cvat_video(_export(tmp_path / "f.xml", "", original=(640, 360)), **SIZE)
