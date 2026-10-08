"""Markdown rendering of benchmark reports (JSON reports are written as is)."""

import json
from pathlib import Path
from typing import Any


def _table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    def cell(value: object) -> str:
        if isinstance(value, float):
            return f"{value:.4g}"
        if isinstance(value, dict):
            return ", ".join(f"{k} {cell(v)}" for k, v in value.items())
        return "—" if value is None else str(value)

    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(cell(row.get(c)) for c in columns) + " |" for row in rows]
    return lines


def _pct(value: float) -> str:
    return f"{value:.2%}"


def _privacy(report: dict[str, Any]) -> list[str]:
    lines = [f"Model: `{report['model']}`. Thresholds: `{report['thresholds']}`.", ""]
    rows = []
    for run in report["runs"]:
        summary = run["summary"]
        rows.append(
            {
                "overrides": ", ".join(f"{k}={v}" for k, v in run["overrides"].items()) or "defaults",
                "gate": "pass" if run["gate"]["passed"] else "FAIL",
                "leakage": _pct(summary["overall"]["leakage_rate"]),
                "readable leakage": _pct(summary["readable"]["leakage_rate"]),
                "tracks ever leaked": f"{summary['tracks_ever_leaked']}/{summary['tracks']}",
                "longest exposure (readable)": summary["longest_exposure_frames_readable"],
                "transient": summary["transient_exposures"],
                "over-blur": _pct(summary["over_blur_ratio"]),
            }
        )
    lines += _table(rows, list(rows[0]))
    for run in report["runs"]:
        title = ", ".join(f"{k}={v}" for k, v in run["overrides"].items()) or "defaults"
        summary = run["summary"]
        lines += ["", f"### {title}", ""]
        if run["gate"]["failures"]:
            lines += [f"- **{failure}**" for failure in run["gate"]["failures"]] + [""]
        breakdown = [{"slice": f"class {name}", **values} for name, values in summary["by_class"].items()] + [
            {"slice": f"size {name}", **values} for name, values in summary["by_size"].items()
        ]
        breakdown.append({"slice": "readable", **summary["readable"]})
        lines += _table(breakdown, ["slice", "object_frames", "unprotected", "leakage_rate"])
        clips = [
            {
                "clip": clip,
                "leakage": _pct(s["overall"]["leakage_rate"]),
                "object-frames": s["overall"]["object_frames"],
            }
            | {"video": f"{s['video']['width']}x{s['video']['height']} {s['video']['projection']}"}
            for clip, s in run["per_clip"].items()
        ]
        lines += ["", *_table(clips, ["clip", "video", "object-frames", "leakage"])]
    return lines


def to_markdown(report: dict[str, Any]) -> str:
    """A short Markdown summary of a report."""
    kind = report["kind"]
    lines = [
        f"## Benchmark: {kind}",
        "",
        f"sgblur-video {report['sgblur_video']}, {report['generated_at']}.",
        "",
    ]
    if kind == "privacy":
        lines += _privacy(report)
    elif kind == "speed":
        video = report["video"]
        lines += [
            f"Video: {video['width']}x{video['height']} {video['projection']} {video['codec']}, "
            f"{report['frames']} timed frames (median).",
            "",
            *_table(
                report["rows"],
                ["model", "device", "profile", "passes", "s_per_frame", "fps", "detections_per_frame"],
            ),
        ]
    return "\n".join(lines) + "\n"


def write_report(report: dict[str, Any], folder: Path) -> tuple[Path, Path]:
    """Write ``<kind>-<timestamp>.json`` and ``.md`` into ``folder``."""
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{report['kind']}-{report['generated_at'].replace(':', '').replace('+0000', 'Z')}"
    json_path, md_path = folder / f"{stem}.json", folder / f"{stem}.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(to_markdown(report), encoding="utf-8")
    return json_path, md_path
