#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        return default if value == "" else float(value)
    except Exception:
        return default


def evaluate(
    quality_csv: Path,
    *,
    min_raw_precision: float,
    min_final_area: int,
    min_final_precision: float,
    occlusion_aware: bool,
    min_occluded_final_area: int,
) -> dict[str, object]:
    with quality_csv.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    failed = []
    for row in rows:
        frame = int(float(row.get("frame", "0") or 0))
        raw_precision = f(row, "raw_overlap_precision")
        final_area = int(f(row, "final_changed_area_px"))
        final_precision = f(row, "final_overlap_precision")
        occlusion_pass = occlusion_aware and final_precision >= min_final_precision and final_area >= min_occluded_final_area
        reasons = []
        if raw_precision < min_raw_precision and not occlusion_pass:
            reasons.append("raw_render_not_on_visible_object")
        if final_area < min_final_area and not occlusion_pass:
            reasons.append("too_little_visible_overlay_after_clipping")
        if final_precision < min_final_precision:
            reasons.append("final_overlay_not_on_visible_object")
        if reasons:
            failed.append(
                {
                    "frame": frame,
                    "reasons": reasons,
                    "raw_overlap_precision": raw_precision,
                    "final_changed_area_px": final_area,
                    "final_overlap_precision": final_precision,
                }
            )
    raw_values = [f(row, "raw_overlap_precision") for row in rows if row.get("raw_overlap_precision", "") != ""]
    final_areas = [int(f(row, "final_changed_area_px")) for row in rows]
    return {
        "quality_csv": str(quality_csv),
        "frames": len(rows),
        "min_raw_precision": min_raw_precision,
        "min_final_area": min_final_area,
        "min_final_precision": min_final_precision,
        "occlusion_aware": occlusion_aware,
        "min_occluded_final_area": min_occluded_final_area,
        "raw_precision_min": min(raw_values) if raw_values else 0.0,
        "raw_precision_median": sorted(raw_values)[len(raw_values) // 2] if raw_values else 0.0,
        "final_area_min": min(final_areas) if final_areas else 0,
        "final_area_median": sorted(final_areas)[len(final_areas) // 2] if final_areas else 0,
        "failed_frames": failed,
        "failed_frame_count": len(failed),
        "pass": len(failed) == 0,
        "note": (
            "Pass requires the rendered overlay to lie on the visible object. "
            "In occlusion-aware mode, a low raw overlap is allowed when the clipped with-human overlay has high final precision and non-trivial visible area."
        ),
    }


def write_markdown(path: Path, report: dict[str, object]) -> None:
    failed = list(report["failed_frames"])  # type: ignore[arg-type]
    lines = [
        "# Overlay Acceptance Report",
        "",
        f"Pass: `{report['pass']}`",
        f"Frames: `{report['frames']}`",
        f"Failed frames: `{report['failed_frame_count']}`",
        f"Raw precision median: `{report['raw_precision_median']}`",
        f"Final visible area median: `{report['final_area_median']}`",
        "",
        "## Failed Frames",
        "",
    ]
    for row in failed[:80]:
        lines.append(
            f"- frame `{row['frame']}` reasons `{','.join(row['reasons'])}` "
            f"raw_precision `{row['raw_overlap_precision']:.3f}` "
            f"final_precision `{row['final_overlap_precision']:.3f}` "
            f"final_area `{row['final_changed_area_px']}`"
        )
    if len(failed) > 80:
        lines.append(f"- ... {len(failed) - 80} more")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate whether rendered overlay is actually on the visible object.")
    ap.add_argument("--quality-csv", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-md", type=Path, required=True)
    ap.add_argument("--min-raw-precision", type=float, default=0.25)
    ap.add_argument("--min-final-area", type=int, default=500)
    ap.add_argument("--min-final-precision", type=float, default=0.95)
    ap.add_argument("--occlusion-aware", action="store_true")
    ap.add_argument("--min-occluded-final-area", type=int, default=150)
    args = ap.parse_args()
    report = evaluate(
        args.quality_csv,
        min_raw_precision=args.min_raw_precision,
        min_final_area=args.min_final_area,
        min_final_precision=args.min_final_precision,
        occlusion_aware=args.occlusion_aware,
        min_occluded_final_area=args.min_occluded_final_area,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(args.out_md, report)
    print(json.dumps(report, indent=2)[:4000])
    if not report["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
