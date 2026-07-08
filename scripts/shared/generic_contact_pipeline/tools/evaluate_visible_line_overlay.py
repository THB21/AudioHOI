#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


def read_line_tracks(path: Path) -> dict[int, tuple[float, float, float, float]]:
    grouped: dict[int, dict[str, tuple[float, float]]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            pid = row.get("point_id", "")
            if pid not in {"tip_a", "tip_b"}:
                continue
            try:
                fr = int(float(row.get("frame", "0") or 0))
                grouped.setdefault(fr, {})[pid] = (float(row["x"]), float(row["y"]))
            except Exception:
                continue
    out = {}
    for fr, pts in grouped.items():
        if "tip_a" in pts and "tip_b" in pts:
            ax, ay = pts["tip_a"]
            bx, by = pts["tip_b"]
            out[fr] = (ax, ay, bx, by)
    return out


def line_mask(shape: tuple[int, int], track: tuple[float, float, float, float], *, pad: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    x1, y1, x2, y2 = track
    length = float(math.hypot(x2 - x1, y2 - y1))
    if length < 20.0:
        return mask
    thickness = max(pad, int(round(length * 0.045)))
    cv2.line(mask, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), 255, thickness, cv2.LINE_AA)
    return cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=1)


def frame_path(frames_dir: Path, frame: int) -> Path:
    for name in (f"{frame:05d}.png", f"{frame:06d}.png", f"{frame:04d}.png", f"{frame:03d}.png"):
        path = frames_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing frame {frame} under {frames_dir}")


def changed_mask(rendered: np.ndarray, original: np.ndarray) -> np.ndarray:
    delta = np.any(np.abs(rendered.astype(np.int16) - original.astype(np.int16)) > 5, axis=2)
    return (delta.astype(np.uint8) * 255)


def evaluate(
    *,
    overlay_video: Path,
    frames_dir: Path,
    tracks_csv: Path,
    out_json: Path,
    out_csv: Path,
    min_precision: float,
    min_coverage: float,
    min_area: int,
    pad_px: int,
) -> dict[str, object]:
    tracks = read_line_tracks(tracks_csv)
    cap = cv2.VideoCapture(str(overlay_video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {overlay_video}")
    rows: list[dict[str, object]] = []
    failed: list[dict[str, object]] = []
    for fr in sorted(tracks):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr - 1)
        ok, rendered = cap.read()
        if not ok:
            continue
        original = cv2.imread(str(frame_path(frames_dir, fr)), cv2.IMREAD_COLOR)
        if original is None:
            continue
        cmask = changed_mask(rendered, original)
        lmask = line_mask(cmask.shape, tracks[fr], pad=pad_px)
        overlay_area = int(np.count_nonzero(cmask))
        line_area = int(np.count_nonzero(lmask))
        overlap = int(np.count_nonzero(cv2.bitwise_and(cmask, lmask)))
        precision = overlap / overlay_area if overlay_area else 0.0
        coverage = overlap / line_area if line_area else 0.0
        reasons = []
        if overlay_area < min_area:
            reasons.append("too_little_visible_overlay")
        if precision < min_precision:
            reasons.append("overlay_pixels_not_on_tracked_line")
        if coverage < min_coverage:
            reasons.append("tracked_line_not_covered_by_overlay")
        row = {
            "frame": fr,
            "overlay_area_px": overlay_area,
            "tracked_line_area_px": line_area,
            "overlap_px": overlap,
            "precision_on_tracked_line": round(precision, 6),
            "coverage_of_tracked_line": round(coverage, 6),
            "pass": not reasons,
            "reasons": "|".join(reasons),
        }
        rows.append(row)
        if reasons:
            failed.append(row)
    cap.release()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        fields = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    precisions = [float(r["precision_on_tracked_line"]) for r in rows]
    coverages = [float(r["coverage_of_tracked_line"]) for r in rows]
    report = {
        "overlay_video": str(overlay_video),
        "tracks_csv": str(tracks_csv),
        "frames": len(rows),
        "min_precision": min_precision,
        "min_coverage": min_coverage,
        "min_area": min_area,
        "pad_px": pad_px,
        "precision_median": sorted(precisions)[len(precisions) // 2] if precisions else 0.0,
        "precision_min": min(precisions) if precisions else 0.0,
        "coverage_median": sorted(coverages)[len(coverages) // 2] if coverages else 0.0,
        "coverage_min": min(coverages) if coverages else 0.0,
        "failed_frame_count": len(failed),
        "failed_frames": failed[:120],
        "pass": len(failed) == 0,
        "csv": str(out_csv),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2) + "\n")
    return report


def evaluate_quality_csv(
    *,
    quality_csv: Path,
    out_json: Path,
    out_csv: Path,
    min_precision: float,
    min_coverage: float,
    min_area: int,
) -> dict[str, object]:
    with quality_csv.open(newline="") as f:
        source_rows = list(csv.DictReader(f))
    rows: list[dict[str, object]] = []
    failed: list[dict[str, object]] = []
    for row in source_rows:
        frame = int(float(row.get("frame", "0") or 0))
        overlay_area = int(float(row.get("final_changed_area_px", "0") or 0))
        line_area = int(float(row.get("visible_object_area_px", "0") or 0))
        overlap = int(float(row.get("final_overlap_px", "0") or 0))
        precision = float(row.get("final_overlap_precision", "0") or 0.0)
        coverage = overlap / line_area if line_area else 0.0
        reasons = []
        if overlay_area < min_area:
            reasons.append("too_little_visible_overlay")
        if precision < min_precision:
            reasons.append("overlay_pixels_not_on_tracked_line")
        if coverage < min_coverage:
            reasons.append("tracked_line_not_covered_by_overlay")
        out_row = {
            "frame": frame,
            "overlay_area_px": overlay_area,
            "tracked_line_area_px": line_area,
            "overlap_px": overlap,
            "precision_on_tracked_line": round(precision, 6),
            "coverage_of_tracked_line": round(coverage, 6),
            "pass": not reasons,
            "reasons": "|".join(reasons),
        }
        rows.append(out_row)
        if reasons:
            failed.append(out_row)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        fields = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    precisions = [float(r["precision_on_tracked_line"]) for r in rows]
    coverages = [float(r["coverage_of_tracked_line"]) for r in rows]
    report = {
        "quality_csv": str(quality_csv),
        "frames": len(rows),
        "min_precision": min_precision,
        "min_coverage": min_coverage,
        "min_area": min_area,
        "precision_median": sorted(precisions)[len(precisions) // 2] if precisions else 0.0,
        "precision_min": min(precisions) if precisions else 0.0,
        "coverage_median": sorted(coverages)[len(coverages) // 2] if coverages else 0.0,
        "coverage_min": min(coverages) if coverages else 0.0,
        "failed_frame_count": len(failed),
        "failed_frames": failed[:120],
        "pass": len(failed) == 0,
        "csv": str(out_csv),
        "note": "Uses renderer-written uncompressed final overlay mask metrics, avoiding video codec noise.",
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate final with-human overlay against tracked visible line segments.")
    ap.add_argument("--overlay-video", type=Path)
    ap.add_argument("--frames-dir", type=Path)
    ap.add_argument("--tracks-csv", type=Path)
    ap.add_argument("--quality-csv", type=Path)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--min-precision", type=float, default=0.92)
    ap.add_argument("--min-coverage", type=float, default=0.12)
    ap.add_argument("--min-area", type=int, default=80)
    ap.add_argument("--pad-px", type=int, default=18)
    args = ap.parse_args()
    if args.quality_csv:
        report = evaluate_quality_csv(
            quality_csv=args.quality_csv,
            out_json=args.out_json,
            out_csv=args.out_csv,
            min_precision=args.min_precision,
            min_coverage=args.min_coverage,
            min_area=args.min_area,
        )
    else:
        if not args.overlay_video or not args.frames_dir or not args.tracks_csv:
            raise SystemExit("--overlay-video, --frames-dir, and --tracks-csv are required unless --quality-csv is used")
        report = evaluate(
            overlay_video=args.overlay_video,
            frames_dir=args.frames_dir,
            tracks_csv=args.tracks_csv,
            out_json=args.out_json,
            out_csv=args.out_csv,
            min_precision=args.min_precision,
            min_coverage=args.min_coverage,
            min_area=args.min_area,
            pad_px=args.pad_px,
        )
    print(json.dumps(report, indent=2)[:4000])
    if not report["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
