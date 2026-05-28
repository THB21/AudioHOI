#!/usr/bin/env python3
"""Build a generic object observation layer from segmentation and tracking."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mask_features(mask_path: Path) -> dict[str, float | int]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Could not read mask {mask_path}")
    binary = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {
            "mask_available": 0,
            "mask_center_x": math.nan,
            "mask_center_y": math.nan,
            "mask_area_px": 0.0,
            "bbox_x1": math.nan,
            "bbox_y1": math.nan,
            "bbox_x2": math.nan,
            "bbox_y2": math.nan,
            "bbox_w_px": math.nan,
            "bbox_h_px": math.nan,
            "enclosing_radius_px": math.nan,
            "major_axis_px": math.nan,
            "minor_axis_px": math.nan,
            "aspect_ratio": math.nan,
            "circularity": 0.0,
        }

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    moments = cv2.moments(contour)
    if abs(moments["m00"]) < 1e-8:
        cx = cy = math.nan
    else:
        cx = float(moments["m10"] / moments["m00"])
        cy = float(moments["m01"] / moments["m00"])

    x, y, w, h = cv2.boundingRect(contour)
    (_circle_center, enclosing_radius) = cv2.minEnclosingCircle(contour)

    if len(contour) >= 5:
        (_ellipse_center, axes, _angle) = cv2.fitEllipse(contour)
        major_axis = float(max(axes))
        minor_axis = float(min(axes))
    else:
        major_axis = float(max(w, h))
        minor_axis = float(min(w, h))

    aspect_ratio = float(w / max(h, 1))
    circularity = float(4.0 * math.pi * area / max(perimeter * perimeter, 1e-8))

    return {
        "mask_available": 1,
        "mask_center_x": cx,
        "mask_center_y": cy,
        "mask_area_px": area,
        "bbox_x1": float(x),
        "bbox_y1": float(y),
        "bbox_x2": float(x + w),
        "bbox_y2": float(y + h),
        "bbox_w_px": float(w),
        "bbox_h_px": float(h),
        "enclosing_radius_px": float(enclosing_radius),
        "major_axis_px": major_axis,
        "minor_axis_px": minor_axis,
        "aspect_ratio": aspect_ratio,
        "circularity": circularity,
    }


def parse_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value is None or value == "":
        return math.nan
    return float(value)


def parse_track_points(row: dict[str, str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in ["center", "left", "right", "top", "bottom"]:
        out[f"{name}_x"] = parse_float(row, f"{name}_x")
        out[f"{name}_y"] = parse_float(row, f"{name}_y")
        out[f"{name}_visible"] = parse_float(row, f"{name}_visible")
    return out


def build_rows(
    frames: list[int],
    object_name: str,
    mask_dir: Path,
    center_by_frame: dict[int, dict[str, str]],
    points_by_frame: dict[int, dict[str, str]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for frame in frames:
        center_row = center_by_frame.get(frame, {})
        points_row = points_by_frame.get(frame, {})
        mask_path = mask_dir / f"{frame:05d}_mask.png"
        feats = mask_features(mask_path) if mask_path.exists() else {
            "mask_available": 0,
            "mask_center_x": math.nan,
            "mask_center_y": math.nan,
            "mask_area_px": 0.0,
            "bbox_x1": math.nan,
            "bbox_y1": math.nan,
            "bbox_x2": math.nan,
            "bbox_y2": math.nan,
            "bbox_w_px": math.nan,
            "bbox_h_px": math.nan,
            "enclosing_radius_px": math.nan,
            "major_axis_px": math.nan,
            "minor_axis_px": math.nan,
            "aspect_ratio": math.nan,
            "circularity": 0.0,
        }

        track_center_x = parse_float(center_row, "ball_center_x")
        track_center_y = parse_float(center_row, "ball_center_y")
        center_source = center_row.get("source", "")
        tracking_available = 0 if math.isnan(track_center_x) or math.isnan(track_center_y) else 1

        if tracking_available:
            fused_center_x = track_center_x
            fused_center_y = track_center_y
            fused_source = center_source or "tracking"
        else:
            fused_center_x = float(feats["mask_center_x"])
            fused_center_y = float(feats["mask_center_y"])
            fused_source = "mask_moments" if int(feats["mask_available"]) == 1 else ""

        points = parse_track_points(points_row) if points_row else {
            f"{name}_{suffix}": math.nan
            for name in ["center", "left", "right", "top", "bottom"]
            for suffix in ["x", "y", "visible"]
        }

        visible_count = 0
        for name in ["left", "right", "top", "bottom"]:
            vis = points[f"{name}_visible"]
            if not math.isnan(vis) and vis > 0.5:
                visible_count += 1
        track_geometry_conf = visible_count / 4.0
        mask_conf = min(1.0, max(0.0, float(feats["circularity"])))
        observation_conf = max(track_geometry_conf, mask_conf if int(feats["mask_available"]) == 1 else 0.0)

        row: dict[str, object] = {
            "frame": frame,
            "time": center_row.get("time") or points_row.get("time") or "",
            "object_name": object_name,
            "center_x": f"{fused_center_x:.3f}" if not math.isnan(fused_center_x) else "",
            "center_y": f"{fused_center_y:.3f}" if not math.isnan(fused_center_y) else "",
            "center_source": fused_source,
            "tracking_available": tracking_available,
            "track_center_x": f"{track_center_x:.3f}" if not math.isnan(track_center_x) else "",
            "track_center_y": f"{track_center_y:.3f}" if not math.isnan(track_center_y) else "",
            "track_center_source": center_source,
            "mask_available": int(feats["mask_available"]),
            "mask_center_x": f"{float(feats['mask_center_x']):.3f}" if not math.isnan(float(feats["mask_center_x"])) else "",
            "mask_center_y": f"{float(feats['mask_center_y']):.3f}" if not math.isnan(float(feats["mask_center_y"])) else "",
            "mask_area_px": f"{float(feats['mask_area_px']):.1f}",
            "bbox_x1": f"{float(feats['bbox_x1']):.3f}" if not math.isnan(float(feats["bbox_x1"])) else "",
            "bbox_y1": f"{float(feats['bbox_y1']):.3f}" if not math.isnan(float(feats["bbox_y1"])) else "",
            "bbox_x2": f"{float(feats['bbox_x2']):.3f}" if not math.isnan(float(feats["bbox_x2"])) else "",
            "bbox_y2": f"{float(feats['bbox_y2']):.3f}" if not math.isnan(float(feats["bbox_y2"])) else "",
            "bbox_w_px": f"{float(feats['bbox_w_px']):.3f}" if not math.isnan(float(feats["bbox_w_px"])) else "",
            "bbox_h_px": f"{float(feats['bbox_h_px']):.3f}" if not math.isnan(float(feats["bbox_h_px"])) else "",
            "enclosing_radius_px": f"{float(feats['enclosing_radius_px']):.3f}" if not math.isnan(float(feats["enclosing_radius_px"])) else "",
            "major_axis_px": f"{float(feats['major_axis_px']):.3f}" if not math.isnan(float(feats["major_axis_px"])) else "",
            "minor_axis_px": f"{float(feats['minor_axis_px']):.3f}" if not math.isnan(float(feats["minor_axis_px"])) else "",
            "aspect_ratio": f"{float(feats['aspect_ratio']):.6f}" if not math.isnan(float(feats["aspect_ratio"])) else "",
            "circularity": f"{float(feats['circularity']):.6f}",
            "track_geometry_conf": f"{track_geometry_conf:.6f}",
            "mask_conf": f"{mask_conf:.6f}",
            "observation_conf": f"{observation_conf:.6f}",
        }

        for name in ["center", "left", "right", "top", "bottom"]:
            row[f"{name}_x"] = f"{points[f'{name}_x']:.3f}" if not math.isnan(points[f"{name}_x"]) else ""
            row[f"{name}_y"] = f"{points[f'{name}_y']:.3f}" if not math.isnan(points[f"{name}_y"]) else ""
            row[f"{name}_visible"] = f"{points[f'{name}_visible']:.6f}" if not math.isnan(points[f"{name}_visible"]) else ""

        rows.append(row)
    return rows


def write_summary(path: Path, rows: list[dict[str, object]], object_name: str) -> None:
    mask_available = sum(int(r["mask_available"]) for r in rows)
    tracking_available = sum(int(r["tracking_available"]) for r in rows)
    observation_conf = np.asarray([float(r["observation_conf"]) for r in rows], dtype=np.float64)
    circularity = np.asarray([float(r["circularity"]) for r in rows], dtype=np.float64)
    with path.open("w") as f:
        f.write("Generic SAM2-based object observation layer.\n")
        f.write(f"object_name: {object_name}\n")
        f.write(f"num_frames: {len(rows)}\n")
        f.write(f"mask_available_frames: {mask_available}\n")
        f.write(f"tracking_available_frames: {tracking_available}\n")
        f.write(f"mean_observation_conf: {float(np.mean(observation_conf)):.6f}\n")
        f.write(f"mean_circularity: {float(np.mean(circularity)):.6f}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build generic object observations from SAM2 masks and tracking outputs.")
    parser.add_argument("--sample-dir", type=Path, default=Path("samples/basketball_01"))
    parser.add_argument("--object-name", type=str, default="basketball")
    parser.add_argument("--mask-dir", type=Path, default=None)
    parser.add_argument("--tracking-center-csv", type=Path, default=None)
    parser.add_argument("--tracking-points-csv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    sample_dir = args.sample_dir
    results_dir = sample_dir / "results"
    mask_dir = args.mask_dir or (results_dir / "segmentation" / "masks")
    center_csv = args.tracking_center_csv or (results_dir / "tracking" / "cotracker_center_trajectory.csv")
    points_csv = args.tracking_points_csv or (results_dir / "tracking" / "cotracker_points.csv")
    out_dir = args.out_dir or (results_dir / "object_observations")
    out_dir.mkdir(parents=True, exist_ok=True)

    center_rows = read_csv_rows(center_csv)
    points_rows = read_csv_rows(points_csv)
    center_by_frame = {int(r["frame"]): r for r in center_rows}
    points_by_frame = {int(r["frame"]): r for r in points_rows}

    frames = sorted(set(center_by_frame) | set(points_by_frame))
    if not frames:
        mask_paths = sorted(mask_dir.glob("*_mask.png"))
        frames = [int(p.stem.split("_")[0]) for p in mask_paths]
    if not frames:
        raise RuntimeError("No observation inputs found")

    rows = build_rows(frames, args.object_name, mask_dir, center_by_frame, points_by_frame)

    out_csv = out_dir / "object_observations.csv"
    summary_txt = out_dir / "object_observations_summary.txt"
    fieldnames = list(rows[0].keys())
    write_csv(out_csv, rows, fieldnames)
    write_summary(summary_txt, rows, args.object_name)

    print(f"object_observations_csv: {out_csv}")
    print(f"object_observations_summary: {summary_txt}")


if __name__ == "__main__":
    main()
