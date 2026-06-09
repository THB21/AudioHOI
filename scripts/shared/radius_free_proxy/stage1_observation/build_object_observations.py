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
            "lowest_visible_x": math.nan,
            "lowest_visible_y": math.nan,
            "lowest_visible_x1": math.nan,
            "lowest_visible_x2": math.nan,
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
    ys, xs = np.where(binary > 0)

    if len(contour) >= 5:
        (_ellipse_center, axes, _angle) = cv2.fitEllipse(contour)
        major_axis = float(max(axes))
        minor_axis = float(min(axes))
    else:
        major_axis = float(max(w, h))
        minor_axis = float(min(w, h))

    aspect_ratio = float(w / max(h, 1))
    circularity = float(4.0 * math.pi * area / max(perimeter * perimeter, 1e-8))

    lowest_y = int(ys.max())
    lowest_xs = xs[ys == lowest_y]
    lowest_x1 = float(lowest_xs.min())
    lowest_x2 = float(lowest_xs.max())
    lowest_x = 0.5 * (lowest_x1 + lowest_x2)

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
        "lowest_visible_x": lowest_x,
        "lowest_visible_y": float(lowest_y),
        "lowest_visible_x1": lowest_x1,
        "lowest_visible_x2": lowest_x2,
        "enclosing_radius_px": float(enclosing_radius),
        "major_axis_px": major_axis,
        "minor_axis_px": minor_axis,
        "aspect_ratio": aspect_ratio,
        "circularity": circularity,
    }


def empty_mug_body_features() -> dict[str, float | str | int]:
    return {
        "body_available": 0,
        "body_center_x": math.nan,
        "body_center_y": math.nan,
        "body_bbox_x1": math.nan,
        "body_bbox_y1": math.nan,
        "body_bbox_x2": math.nan,
        "body_bbox_y2": math.nan,
        "body_bbox_w_px": math.nan,
        "body_bbox_h_px": math.nan,
        "handle_side": "",
        "handle_observed_side": "",
        "handle_visible": 0,
        "handle_conf": 0.0,
        "handle_center_x": math.nan,
        "handle_center_y": math.nan,
        "handle_bbox_x1": math.nan,
        "handle_bbox_y1": math.nan,
        "handle_bbox_x2": math.nan,
        "handle_bbox_y2": math.nan,
        "handle_area_px": 0.0,
        "handle_extra_left_px": math.nan,
        "handle_extra_right_px": math.nan,
    }


def mug_body_features(mask_path: Path) -> dict[str, float | str | int]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Could not read mask {mask_path}")
    binary = mask > 0
    ys, xs = np.where(binary)
    if len(xs) == 0:
        return empty_mug_body_features()

    y1 = int(ys.min())
    y2 = int(ys.max())
    x1 = int(xs.min())
    x2 = int(xs.max())
    height = y2 - y1 + 1
    if height < 5:
        return empty_mug_body_features()

    top = y1 + int(round(0.20 * height))
    bottom = y1 + int(round(0.80 * height))
    row_lefts: list[int] = []
    row_rights: list[int] = []
    for y in range(top, min(bottom + 1, binary.shape[0])):
        cols = np.where(binary[y])[0]
        if len(cols) == 0:
            continue
        row_lefts.append(int(cols.min()))
        row_rights.append(int(cols.max()))

    if not row_lefts or not row_rights:
        return empty_mug_body_features()

    body_x1 = int(round(float(np.median(np.asarray(row_lefts, dtype=np.float32)))))
    body_x2 = int(round(float(np.median(np.asarray(row_rights, dtype=np.float32)))))
    body_x1 = max(body_x1, x1)
    body_x2 = min(body_x2, x2)
    if body_x2 <= body_x1:
        body_x1, body_x2 = x1, x2

    body_mask = binary.copy()
    body_mask[:, :body_x1] = False
    body_mask[:, body_x2 + 1 :] = False
    body_ys, body_xs = np.where(body_mask)
    if len(body_xs) == 0:
        return empty_mug_body_features()

    body_center_x = float(np.mean(body_xs))
    body_center_y = float(np.mean(body_ys))
    left_extra = float(max(0, body_x1 - x1))
    right_extra = float(max(0, x2 - body_x2))

    xx = np.indices(binary.shape)[1]
    left_region = binary & (xx < body_x1)
    right_region = binary & (xx > body_x2)
    left_area = float(np.count_nonzero(left_region))
    right_area = float(np.count_nonzero(right_region))
    total_area = float(max(np.count_nonzero(binary), 1))
    handle_side = "unknown"
    handle_region = np.zeros_like(binary, dtype=bool)
    handle_conf = 0.0
    if left_area > right_area * 1.25 and left_extra > 1.0:
        handle_side = "left"
        handle_region = left_region
        handle_conf = min(1.0, (left_area / total_area) * max(1.0, left_extra / 2.0) * 4.0)
    elif right_area > left_area * 1.25 and right_extra > 1.0:
        handle_side = "right"
        handle_region = right_region
        handle_conf = min(1.0, (right_area / total_area) * max(1.0, right_extra / 2.0) * 4.0)

    handle_ys, handle_xs = np.where(handle_region)
    handle_visible = int(handle_side != "unknown" and len(handle_xs) > 0 and handle_conf > 0.03)
    if handle_visible:
        handle_center_x = float(np.mean(handle_xs))
        handle_center_y = float(np.mean(handle_ys))
        handle_bbox_x1 = float(handle_xs.min())
        handle_bbox_y1 = float(handle_ys.min())
        handle_bbox_x2 = float(handle_xs.max())
        handle_bbox_y2 = float(handle_ys.max())
        handle_area = float(len(handle_xs))
    else:
        handle_center_x = handle_center_y = math.nan
        handle_bbox_x1 = handle_bbox_y1 = handle_bbox_x2 = handle_bbox_y2 = math.nan
        handle_area = 0.0

    return {
        "body_available": 1,
        "body_center_x": body_center_x,
        "body_center_y": body_center_y,
        "body_bbox_x1": float(body_x1),
        "body_bbox_y1": float(y1),
        "body_bbox_x2": float(body_x2),
        "body_bbox_y2": float(y2),
        "body_bbox_w_px": float(body_x2 - body_x1 + 1),
        "body_bbox_h_px": float(height),
        "handle_side": handle_side,
        "handle_observed_side": handle_side,
        "handle_visible": handle_visible,
        "handle_conf": float(handle_conf if handle_visible else 0.0),
        "handle_center_x": handle_center_x,
        "handle_center_y": handle_center_y,
        "handle_bbox_x1": handle_bbox_x1,
        "handle_bbox_y1": handle_bbox_y1,
        "handle_bbox_x2": handle_bbox_x2,
        "handle_bbox_y2": handle_bbox_y2,
        "handle_area_px": handle_area,
        "handle_extra_left_px": left_extra,
        "handle_extra_right_px": right_extra,
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
    is_mug = object_name.lower() == "mug"
    latent_handle_side = ""
    latent_handle_conf = 0.0
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
            "lowest_visible_x": math.nan,
            "lowest_visible_y": math.nan,
            "lowest_visible_x1": math.nan,
            "lowest_visible_x2": math.nan,
            "enclosing_radius_px": math.nan,
            "major_axis_px": math.nan,
            "minor_axis_px": math.nan,
            "aspect_ratio": math.nan,
            "circularity": 0.0,
        }
        mug_feats = mug_body_features(mask_path) if is_mug and mask_path.exists() else empty_mug_body_features()
        if is_mug:
            observed_side = str(mug_feats.get("handle_observed_side", "") or "")
            observed_conf = float(mug_feats.get("handle_conf", 0.0) or 0.0)
            if int(mug_feats.get("handle_visible", 0) or 0) == 1 and observed_side not in {"", "unknown"} and observed_conf > 0.03:
                latent_handle_side = observed_side
                latent_handle_conf = max(latent_handle_conf * 0.9, observed_conf)
            mug_feats["handle_side"] = latent_handle_side or observed_side
            if int(mug_feats.get("handle_visible", 0) or 0) == 0:
                mug_feats["handle_conf"] = 0.0

        track_center_x = parse_float(center_row, "ball_center_x")
        track_center_y = parse_float(center_row, "ball_center_y")
        center_source = center_row.get("source", "")
        tracking_available = 0 if math.isnan(track_center_x) or math.isnan(track_center_y) else 1

        mug_body_available = int(mug_feats["body_available"]) == 1
        if is_mug and mug_body_available:
            fused_center_x = float(mug_feats["body_center_x"])
            fused_center_y = float(mug_feats["body_center_y"])
            fused_source = "mug_body_mask"
        elif tracking_available:
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
            "center_x": f"{float(mug_feats['body_center_x']):.3f}" if is_mug and int(mug_feats["body_available"]) == 1 and not math.isnan(float(mug_feats["body_center_x"])) else (f"{fused_center_x:.3f}" if not math.isnan(fused_center_x) else ""),
            "center_y": f"{float(mug_feats['body_center_y']):.3f}" if is_mug and int(mug_feats["body_available"]) == 1 and not math.isnan(float(mug_feats["body_center_y"])) else (f"{fused_center_y:.3f}" if not math.isnan(fused_center_y) else ""),
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
            "lowest_visible_x": f"{float(feats['lowest_visible_x']):.3f}" if not math.isnan(float(feats["lowest_visible_x"])) else "",
            "lowest_visible_y": f"{float(feats['lowest_visible_y']):.3f}" if not math.isnan(float(feats["lowest_visible_y"])) else "",
            "lowest_visible_x1": f"{float(feats['lowest_visible_x1']):.3f}" if not math.isnan(float(feats["lowest_visible_x1"])) else "",
            "lowest_visible_x2": f"{float(feats['lowest_visible_x2']):.3f}" if not math.isnan(float(feats["lowest_visible_x2"])) else "",
            "enclosing_radius_px": f"{float(feats['enclosing_radius_px']):.3f}" if not math.isnan(float(feats["enclosing_radius_px"])) else "",
            "major_axis_px": f"{float(feats['major_axis_px']):.3f}" if not math.isnan(float(feats["major_axis_px"])) else "",
            "minor_axis_px": f"{float(feats['minor_axis_px']):.3f}" if not math.isnan(float(feats["minor_axis_px"])) else "",
            "aspect_ratio": f"{float(feats['aspect_ratio']):.6f}" if not math.isnan(float(feats["aspect_ratio"])) else "",
            "circularity": f"{float(feats['circularity']):.6f}",
            "body_available": int(mug_feats["body_available"]),
            "body_center_x": f"{float(mug_feats['body_center_x']):.3f}" if not math.isnan(float(mug_feats["body_center_x"])) else "",
            "body_center_y": f"{float(mug_feats['body_center_y']):.3f}" if not math.isnan(float(mug_feats["body_center_y"])) else "",
            "body_bbox_x1": f"{float(mug_feats['body_bbox_x1']):.3f}" if not math.isnan(float(mug_feats["body_bbox_x1"])) else "",
            "body_bbox_y1": f"{float(mug_feats['body_bbox_y1']):.3f}" if not math.isnan(float(mug_feats["body_bbox_y1"])) else "",
            "body_bbox_x2": f"{float(mug_feats['body_bbox_x2']):.3f}" if not math.isnan(float(mug_feats["body_bbox_x2"])) else "",
            "body_bbox_y2": f"{float(mug_feats['body_bbox_y2']):.3f}" if not math.isnan(float(mug_feats["body_bbox_y2"])) else "",
            "body_bbox_w_px": f"{float(mug_feats['body_bbox_w_px']):.3f}" if not math.isnan(float(mug_feats["body_bbox_w_px"])) else "",
            "body_bbox_h_px": f"{float(mug_feats['body_bbox_h_px']):.3f}" if not math.isnan(float(mug_feats["body_bbox_h_px"])) else "",
            "handle_side": str(mug_feats["handle_side"]),
            "handle_observed_side": str(mug_feats["handle_observed_side"]),
            "handle_visible": int(mug_feats["handle_visible"]),
            "handle_conf": f"{float(mug_feats['handle_conf']):.6f}",
            "handle_center_x": f"{float(mug_feats['handle_center_x']):.3f}" if not math.isnan(float(mug_feats["handle_center_x"])) else "",
            "handle_center_y": f"{float(mug_feats['handle_center_y']):.3f}" if not math.isnan(float(mug_feats["handle_center_y"])) else "",
            "handle_bbox_x1": f"{float(mug_feats['handle_bbox_x1']):.3f}" if not math.isnan(float(mug_feats["handle_bbox_x1"])) else "",
            "handle_bbox_y1": f"{float(mug_feats['handle_bbox_y1']):.3f}" if not math.isnan(float(mug_feats["handle_bbox_y1"])) else "",
            "handle_bbox_x2": f"{float(mug_feats['handle_bbox_x2']):.3f}" if not math.isnan(float(mug_feats["handle_bbox_x2"])) else "",
            "handle_bbox_y2": f"{float(mug_feats['handle_bbox_y2']):.3f}" if not math.isnan(float(mug_feats["handle_bbox_y2"])) else "",
            "handle_area_px": f"{float(mug_feats['handle_area_px']):.1f}",
            "handle_extra_left_px": f"{float(mug_feats['handle_extra_left_px']):.3f}" if not math.isnan(float(mug_feats["handle_extra_left_px"])) else "",
            "handle_extra_right_px": f"{float(mug_feats['handle_extra_right_px']):.3f}" if not math.isnan(float(mug_feats["handle_extra_right_px"])) else "",
            "track_geometry_conf": f"{track_geometry_conf:.6f}",
            "mask_conf": f"{mask_conf:.6f}",
            "observation_conf": f"{observation_conf:.6f}",
        }

        for name in ["center", "left", "right", "top", "bottom"]:
            if name == "center":
                row["center_visible"] = f"{points['center_visible']:.6f}" if not math.isnan(points["center_visible"]) else ""
                continue
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
