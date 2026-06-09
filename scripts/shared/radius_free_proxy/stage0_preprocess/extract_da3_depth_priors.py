#!/usr/bin/env python3
"""Extract floor and ball depth priors from registered DA3 depth maps."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def read_tracking(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("ball_center_x") or not row.get("ball_center_y"):
                continue
            rows.append(
                {
                    "frame": int(row["frame"]),
                    "time": float(row["time"]),
                    "u": float(row["ball_center_x"]),
                    "v": float(row["ball_center_y"]),
                    "r": float(row.get("radius", 0.0) or 0.0),
                }
            )
    if not rows:
        raise RuntimeError(f"No valid tracking rows in {path}")
    return rows


def read_index(path: Path) -> dict[int, Path]:
    depth_dir = path.parent
    frame_to_path: dict[int, Path] = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_to_path[int(row["frame"])] = depth_dir / row["file"]
    if not frame_to_path:
        raise RuntimeError(f"No DA3 entries in {path}")
    return frame_to_path


def load_depth(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
    elif path.suffix.lower() == ".npz":
        data = np.load(path)
        key = next(iter(data.files))
        arr = data[key]
    else:
        arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise RuntimeError(f"Could not load depth map: {path}")
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


def get_video_hw(video_path: Path) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for size query: {video_path}")
    width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video size from {video_path}: {(width, height)}")
    return width, height


def map_to_depth_uv(u: float, v: float, src_hw: tuple[int, int], depth_hw: tuple[int, int]) -> tuple[float, float]:
    src_w, src_h = src_hw
    depth_h, depth_w = depth_hw
    su = depth_w / float(src_w)
    sv = depth_h / float(src_h)
    return u * su, v * sv


def robust_patch_median(depth: np.ndarray, u: float, v: float, radius: float) -> float:
    h, w = depth.shape[:2]
    patch_r = max(2, int(round(max(radius * 0.35, 3.0))))
    cx = int(round(u))
    cy = int(round(v))
    x0 = max(0, cx - patch_r)
    x1 = min(w, cx + patch_r + 1)
    y0 = max(0, cy - patch_r)
    y1 = min(h, cy + patch_r + 1)
    patch = depth[y0:y1, x0:x1]
    vals = patch[np.isfinite(patch)]
    vals = vals[vals > 0]
    if vals.size == 0:
        return float("nan")
    return float(np.median(vals))


def estimate_floor_proxy(depth: np.ndarray, u: float) -> tuple[float, float]:
    h, w = depth.shape[:2]
    cx = int(round(u))
    col_half = max(6, int(round(w * 0.02)))
    x0 = max(0, cx - col_half)
    x1 = min(w, cx + col_half + 1)
    y0 = int(round(h * 0.78))
    region = depth[y0:h, x0:x1]
    vals = region[np.isfinite(region)]
    vals = vals[vals > 0]
    floor_depth = float(np.median(vals)) if vals.size else float("nan")
    return floor_depth, float(y0)


def moving_average_nan(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    out = np.empty_like(values, dtype=np.float32)
    n = len(values)
    half = window // 2
    for i in range(n):
        a = max(0, i - half)
        b = min(n, i + half + 1)
        chunk = values[a:b]
        chunk = chunk[np.isfinite(chunk)]
        out[i] = float(np.mean(chunk)) if chunk.size else np.nan
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract DA3 ball/floor priors.")
    parser.add_argument("--sample-dir", type=Path, default=Path("samples/basketball_01"))
    parser.add_argument("--smooth-window", type=int, default=7)
    args = parser.parse_args()

    sample_dir = args.sample_dir
    results_dir = sample_dir / "results"
    da3_dir = results_dir / "da3"
    scene_depth_dir = da3_dir / "scene_depth"
    index_csv = scene_depth_dir / "index.csv"
    if not index_csv.exists():
        raise RuntimeError(f"Missing DA3 index: {index_csv}")

    frame_to_depth = read_index(index_csv)
    tracking_rows = read_tracking(results_dir / "tracking" / "ball_trajectory.csv")
    video_hw = get_video_hw(sample_dir / "video.mp4")

    priors_dir = da3_dir / "priors"
    priors_dir.mkdir(parents=True, exist_ok=True)
    ball_csv = priors_dir / "ball_depth_prior.csv"
    floor_csv = priors_dir / "floor_prior.csv"
    meta_json = priors_dir / "meta.json"

    ball_rows: list[dict[str, object]] = []
    floor_rows: list[dict[str, object]] = []
    raw_depths: list[float] = []
    frame_keys: list[int] = []

    per_frame_data: list[tuple[dict[str, float], float, float, float]] = []
    for row in tracking_rows:
        frame = int(row["frame"])
        if frame not in frame_to_depth:
            continue
        depth = load_depth(frame_to_depth[frame])
        u_depth, v_depth = map_to_depth_uv(row["u"], row["v"], video_hw, depth.shape[:2])
        scale_r = depth.shape[1] / float(video_hw[0])
        depth_raw = robust_patch_median(depth, u_depth, v_depth, row["r"] * scale_r)
        floor_depth, floor_row = estimate_floor_proxy(depth, u_depth)
        per_frame_data.append((row, depth_raw, floor_depth, floor_row))
        raw_depths.append(depth_raw)
        frame_keys.append(frame)

    if not per_frame_data:
        raise RuntimeError("No overlapping DA3 depth frames and ball tracking rows")

    depth_raw_arr = np.asarray(raw_depths, dtype=np.float32)
    depth_smooth_arr = moving_average_nan(depth_raw_arr, args.smooth_window)

    for idx, (row, depth_raw, floor_depth, floor_row) in enumerate(per_frame_data):
        ball_rows.append(
            {
                "frame": int(row["frame"]),
                "time": f"{row['time']:.6f}",
                "u": f"{row['u']:.3f}",
                "v": f"{row['v']:.3f}",
                "radius_px": f"{row['r']:.3f}",
                "da3_depth_raw": f"{depth_raw:.6f}" if np.isfinite(depth_raw) else "",
                "da3_depth_smooth": f"{depth_smooth_arr[idx]:.6f}" if np.isfinite(depth_smooth_arr[idx]) else "",
            }
        )
        floor_rows.append(
            {
                "frame": int(row["frame"]),
                "time": f"{row['time']:.6f}",
                "ball_u": f"{row['u']:.3f}",
                "floor_depth_proxy": f"{floor_depth:.6f}" if np.isfinite(floor_depth) else "",
                "floor_row_proxy": f"{floor_row:.3f}",
            }
        )

    with ball_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "time", "u", "v", "radius_px", "da3_depth_raw", "da3_depth_smooth"])
        writer.writeheader()
        writer.writerows(ball_rows)

    with floor_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "time", "ball_u", "floor_depth_proxy", "floor_row_proxy"])
        writer.writeheader()
        writer.writerows(floor_rows)

    meta = {
        "smooth_window": int(args.smooth_window),
        "num_prior_frames": len(ball_rows),
        "scene_depth_dir": str(scene_depth_dir),
        "video_hw": list(video_hw),
    }
    meta_json.write_text(json.dumps(meta, indent=2))

    print(f"ball_depth_prior_csv: {ball_csv}")
    print(f"floor_prior_csv: {floor_csv}")
    print(f"priors_meta_json: {meta_json}")
    print(f"num_prior_frames: {len(ball_rows)}")


if __name__ == "__main__":
    main()
