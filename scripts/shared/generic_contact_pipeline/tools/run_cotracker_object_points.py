#!/usr/bin/env python3
"""Track object query points with CoTracker from SAM2 masks.

This keeps the original basketball preprocessing contract but generalizes the
query initialization. For line-like rigid objects, the query points are the
SAM2-mask major-axis endpoints plus center, which feed existing line-object
correspondence through `object_mesh_tracks_test.csv`.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


BALL_POINT_NAMES = ["center", "left", "right", "top", "bottom"]


def read_case_config(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    import yaml

    with path.open() as f:
        return yaml.safe_load(f) or {}


def read_frames(frames_dir: Path, resize_width: int | None) -> tuple[np.ndarray, float, float]:
    frame_paths = sorted(frames_dir.glob("*.png"))
    if not frame_paths:
        frame_paths = sorted(frames_dir.glob("*.jpg"))
    if not frame_paths:
        raise RuntimeError(f"No frames found in {frames_dir}")

    frames = []
    sx = sy = 1.0
    for frame_path in frame_paths:
        bgr = cv2.imread(str(frame_path))
        if bgr is None:
            raise RuntimeError(f"Could not read {frame_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if resize_width and rgb.shape[1] > resize_width:
            scale = resize_width / rgb.shape[1]
            new_size = (resize_width, int(round(rgb.shape[0] * scale)))
            rgb = cv2.resize(rgb, new_size, interpolation=cv2.INTER_AREA)
            sx = new_size[0] / bgr.shape[1]
            sy = new_size[1] / bgr.shape[0]
        frames.append(rgb)
    return np.stack(frames, axis=0), sx, sy


def _major_axis_points(xs: np.ndarray, ys: np.ndarray) -> list[tuple[str, np.ndarray]]:
    pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    center = pts.mean(axis=0)
    centered = pts - center
    if len(pts) < 2:
        return [("tip_a", center), ("tip_b", center), ("center", center)]
    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    if axis[0] < 0:
        axis = -axis
    proj = centered @ axis
    lo = float(np.percentile(proj, 1.0))
    hi = float(np.percentile(proj, 99.0))
    tip_a = center + axis * lo
    tip_b = center + axis * hi
    return [("tip_a", tip_a), ("tip_b", tip_b), ("center", center)]


def initial_points_from_mask(mask: np.ndarray, *, object_family: str) -> list[tuple[str, np.ndarray]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise RuntimeError("Empty SAM2 mask")
    if object_family in {"rigid_staff", "line_object", "elongated_rigid", "staff", "stick"}:
        return _major_axis_points(xs, ys)

    center_x = float(xs.mean())
    center_y = float(ys.mean())
    points = [
        ("center", np.array([center_x, center_y], dtype=np.float64)),
        ("left", np.array([float(xs.min()), center_y], dtype=np.float64)),
        ("right", np.array([float(xs.max()), center_y], dtype=np.float64)),
        ("top", np.array([center_x, float(ys.min())], dtype=np.float64)),
        ("bottom", np.array([center_x, float(ys.max())], dtype=np.float64)),
    ]
    return points


def read_mask_points(mask_path: Path, *, object_family: str) -> list[tuple[str, np.ndarray]]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Could not read mask {mask_path}")
    return initial_points_from_mask(mask, object_family=object_family)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _point_type(name: str) -> str:
    return "boundary" if name.startswith("tip") or name in {"left", "right", "top", "bottom"} else "center"


def _local_x(name: str) -> float:
    if name in {"tip_a", "left"}:
        return -0.5
    if name in {"tip_b", "right"}:
        return 0.5
    return 0.0


def run_cotracker(
    *,
    sample_dir: Path,
    fps: float,
    object_family: str,
    resize_width: int | None,
    chunk_len: int,
    device: str,
) -> dict[str, object]:
    import torch

    results_dir = sample_dir / "results"
    tracking_dir = results_dir / "tracking"
    masks_dir = results_dir / "segmentation" / "masks"
    frames, sx, sy = read_frames(sample_dir / "frames", resize_width)
    cotracker = torch.hub.load(
        "facebookresearch/co-tracker",
        "cotracker3_offline",
        trust_repo=True,
    ).to(device)
    cotracker.eval()

    track_by_frame: dict[int, list[tuple[str, np.ndarray]]] = {}
    visibility_by_frame: dict[int, np.ndarray] = {}
    with torch.inference_mode():
        for start in range(0, frames.shape[0], chunk_len):
            end = min(start + chunk_len, frames.shape[0])
            mask_path = masks_dir / f"{start + 1:05d}_mask.png"
            named_points = read_mask_points(mask_path, object_family=object_family)
            names = [name for name, _point in named_points]
            points = np.stack([point for _name, point in named_points]).astype(np.float32)
            scaled_points = points.copy()
            scaled_points[:, 0] *= sx
            scaled_points[:, 1] *= sy

            video = (
                torch.from_numpy(frames[start:end])
                .permute(0, 3, 1, 2)[None]
                .float()
                .to(device)
            )
            queries = torch.zeros((1, len(scaled_points), 3), dtype=torch.float32, device=device)
            queries[0, :, 1:] = torch.from_numpy(scaled_points).to(device)
            pred_tracks, pred_visibility = cotracker(video, queries=queries)
            tracks_chunk = pred_tracks[0].detach().cpu().numpy()
            visibility_chunk = pred_visibility[0].detach().cpu().numpy()
            tracks_chunk[:, :, 0] /= sx
            tracks_chunk[:, :, 1] /= sy
            for offset in range(tracks_chunk.shape[0]):
                frame_idx = start + offset
                if frame_idx < frames.shape[0]:
                    track_by_frame[frame_idx] = [(name, tracks_chunk[offset, i]) for i, name in enumerate(names)]
                    visibility_by_frame[frame_idx] = visibility_chunk[offset]

    if not track_by_frame:
        raise RuntimeError("CoTracker produced no tracks")

    center_rows: list[dict[str, object]] = []
    point_rows: list[dict[str, object]] = []
    mesh_rows: list[dict[str, object]] = []
    for frame_idx in sorted(track_by_frame):
        frame_1based = frame_idx + 1
        time = f"{frame_idx / fps:.6f}"
        named_tracks = track_by_frame[frame_idx]
        visibility = visibility_by_frame[frame_idx]
        by_name = {name: xy for name, xy in named_tracks}
        center = by_name.get("center")
        if center is None:
            center = np.stack([xy for _name, xy in named_tracks]).mean(axis=0)
        center_rows.append(
            {
                "frame": frame_1based,
                "time": time,
                "center_x": f"{float(center[0]):.3f}",
                "center_y": f"{float(center[1]):.3f}",
                "source": "cotracker_sam2_mask_points",
            }
        )

        row: dict[str, object] = {"frame": frame_1based, "time": time}
        if object_family in {"rigid_staff", "line_object", "elongated_rigid", "staff", "stick"}:
            bottom_y = max(float(xy[1]) for _name, xy in named_tracks)
            row["bottom_y"] = f"{bottom_y:.3f}"
        else:
            for point_idx, name in enumerate(BALL_POINT_NAMES):
                if name not in by_name:
                    continue
                xy = by_name[name]
                row[f"{name}_x"] = f"{float(xy[0]):.3f}"
                row[f"{name}_y"] = f"{float(xy[1]):.3f}"
                row[f"{name}_visible"] = f"{float(visibility[point_idx]):.6f}"
        point_rows.append(row)

        for point_idx, (name, xy) in enumerate(named_tracks):
            mesh_rows.append(
                {
                    "frame": frame_1based,
                    "time": time,
                    "point_id": name,
                    "point_type": _point_type(name),
                    "x": f"{float(xy[0]):.3f}",
                    "y": f"{float(xy[1]):.3f}",
                    "visible": f"{float(visibility[point_idx]):.6f}",
                    "local_x": f"{_local_x(name):.6f}",
                    "local_y": "0.000000",
                    "local_z": "0.000000",
                    "source": "cotracker_sam2_mask_points",
                }
            )

    write_csv(tracking_dir / "object_center_trajectory.csv", center_rows, ["frame", "time", "center_x", "center_y", "source"])
    if object_family in {"rigid_staff", "line_object", "elongated_rigid", "staff", "stick"}:
        point_fields = ["frame", "time", "bottom_y"]
    else:
        point_fields = ["frame", "time"]
        for name in BALL_POINT_NAMES:
            point_fields.extend([f"{name}_x", f"{name}_y", f"{name}_visible"])
    write_csv(tracking_dir / "object_points.csv", point_rows, point_fields)
    write_csv(
        tracking_dir / "object_mesh_tracks_test.csv",
        mesh_rows,
        ["frame", "time", "point_id", "point_type", "x", "y", "visible", "local_x", "local_y", "local_z", "source"],
    )
    return {
        "frames": len(center_rows),
        "points_per_frame": len(track_by_frame[min(track_by_frame)]),
        "center_trajectory": str(tracking_dir / "object_center_trajectory.csv"),
        "object_points": str(tracking_dir / "object_points.csv"),
        "object_mesh_tracks": str(tracking_dir / "object_mesh_tracks_test.csv"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-config", type=Path, default=None)
    parser.add_argument("--sample-dir", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--object-family", default=None)
    parser.add_argument("--resize-width", type=int, default=256)
    parser.add_argument("--chunk-len", type=int, default=32)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    import json
    import torch

    case_config = read_case_config(args.case_config)
    sample_dir = args.sample_dir or Path(str(case_config.get("sample_dir", "")))
    if not str(sample_dir):
        raise ValueError("--sample-dir or case_config.sample_dir is required")
    object_family = args.object_family or str(case_config.get("object_family", "generic_object"))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    summary = run_cotracker(
        sample_dir=sample_dir,
        fps=args.fps,
        object_family=object_family,
        resize_width=args.resize_width,
        chunk_len=args.chunk_len,
        device=device,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
