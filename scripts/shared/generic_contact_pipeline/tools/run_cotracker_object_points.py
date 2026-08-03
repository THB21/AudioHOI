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
import hashlib
import json
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


def sample_persistent_queries(
    mask: np.ndarray,
    *,
    object_family: str,
    grid_size: int,
) -> list[tuple[str, np.ndarray]]:
    """Create one stable query set for the complete tracking attempt."""

    if grid_size < 2:
        raise ValueError("persistent CoTracker grid size must be at least 2")
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise RuntimeError("Empty SAM2 mask")
    named_points = initial_points_from_mask(mask, object_family=object_family)
    existing = {name for name, _point in named_points}
    x_grid = np.linspace(float(xs.min()), float(xs.max()), grid_size)
    y_grid = np.linspace(float(ys.min()), float(ys.max()), grid_size)
    for row, y in enumerate(y_grid):
        for col, x in enumerate(x_grid):
            xi = int(np.clip(round(x), 0, mask.shape[1] - 1))
            yi = int(np.clip(round(y), 0, mask.shape[0] - 1))
            if mask[yi, xi] == 0:
                continue
            point_id = f"grid_{row:02d}_{col:02d}"
            if point_id not in existing:
                named_points.append((point_id, np.array([x, y], dtype=np.float64)))
    if len(named_points) < 8:
        raise RuntimeError(
            f"Persistent rigid tracking requires at least 8 mask-interior queries, got {len(named_points)}"
        )
    return named_points


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
    sequence_mode: str,
    grid_size: int,
    device: str,
) -> dict[str, object]:
    import torch

    results_dir = sample_dir / "results"
    tracking_dir = results_dir / "tracking"
    masks_dir = results_dir / "segmentation" / "masks"
    frames, sx, sy = read_frames(sample_dir / "frames", resize_width)
    tracker_hub_model = "cotracker3_online" if sequence_mode == "persistent_online" else "cotracker3_offline"
    cotracker = torch.hub.load(
        "facebookresearch/co-tracker",
        tracker_hub_model,
        trust_repo=True,
    ).to(device)
    cotracker.eval()

    if sequence_mode not in {"chunked_legacy", "persistent_offline", "persistent_online"}:
        raise ValueError(f"Unsupported CoTracker sequence mode: {sequence_mode}")

    track_by_frame: dict[int, list[tuple[str, np.ndarray]]] = {}
    visibility_by_frame: dict[int, np.ndarray] = {}
    query_frame_by_name: dict[str, int] = {}
    query_mask_path: Path | None = None
    with torch.inference_mode():
        if sequence_mode in {"persistent_offline", "persistent_online"}:
            query_mask_path = masks_dir / "00001_mask.png"
            query_mask = cv2.imread(str(query_mask_path), cv2.IMREAD_GRAYSCALE)
            if query_mask is None:
                raise RuntimeError(f"Could not read mask {query_mask_path}")
            named_points = sample_persistent_queries(
                query_mask,
                object_family=object_family,
                grid_size=grid_size,
            )
            names = [name for name, _point in named_points]
            query_frame_by_name.update({name: 1 for name in names})
            points = np.stack([point for _name, point in named_points]).astype(np.float32)
            scaled_points = points.copy()
            scaled_points[:, 0] *= sx
            scaled_points[:, 1] *= sy
            queries = torch.zeros((1, len(scaled_points), 3), dtype=torch.float32, device=device)
            queries[0, :, 1:] = torch.from_numpy(scaled_points).to(device)
            if sequence_mode == "persistent_offline":
                video = (
                    torch.from_numpy(frames)
                    .permute(0, 3, 1, 2)[None]
                    .float()
                    .to(device)
                )
                pred_tracks, pred_visibility = cotracker(video, queries=queries)
            else:
                window_frames: list[np.ndarray] = []
                is_first_step = True
                pred_tracks = pred_visibility = None

                def process_online_window(window: list[np.ndarray], first_step: bool):
                    video_chunk = (
                        torch.from_numpy(np.stack(window[-cotracker.step * 2 :]))
                        .permute(0, 3, 1, 2)[None]
                        .float()
                        .to(device)
                    )
                    return cotracker(
                        video_chunk,
                        is_first_step=first_step,
                        queries=queries if first_step else None,
                        grid_size=0,
                    )

                for frame_idx, frame in enumerate(frames):
                    if frame_idx % cotracker.step == 0 and frame_idx != 0:
                        pred_tracks, pred_visibility = process_online_window(
                            window_frames,
                            is_first_step,
                        )
                        is_first_step = False
                    window_frames.append(frame)
                final_window_start = -(frame_idx % cotracker.step) - cotracker.step - 1
                pred_tracks, pred_visibility = process_online_window(
                    window_frames[final_window_start:],
                    is_first_step,
                )
                if pred_tracks is None or pred_visibility is None:
                    raise RuntimeError("Online CoTracker produced no accumulated tracks")
            tracks_chunk = pred_tracks[0].detach().cpu().numpy()
            visibility_chunk = pred_visibility[0].detach().cpu().numpy()
            tracks_chunk[:, :, 0] /= sx
            tracks_chunk[:, :, 1] /= sy
            for offset in range(tracks_chunk.shape[0]):
                track_by_frame[offset] = [
                    (name, tracks_chunk[offset, i]) for i, name in enumerate(names)
                ]
                visibility_by_frame[offset] = visibility_chunk[offset]
        else:
            for start in range(0, frames.shape[0], chunk_len):
                end = min(start + chunk_len, frames.shape[0])
                mask_path = masks_dir / f"{start + 1:05d}_mask.png"
                named_points = read_mask_points(mask_path, object_family=object_family)
                names = [name for name, _point in named_points]
                query_frame_by_name.update({f"{start + 1}:{name}": start + 1 for name in names})
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
                        track_by_frame[frame_idx] = [
                            (name, tracks_chunk[offset, i]) for i, name in enumerate(names)
                        ]
                        visibility_by_frame[frame_idx] = visibility_chunk[offset]

    if not track_by_frame:
        raise RuntimeError("CoTracker produced no tracks")

    center_rows: list[dict[str, object]] = []
    point_rows: list[dict[str, object]] = []
    mesh_rows: list[dict[str, object]] = []
    rigid_track_rows: list[dict[str, object]] = []
    query_mask_hash = ""
    if query_mask_path is not None:
        query_mask_hash = hashlib.sha256(query_mask_path.read_bytes()).hexdigest()
    attempt_id = f"cotracker-{sequence_mode}-{query_mask_hash[:12] or 'legacy'}"
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
            visible = float(visibility[point_idx])
            if sequence_mode in {"persistent_offline", "persistent_online"}:
                rigid_track_rows.append(
                    {
                        "frame": frame_1based,
                        "time": time,
                        "track_id": name,
                        "query_frame": query_frame_by_name[name],
                        "x": f"{float(xy[0]):.3f}",
                        "y": f"{float(xy[1]):.3f}",
                        "visible": f"{visible:.6f}",
                        "confidence": f"{visible:.6f}",
                        "semantic_feature_id": "",
                        "source": "cotracker3_persistent_sam2_queries",
                        "attempt_id": attempt_id,
                    }
                )
            mesh_rows.append(
                {
                    "frame": frame_1based,
                    "time": time,
                    "point_id": name,
                    "point_type": _point_type(name),
                    "x": f"{float(xy[0]):.3f}",
                    "y": f"{float(xy[1]):.3f}",
                    "visible": f"{visible:.6f}",
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
    rigid_tracks_path = tracking_dir / "rigid_point_tracks.csv"
    rigid_manifest_path = tracking_dir / "rigid_point_tracks_manifest.json"
    if sequence_mode in {"persistent_offline", "persistent_online"}:
        write_csv(
            rigid_tracks_path,
            rigid_track_rows,
            [
                "frame",
                "time",
                "track_id",
                "query_frame",
                "x",
                "y",
                "visible",
                "confidence",
                "semantic_feature_id",
                "source",
                "attempt_id",
            ],
        )
        manifest = {
            "schema_version": 1,
            "attempt_id": attempt_id,
            "tracker": f"facebookresearch/co-tracker:{tracker_hub_model}",
            "sequence_mode": sequence_mode,
            "query_policy": "sam2_mask_interior_grid_plus_legacy_anchors",
            "query_frame": 1,
            "query_mask": str(query_mask_path),
            "query_mask_sha256": query_mask_hash,
            "query_count": len(track_by_frame[0]),
            "frame_count": len(track_by_frame),
            "resize_scale_xy": [sx, sy],
            "grid_size": grid_size,
            "reinitialization_frames": [],
            "local_3d_coordinates_assigned": False,
        }
        rigid_manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "frames": len(center_rows),
        "points_per_frame": len(track_by_frame[min(track_by_frame)]),
        "center_trajectory": str(tracking_dir / "object_center_trajectory.csv"),
        "object_points": str(tracking_dir / "object_points.csv"),
        "object_mesh_tracks": str(tracking_dir / "object_mesh_tracks_test.csv"),
        "sequence_mode": sequence_mode,
        "rigid_point_tracks": str(rigid_tracks_path) if rigid_tracks_path.is_file() else None,
        "rigid_point_tracks_manifest": str(rigid_manifest_path) if rigid_manifest_path.is_file() else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-config", type=Path, default=None)
    parser.add_argument("--sample-dir", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--object-family", default=None)
    parser.add_argument("--resize-width", type=int, default=256)
    parser.add_argument("--chunk-len", type=int, default=32)
    parser.add_argument(
        "--sequence-mode",
        choices=("chunked_legacy", "persistent_offline", "persistent_online"),
        default=None,
    )
    parser.add_argument("--grid-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    import json
    import torch

    case_config = read_case_config(args.case_config)
    sample_dir = args.sample_dir or Path(str(case_config.get("sample_dir", "")))
    if not str(sample_dir):
        raise ValueError("--sample-dir or case_config.sample_dir is required")
    object_family = args.object_family or str(case_config.get("object_family", "generic_object"))
    preprocess = case_config.get("preprocess", {})
    preprocess = dict(preprocess) if isinstance(preprocess, dict) else {}
    sequence_mode = args.sequence_mode or str(preprocess.get("tracker_sequence_mode", "chunked_legacy"))
    grid_size = args.grid_size or int(preprocess.get("tracker_grid_size", 12))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    summary = run_cotracker(
        sample_dir=sample_dir,
        fps=args.fps,
        object_family=object_family,
        resize_width=args.resize_width,
        chunk_len=args.chunk_len,
        sequence_mode=sequence_mode,
        grid_size=grid_size,
        device=device,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
