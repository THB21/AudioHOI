#!/usr/bin/env python3
"""Track basketball points with CoTracker3."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch


POINT_NAMES = ["center", "left", "right", "top", "bottom"]


def read_frames(frames_dir: Path, resize_width: int | None) -> tuple[np.ndarray, float, float]:
    frame_paths = sorted(frames_dir.glob("*.png"))
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


def initial_points_from_mask(mask_path: Path) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Could not read mask {mask_path}")
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise RuntimeError(f"Empty mask {mask_path}")
    center_x = float(xs.mean())
    center_y = float(ys.mean())
    left = float(xs.min()), center_y
    right = float(xs.max()), center_y
    top = center_x, float(ys.min())
    bottom = center_x, float(ys.max())
    center = center_x, center_y
    return np.array([center, left, right, top, bottom], dtype=np.float32)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-dir", type=Path, default=Path("samples/basketball_01"))
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--resize-width", type=int, default=256)
    parser.add_argument("--chunk-len", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    sample_dir = args.sample_dir
    results_dir = sample_dir / "results"
    masks_dir = results_dir / "masks"

    frames, sx, sy = read_frames(sample_dir / "frames", args.resize_width)
    cotracker = torch.hub.load(
        "facebookresearch/co-tracker",
        "cotracker3_offline",
        trust_repo=True,
    ).to(args.device)
    cotracker.eval()

    track_by_frame: dict[int, np.ndarray] = {}
    visibility_by_frame: dict[int, np.ndarray] = {}
    with torch.inference_mode():
        # Track the video in short chunks instead of sending all 192 frames to
        # CoTracker at once. Full-video offline tracking exceeded 10 GB GPU
        # memory on the RTX 3080, while 32-frame chunks fit comfortably.
        for start in range(0, frames.shape[0], args.chunk_len):
            end = min(start + args.chunk_len, frames.shape[0])

            # Re-initialize the five basketball query points from the SAM2 mask
            # at the first frame of this chunk: center, left, right, top, bottom.
            # This keeps each chunk anchored to the current ball location.
            mask_path = masks_dir / f"{start + 1:05d}_mask.png"
            points = initial_points_from_mask(mask_path)
            scaled_points = points.copy()
            scaled_points[:, 0] *= sx
            scaled_points[:, 1] *= sy

            video = (
                torch.from_numpy(frames[start:end])
                .permute(0, 3, 1, 2)[None]
                .float()
                .to(args.device)
            )
            queries = torch.zeros((1, len(scaled_points), 3), dtype=torch.float32, device=args.device)
            queries[0, :, 1:] = torch.from_numpy(scaled_points).to(args.device)
            pred_tracks, pred_visibility = cotracker(video, queries=queries)
            tracks_chunk = pred_tracks[0].detach().cpu().numpy()
            visibility_chunk = pred_visibility[0].detach().cpu().numpy()
            tracks_chunk[:, :, 0] /= sx
            tracks_chunk[:, :, 1] /= sy
            for offset in range(tracks_chunk.shape[0]):
                # Stitch the local chunk frame index back into the original
                # 192-frame video timeline.
                frame_idx = start + offset
                if frame_idx < frames.shape[0]:
                    track_by_frame[frame_idx] = tracks_chunk[offset]
                    visibility_by_frame[frame_idx] = visibility_chunk[offset]

    if not track_by_frame:
        raise RuntimeError("CoTracker produced no tracks")

    point_rows: list[dict[str, object]] = []
    center_rows: list[dict[str, object]] = []
    for frame_idx in sorted(track_by_frame):
        tracks = track_by_frame[frame_idx]
        visibility = visibility_by_frame[frame_idx]
        frame_1based = frame_idx + 1
        row = {"frame": frame_1based, "time": f"{frame_idx / args.fps:.6f}"}
        for point_idx, name in enumerate(POINT_NAMES):
            x, y = tracks[point_idx]
            row[f"{name}_x"] = f"{x:.3f}"
            row[f"{name}_y"] = f"{y:.3f}"
            row[f"{name}_visible"] = f"{float(visibility[point_idx]):.6f}"
        point_rows.append(row)

        cx, cy = tracks[0]
        center_rows.append(
            {
                "frame": frame_1based,
                "time": f"{frame_idx / args.fps:.6f}",
                "ball_center_x": f"{cx:.3f}",
                "ball_center_y": f"{cy:.3f}",
                "source": "cotracker_center",
            }
        )

    point_fields = ["frame", "time"]
    for name in POINT_NAMES:
        point_fields.extend([f"{name}_x", f"{name}_y", f"{name}_visible"])
    write_csv(results_dir / "cotracker_points.csv", point_rows, point_fields)
    write_csv(
        results_dir / "ball_trajectory.csv",
        center_rows,
        ["frame", "time", "ball_center_x", "ball_center_y", "source"],
    )
    print(f"frames: {len(center_rows)}")
    print(f"points: {POINT_NAMES}")
    print(f"trajectory: {results_dir / 'ball_trajectory.csv'}")
    print(f"points_csv: {results_dir / 'cotracker_points.csv'}")


if __name__ == "__main__":
    main()
