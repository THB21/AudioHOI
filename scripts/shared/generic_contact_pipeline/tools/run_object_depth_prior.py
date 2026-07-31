#!/usr/bin/env python3
"""Reduce dense scene depth to one tracked-object metric depth prior per frame."""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _by_frame(path: Path) -> dict[int, dict[str, str]]:
    return {int(float(row["frame"])): row for row in _rows(path)}


def _install_directory_atomic(staged: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    backup = target.parent / f".{target.name}.previous"
    if backup.exists():
        shutil.rmtree(backup)
    if target.exists():
        os.replace(target, backup)
    try:
        os.replace(staged, target)
    except Exception:
        if backup.exists() and not target.exists():
            os.replace(backup, target)
        raise
    shutil.rmtree(backup, ignore_errors=True)


def build_object_depth_prior(sample_dir: Path, *, smooth_window: int = 7) -> dict[str, object]:
    sample_dir = sample_dir.resolve()
    if smooth_window <= 0 or smooth_window % 2 == 0:
        raise ValueError("smooth window must be a positive odd integer")
    results = sample_dir / "results"
    depth_dir = results / "da3/scene_depth"
    depth_index = depth_dir / "index.csv"
    masks_dir = results / "segmentation/masks"
    trajectory = results / "tracking/object_trajectory.csv"
    center_trajectory = results / "tracking/object_center_trajectory.csv"
    for required in (depth_index, masks_dir, trajectory, center_trajectory):
        if not required.exists():
            raise FileNotFoundError(f"object depth prior input is missing: {required}")

    depth_rows = _by_frame(depth_index)
    track_rows = _by_frame(trajectory)
    center_rows = _by_frame(center_trajectory)
    frames = tuple(sorted(depth_rows))
    if (
        frames != tuple(range(1, len(frames) + 1))
        or set(track_rows) != set(frames)
        or set(center_rows) != set(frames)
    ):
        raise ValueError("depth index and object trajectories must cover the same contiguous frames")

    observed_depth: list[float] = []
    confidence: list[float] = []
    observation_state: list[str] = []
    for frame in frames:
        depth_path = depth_dir / depth_rows[frame]["file"]
        mask_path = masks_dir / f"{frame:05d}_mask.png"
        if not depth_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(f"missing frame-aligned depth/mask for frame {frame}")
        depth = np.load(depth_path, allow_pickle=False)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if depth.ndim != 2 or mask is None:
            raise ValueError(f"invalid depth or mask for frame {frame}")
        selected = cv2.resize(mask, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST) > 127
        values = np.asarray(depth[selected], dtype=float)
        values = values[np.isfinite(values) & (values > 0.0)]
        if not len(values):
            observed_depth.append(float("nan"))
            confidence.append(0.0)
            observation_state.append("missing_mask_interpolated")
            continue
        observed_depth.append(float(np.median(values)))
        confidence.append(float(min(1.0, len(values) / 64.0)))
        observation_state.append("observed_mask_median")

    observed_array = np.asarray(observed_depth, dtype=float)
    valid = np.isfinite(observed_array)
    if not np.any(valid):
        raise ValueError("object masks select no valid metric depth in any frame")
    indices = np.arange(len(observed_array), dtype=float)
    raw_depth = np.interp(indices, indices[valid], observed_array[valid]).tolist()

    half = smooth_window // 2
    smooth_depth = [
        float(np.median(raw_depth[max(0, index - half) : min(len(raw_depth), index + half + 1)]))
        for index in range(len(raw_depth))
    ]
    output_rows: list[dict[str, object]] = []
    for index, frame in enumerate(frames):
        track = track_rows[frame]
        center = center_rows[frame]
        u = track.get("ball_center_x") or track.get("center_x") or center.get("ball_center_x") or center.get("center_x", "")
        v = track.get("ball_center_y") or track.get("center_y") or center.get("ball_center_y") or center.get("center_y", "")
        output_rows.append(
            {
                "frame": frame,
                "time": track.get("time", ""),
                "u": u,
                "v": v,
                "radius_px": track.get("radius", ""),
                "da3_depth_raw": f"{raw_depth[index]:.6f}",
                "da3_depth_smooth": f"{smooth_depth[index]:.6f}",
                "object_depth_confidence": f"{confidence[index]:.6f}",
                "depth_observation_state": observation_state[index],
                "source": (
                    "da3_metric_sam2_mask_median"
                    if observation_state[index] == "observed_mask_median"
                    else "da3_metric_temporal_interpolation_for_missing_mask"
                ),
            }
        )

    work = Path(tempfile.mkdtemp(prefix="audiohoi-object-depth-", dir=results))
    staged = work / "priors"
    staged.mkdir()
    fields = tuple(output_rows[0])
    with (staged / "object_depth_prior.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    summary = {
        "schema_version": 1,
        "status": "generated",
        "sample_dir": str(sample_dir),
        "frame_count": len(frames),
        "smooth_window": smooth_window,
        "scene_depth_dir": str(depth_dir),
        "mask_dir": str(masks_dir),
        "object_prior_source": "da3_metric_sam2_mask_median",
        "missing_mask_interpolated_frames": [
            frame
            for frame, state in zip(frames, observation_state)
            if state == "missing_mask_interpolated"
        ],
    }
    (staged / "meta.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    target = results / "da3/priors"
    try:
        _install_directory_atomic(staged, target)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return summary | {"output": str(target / "object_depth_prior.csv")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--smooth-window", type=int, default=7)
    args = parser.parse_args()
    print(json.dumps(build_object_depth_prior(args.sample_dir, smooth_window=args.smooth_window), sort_keys=True))


if __name__ == "__main__":
    main()
