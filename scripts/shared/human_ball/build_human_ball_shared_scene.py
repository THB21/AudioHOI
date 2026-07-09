#!/usr/bin/env python3
"""Export a first shared camera-frame scene for GVHMR human + basketball pose6d."""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np


def read_ball_pose(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "frame": int(row["frame"]),
                    "time": float(row["time"]),
                    "ball_x": float(row["tx"]),
                    "ball_y": float(row["ty"]),
                    "ball_z": float(row["tz"]),
                    "ball_radius_m": float(row["radius_m"]),
                }
            )
    if not rows:
        raise RuntimeError(f"No ball pose rows found in {path}")
    return rows


def read_human_incam(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as f:
        data = pickle.load(f)
    transl = np.asarray(data["smpl_params_incam"]["transl"], dtype=np.float64)
    K = np.asarray(data["K_fullimg"], dtype=np.float64)
    if transl.ndim != 2 or transl.shape[1] != 3:
        raise RuntimeError(f"Unexpected human transl shape: {transl.shape}")
    return transl, K


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a first shared camera-frame scene for human and basketball.")
    parser.add_argument("--sample-dir", type=Path, default=Path("samples/basketball_01"))
    args = parser.parse_args()

    sample_dir = args.sample_dir
    results_dir = sample_dir / "results"
    joint_dir = results_dir / "joint"
    joint_dir.mkdir(parents=True, exist_ok=True)

    ball_rows = read_ball_pose(results_dir / "pose6d" / "ball_pose6d_trajectory.csv")
    human_transl, K = read_human_incam(results_dir / "gvhmr" / "result.pkl")

    if len(ball_rows) != len(human_transl):
        raise RuntimeError(f"Length mismatch: ball={len(ball_rows)} human={len(human_transl)}")

    shared_rows: list[dict[str, object]] = []
    dists = []
    for idx, ball in enumerate(ball_rows):
        human = human_transl[idx]
        dx = ball["ball_x"] - float(human[0])
        dy = ball["ball_y"] - float(human[1])
        dz = ball["ball_z"] - float(human[2])
        dist = float(np.sqrt(dx * dx + dy * dy + dz * dz))
        dists.append(dist)
        shared_rows.append(
            {
                "frame": ball["frame"],
                "time": f"{ball['time']:.6f}",
                "human_root_x": f"{human[0]:.6f}",
                "human_root_y": f"{human[1]:.6f}",
                "human_root_z": f"{human[2]:.6f}",
                "ball_x": f"{ball['ball_x']:.6f}",
                "ball_y": f"{ball['ball_y']:.6f}",
                "ball_z": f"{ball['ball_z']:.6f}",
                "ball_radius_m": f"{ball['ball_radius_m']:.6f}",
                "delta_x": f"{dx:.6f}",
                "delta_y": f"{dy:.6f}",
                "delta_z": f"{dz:.6f}",
                "root_to_ball_dist": f"{dist:.6f}",
            }
        )

    csv_path = joint_dir / "human_ball_camera_scene.csv"
    write_csv(
        csv_path,
        shared_rows,
        [
            "frame",
            "time",
            "human_root_x",
            "human_root_y",
            "human_root_z",
            "ball_x",
            "ball_y",
            "ball_z",
            "ball_radius_m",
            "delta_x",
            "delta_y",
            "delta_z",
            "root_to_ball_dist",
        ],
    )

    summary_path = joint_dir / "human_ball_camera_scene_summary.txt"
    summary_lines = [
        "Shared frame: camera coordinates",
        "Human source: GVHMR smpl_params_incam.transl",
        "Ball source: pose6d tx/ty/tz",
        f"num_frames: {len(shared_rows)}",
        f"mean_root_to_ball_dist_m: {np.mean(dists):.6f}",
        f"min_root_to_ball_dist_m: {np.min(dists):.6f}",
        f"max_root_to_ball_dist_m: {np.max(dists):.6f}",
        f"human_K_fx_fy_cx_cy: {K[0,0,0]:.6f}, {K[0,1,1]:.6f}, {K[0,0,2]:.6f}, {K[0,1,2]:.6f}",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n")

    print(f"shared_scene_csv: {csv_path}")
    print(f"summary_txt: {summary_path}")
    print(f"mean_root_to_ball_dist_m: {np.mean(dists):.6f}")


if __name__ == "__main__":
    main()
