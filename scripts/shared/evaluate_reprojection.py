#!/usr/bin/env python3
"""Evaluate reprojection of lifted 3D trajectory back to image plane.

Reads `samples/.../results/lifting/ball_3d_lifted_trajectory.csv` and writes
`ball_3d_reprojection_comparison.csv` and `ball_3d_reprojection_comparison.png`.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_lifted(path: Path) -> list[dict[str, float]]:
    rows = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                {
                    "frame": int(r["frame"]),
                    "u_obs": float(r["u"]),
                    "v_obs": float(r["v"]),
                    "radius_px": float(r["radius_px"]),
                    "X": float(r["X"]),
                    "Y": float(r["Y"]),
                    "Z": float(r["Z"]),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    sample_dir = Path("samples/basketball_01")
    results_dir = sample_dir / "results"
    lifting_dir = results_dir / "lifting"
    events_dir = results_dir / "events"
    lifted = read_lifted(lifting_dir / "ball_3d_lifted_trajectory.csv")

    image_width = 1280
    image_height = 720
    fx = fy = image_width * 0.90
    cx = image_width * 0.5
    cy = image_height * 0.5
    ball_radius_m = 0.12

    # compute floor_v using visual contact frames (same logic as the lifting script)
    contact_frames = []
    with (events_dir / "visual_events.csv").open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            contact_frames.append(int(row["visual_frame"]))
    frame_to_row = {r["frame"]: r for r in lifted}
    contact_bottoms = [frame_to_row[f]["v_obs"] + frame_to_row[f]["radius_px"] for f in contact_frames if f in frame_to_row]
    if not contact_bottoms:
        bottoms = [r["v_obs"] + r["radius_px"] for r in lifted]
        floor_v = float(np.median(bottoms))
    else:
        floor_v = float(np.median(contact_bottoms))

    out_rows = []
    for r in lifted:
        X = r["X"]
        Y = r["Y"]
        Z = r["Z"]
        # projected radius (px)
        r_proj = fx * ball_radius_m / max(Z, 1e-6)
        bottom_v_reproj = floor_v - (Y - ball_radius_m) * fy / max(Z, 1e-6)
        v_reproj = bottom_v_reproj - r_proj
        u_reproj = cx + X * fx / max(Z, 1e-6)

        out_rows.append(
            {
                "frame": int(r["frame"]),
                "u_obs": f"{r['u_obs']:.3f}",
                "v_obs": f"{r['v_obs']:.3f}",
                "u_reproj": f"{u_reproj:.3f}",
                "v_reproj": f"{v_reproj:.3f}",
                "error_u": f"{u_reproj - r['u_obs']:.6f}",
                "error_v": f"{v_reproj - r['v_obs']:.6f}",
                "error_px": f"{np.hypot(u_reproj - r['u_obs'], v_reproj - r['v_obs']):.6f}",
            }
        )

    out_csv = lifting_dir / "ball_3d_reprojection_comparison.csv"
    write_csv(out_csv, out_rows, ["frame", "u_obs", "v_obs", "u_reproj", "v_reproj", "error_u", "error_v", "error_px"])

    # plot reprojection comparison
    frames = [int(r["frame"]) for r in out_rows]
    u_obs = np.array([float(r["u_obs"]) for r in out_rows])
    v_obs = np.array([float(r["v_obs"]) for r in out_rows])
    u_reproj = np.array([float(r["u_reproj"]) for r in out_rows])
    v_reproj = np.array([float(r["v_reproj"]) for r in out_rows])

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), dpi=140, sharex=True)
    axes[0].plot(frames, u_obs, label="u_obs", color="#4c72b0")
    axes[0].plot(frames, u_reproj, label="u_reproj", color="#dd8452")
    axes[0].set_ylabel("u (px)")
    axes[0].legend()

    axes[1].plot(frames, v_obs, label="v_obs", color="#4c72b0")
    axes[1].plot(frames, v_reproj, label="v_reproj", color="#dd8452")
    axes[1].set_ylabel("v (px)")
    axes[1].set_xlabel("frame")
    axes[1].legend()

    fig.tight_layout()
    out_png = lifting_dir / "ball_3d_reprojection_comparison.png"
    fig.savefig(out_png)
    plt.close(fig)

    print(f"wrote: {out_csv}")
    print(f"wrote: {out_png}")


if __name__ == "__main__":
    main()
