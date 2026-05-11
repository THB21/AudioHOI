#!/usr/bin/env python3
"""Minimal monocular 3D lifting baseline for the basketball sample."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BALL_RADIUS_M = 0.12


def read_trajectory(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("ball_center_x", "") == "" or row.get("ball_center_y", "") == "":
                continue
            radius = row.get("radius", "")
            if radius == "":
                continue
            rows.append(
                {
                    "frame": float(row["frame"]),
                    "time": float(row["time"]),
                    "u": float(row["ball_center_x"]),
                    "v": float(row["ball_center_y"]),
                    "r": float(radius),
                }
            )
    if not rows:
        raise RuntimeError(f"No valid trajectory rows in {path}")
    return rows


def read_contact_frames(path: Path) -> list[int]:
    frames: list[int] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames.append(int(row["visual_frame"]))
    if not frames:
        raise RuntimeError(f"No contact frames found in {path}")
    return frames


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fit_segments(times: np.ndarray, values: np.ndarray, contact_indices: list[int], degree: int) -> np.ndarray:
    fitted = values.copy()
    boundaries = [0] + sorted(set(contact_indices)) + [len(values) - 1]
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end <= start:
            continue
        # Include both boundary contact frames in the local fit.
        seg_slice = slice(start, end + 1)
        seg_t = times[seg_slice]
        seg_v = values[seg_slice]
        deg = min(degree, len(seg_v) - 1)
        if deg < 1:
            continue
        coeffs = np.polyfit(seg_t, seg_v, deg)
        fitted[seg_slice] = np.polyval(coeffs, seg_t)
    return fitted


def fit_segments_parabolic_with_contact(
    times: np.ndarray, values: np.ndarray, contact_indices: list[int], floor_value: float
) -> np.ndarray:
    """Fit piecewise parabolic (degree=2) trajectories while enforcing contact equality.

    For each segment between contact boundaries we solve a constrained least-squares
    problem for coefficients a,b,c of y(t)=a*t^2+b*t+c such that at any contact
    frames inside the segment y(contact_time)==floor_value.
    """
    fitted = values.copy()
    boundaries = [0] + sorted(set(contact_indices)) + [len(values) - 1]
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end <= start:
            continue
        seg_slice = slice(start, end + 1)
        seg_t = times[seg_slice]
        seg_v = values[seg_slice]

        # If too few points, skip fitting.
        if len(seg_t) < 3:
            # If any contact frames present, force them to floor_value
            for idx in range(start, end + 1):
                if idx in contact_indices:
                    fitted[idx] = floor_value
            continue

        # Build design matrix for quadratic fit: [t^2, t, 1]
        T = np.vstack([seg_t ** 2, seg_t, np.ones_like(seg_t)]).T

        # Equality constraints: rows of C a = d, where a=[a,b,c]
        C_rows = []
        d_rows = []
        for ci in contact_indices:
            if ci >= start and ci <= end:
                local_idx = ci - start
                t_c = seg_t[local_idx]
                C_rows.append([t_c ** 2, t_c, 1.0])
                d_rows.append(floor_value)

        if not C_rows:
            # No equality constraints in this segment; ordinary least squares.
            coeffs, *_ = np.linalg.lstsq(T, seg_v, rcond=None)
        else:
            C = np.array(C_rows, dtype=float)
            d = np.array(d_rows, dtype=float)
            # Solve KKT system for constrained least squares:
            # [T^T T   C^T] [a] = [T^T y]
            # [  C     0 ] [lam] = [ d ]
            TT = T.T @ T
            rhs_top = T.T @ seg_v
            top = np.concatenate([TT, C.T], axis=1)
            bottom = np.concatenate([C, np.zeros((C.shape[0], C.shape[0]))], axis=1)
            KKT = np.concatenate([top, bottom], axis=0)
            rhs = np.concatenate([rhs_top, d], axis=0)
            try:
                sol = np.linalg.solve(KKT, rhs)
                coeffs = sol[:3]
            except np.linalg.LinAlgError:
                # Fallback to unconstrained fit if KKT singular.
                coeffs, *_ = np.linalg.lstsq(T, seg_v, rcond=None)

        fitted[seg_slice] = T @ coeffs

    return fitted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-dir", type=Path, default=Path("samples/basketball_01"))
    parser.add_argument("--ball-radius-m", type=float, default=BALL_RADIUS_M)
    parser.add_argument("--focal-px", type=float, default=None, help="Approximate focal length in pixels.")
    parser.add_argument("--image-width", type=int, default=1280)
    parser.add_argument("--image-height", type=int, default=720)
    args = parser.parse_args()

    sample_dir = args.sample_dir
    results_dir = sample_dir / "results"
    tracking_dir = results_dir / "tracking"
    events_dir = results_dir / "events"
    lifting_dir = results_dir / "lifting"
    lifting_dir.mkdir(parents=True, exist_ok=True)
    trajectory_rows = read_trajectory(tracking_dir / "ball_trajectory.csv")
    contact_frames = read_contact_frames(events_dir / "visual_events.csv")

    frames = np.array([int(row["frame"]) for row in trajectory_rows], dtype=np.int32)
    times = np.array([row["time"] for row in trajectory_rows], dtype=np.float32)
    us = np.array([row["u"] for row in trajectory_rows], dtype=np.float32)
    vs = np.array([row["v"] for row in trajectory_rows], dtype=np.float32)
    rs = np.array([row["r"] for row in trajectory_rows], dtype=np.float32)

    fx = fy = args.focal_px if args.focal_px is not None else args.image_width * 0.90
    cx = args.image_width * 0.5
    cy = args.image_height * 0.5

    # Monocular depth from apparent size.
    z_raw = fx * args.ball_radius_m / np.maximum(rs, 1e-6)

    # Recover bottom in image space and anchor vertical height to the deepest contact floor.
    bottom_v = vs + rs
    contact_mask = np.isin(frames, np.array(contact_frames, dtype=np.int32))
    if not np.any(contact_mask):
        raise RuntimeError("No overlap between trajectory frames and visual contact frames")
    floor_v = float(np.median(bottom_v[contact_mask]))

    x_raw = (us - cx) * z_raw / fx
    # Height above floor from bottom pixel displacement, projected using current depth.
    y_raw = args.ball_radius_m + np.maximum(0.0, floor_v - bottom_v) * z_raw / fy

    frame_to_idx = {frame: idx for idx, frame in enumerate(frames.tolist())}
    contact_indices = [frame_to_idx[frame] for frame in contact_frames if frame in frame_to_idx]

    x_fit = fit_segments(times, x_raw, contact_indices, degree=1)
    y_fit = fit_segments_parabolic_with_contact(times, y_raw, contact_indices, floor_value=args.ball_radius_m)
    z_fit = fit_segments(times, z_raw, contact_indices, degree=1)
    y_fit = np.maximum(args.ball_radius_m, y_fit)

    csv_rows: list[dict[str, object]] = []
    for idx, frame in enumerate(frames):
        csv_rows.append(
            {
                "frame": int(frame),
                "time": f"{times[idx]:.6f}",
                "u": f"{us[idx]:.3f}",
                "v": f"{vs[idx]:.3f}",
                "radius_px": f"{rs[idx]:.3f}",
                "bottom_v": f"{bottom_v[idx]:.3f}",
                "X_raw": f"{x_raw[idx]:.6f}",
                "Y_raw": f"{y_raw[idx]:.6f}",
                "Z_raw": f"{z_raw[idx]:.6f}",
                "X": f"{x_fit[idx]:.6f}",
                "Y": f"{y_fit[idx]:.6f}",
                "Z": f"{z_fit[idx]:.6f}",
            }
        )

    out_csv = lifting_dir / "ball_3d_lifted_trajectory.csv"
    write_csv(
        out_csv,
        csv_rows,
        ["frame", "time", "u", "v", "radius_px", "bottom_v", "X_raw", "Y_raw", "Z_raw", "X", "Y", "Z"],
    )

    # Static diagnostic plots.
    fig = plt.figure(figsize=(8.5, 6.5), dpi=140)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x_fit, z_fit, y_fit, linewidth=2.2, color="#4c72b0")
    ax.scatter(x_fit, z_fit, y_fit, c=times, cmap="viridis", s=14)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_zlabel("Y (m)")
    ax.set_title("Basketball 3D lifting baseline")
    ax.view_init(elev=22, azim=-58)
    fig.tight_layout()
    out_plot = lifting_dir / "ball_3d_lifted_plot.png"
    fig.savefig(out_plot)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(9, 7), dpi=140, sharex=True)
    axes[0].plot(times, x_raw, label="X raw", color="#a0b4d6")
    axes[0].plot(times, x_fit, label="X fit", color="#4c72b0")
    axes[0].set_ylabel("X (m)")
    axes[0].legend(loc="upper right")
    axes[1].plot(times, y_raw, label="Y raw", color="#c6e2b5")
    axes[1].plot(times, y_fit, label="Y fit", color="#55a868")
    axes[1].set_ylabel("Y (m)")
    axes[1].legend(loc="upper right")
    axes[2].plot(times, z_raw, label="Z raw", color="#f3c7a0")
    axes[2].plot(times, z_fit, label="Z fit", color="#dd8452")
    axes[2].set_ylabel("Z (m)")
    axes[2].set_xlabel("time (s)")
    axes[2].legend(loc="upper right")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_diag = lifting_dir / "ball_3d_lifted_components.png"
    fig.savefig(out_diag)
    plt.close(fig)

    print(f"trajectory_csv: {out_csv}")
    print(f"plot_png: {out_plot}")
    print(f"components_png: {out_diag}")
    print(f"focal_px: {fx:.2f}")
    print(f"floor_v: {floor_v:.3f}")


if __name__ == "__main__":
    main()
