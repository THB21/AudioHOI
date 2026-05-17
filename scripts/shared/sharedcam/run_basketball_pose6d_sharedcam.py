#!/usr/bin/env python3
"""Alternative basketball pose baseline in the GVHMR full-image camera frame.

This keeps the current object-side branch intact and writes a parallel result under
`results/pose6d_sharedcam/`.
"""

from __future__ import annotations

import argparse
import csv
import pickle
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from scipy.optimize import least_squares

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BALL_RADIUS_M = 0.12


@dataclass
class CameraModel:
    fx: float
    fy: float
    cx: float
    cy: float
    floor_v: float


class SphereShape:
    def __init__(self, radius_m: float) -> None:
        self.radius_m = float(radius_m)

    def project(self, translation: np.ndarray, camera: CameraModel) -> dict[str, float]:
        x, y, z = translation.tolist()
        z = max(z, 1e-6)
        r_px = camera.fx * self.radius_m / z
        u = camera.cx + camera.fx * x / z
        v = camera.cy + camera.fy * y / z
        bottom_v = v + r_px
        return {
            "u": float(u),
            "v": float(v),
            "r_px": float(r_px),
            "diameter_px": float(2.0 * r_px),
            "left": float(u - r_px),
            "right": float(u + r_px),
            "top": float(v - r_px),
            "bottom": float(bottom_v),
            "bottom_v": float(bottom_v),
            "area_px": float(np.pi * r_px * r_px),
        }

    def support_point(self, translation: np.ndarray) -> np.ndarray:
        x, y, z = translation.tolist()
        return np.array([x, y + self.radius_m, z], dtype=np.float64)


def read_tracking(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("ball_center_x") or not row.get("ball_center_y") or not row.get("radius"):
                continue
            rows.append(
                {
                    "frame": float(row["frame"]),
                    "time": float(row["time"]),
                    "u": float(row["ball_center_x"]),
                    "v": float(row["ball_center_y"]),
                    "r": float(row["radius"]),
                    "mask_area": float(row.get("mask_area", 0.0) or 0.0),
                }
            )
    if not rows:
        raise RuntimeError(f"No valid tracking rows in {path}")
    return rows


def sample_contour_points(contour: np.ndarray, num_points: int = 24) -> np.ndarray:
    if contour.shape[0] <= num_points:
        return contour.astype(np.float64)
    idx = np.linspace(0, contour.shape[0] - 1, num_points, dtype=int)
    return contour[idx].astype(np.float64)


def read_mask_silhouettes(masks_dir: Path) -> dict[int, dict[str, float]]:
    silhouettes: dict[int, dict[str, object]] = {}
    for mask_path in sorted(masks_dir.glob("*_mask.png")):
        frame = int(mask_path.stem.split("_")[0])
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        binary = (mask > 0).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
        ys, xs = np.where(mask > 0)
        if xs.size == 0 or ys.size == 0:
            continue
        xmin = float(xs.min())
        xmax = float(xs.max())
        ymin = float(ys.min())
        ymax = float(ys.max())
        area = float(xs.size)
        silhouettes[frame] = {
            "mask_left": xmin,
            "mask_right": xmax,
            "mask_top": ymin,
            "mask_bottom": ymax,
            "mask_width": xmax - xmin,
            "mask_height": ymax - ymin,
            "mask_size": 0.5 * ((xmax - xmin) + (ymax - ymin)),
            "mask_area": area,
            "contour_points": sample_contour_points(contour, num_points=24),
        }
    if not silhouettes:
        raise RuntimeError(f"No valid masks in {masks_dir}")
    return silhouettes


def circle_contour_points(pred: dict[str, float], num_points: int = 24) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)
    cx = pred["u"]
    cy = pred["v"]
    r = pred["r_px"]
    return np.stack([cx + r * np.cos(angles), cy + r * np.sin(angles)], axis=1).astype(np.float64)


def contour_chamfer_residuals(pred_pts: np.ndarray, obs_pts: np.ndarray, scale: float) -> np.ndarray:
    if pred_pts.size == 0 or obs_pts.size == 0:
        return np.zeros(0, dtype=np.float64)
    diff = pred_pts[:, None, :] - obs_pts[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    pred_to_obs = dist.min(axis=1) / scale
    obs_to_pred = dist.min(axis=0) / scale
    return np.concatenate([pred_to_obs, obs_to_pred], axis=0)


def load_observations(tracking_path: Path, masks_dir: Path) -> list[dict[str, float]]:
    tracking_rows = read_tracking(tracking_path)
    silhouettes = read_mask_silhouettes(masks_dir)
    rows: list[dict[str, float]] = []
    for row in tracking_rows:
        frame = int(row["frame"])
        if frame not in silhouettes:
            continue
        merged = dict(row)
        merged.update(silhouettes[frame])
        rows.append(merged)
    if not rows:
        raise RuntimeError("No overlapping tracking rows and silhouettes")
    return rows


def read_contact_frames(path: Path) -> list[int]:
    frames: list[int] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames.append(int(row["visual_frame"]))
    if not frames:
        raise RuntimeError(f"No contact frames in {path}")
    return frames


def read_alignment_frames(path: Path) -> list[int]:
    if not path.exists():
        return []
    frames: list[int] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            vf = row.get("visual_frame", "")
            if vf != "":
                frames.append(int(vf))
    return frames


def read_gvhmr_camera(path: Path) -> CameraModel:
    with path.open("rb") as f:
        data = pickle.load(f)
    K = np.asarray(data["K_fullimg"], dtype=np.float64)
    K0 = K[0]
    return CameraModel(
        fx=float(K0[0, 0]),
        fy=float(K0[1, 1]),
        cx=float(K0[0, 2]),
        cy=float(K0[1, 2]),
        floor_v=0.0,  # filled later from contact frames
    )


def build_init_translations(obs_rows: list[dict[str, float]], camera: CameraModel, shape: SphereShape) -> np.ndarray:
    translations = []
    for row in obs_rows:
        z = camera.fx * shape.radius_m / max(row["r"], 1e-6)
        x = (row["u"] - camera.cx) * z / camera.fx
        y = (row["v"] - camera.cy) * z / camera.fy
        translations.append(np.array([x, y, z], dtype=np.float64))
    return np.stack(translations, axis=0)


def build_segments(num_frames: int, contact_indices: set[int], times: np.ndarray) -> list[dict[str, np.ndarray | int]]:
    contacts = sorted(contact_indices)
    if not contacts:
        return [{"start": 0, "end": num_frames - 1, "indices": np.arange(num_frames), "tau": times - times[0]}]

    segments: list[dict[str, np.ndarray | int]] = []
    start = 0
    for contact_idx in contacts:
        end = max(start, contact_idx)
        indices = np.arange(start, end + 1, dtype=np.int32)
        tau = times[indices] - times[indices[0]]
        segments.append({"start": start, "end": end, "indices": indices, "tau": tau})
        start = contact_idx + 1
    if start <= num_frames - 1:
        indices = np.arange(start, num_frames, dtype=np.int32)
        tau = times[indices] - times[indices[0]]
        segments.append({"start": start, "end": num_frames - 1, "indices": indices, "tau": tau})
    return segments


def fit_segment_lines(init_t: np.ndarray, segments: list[dict[str, np.ndarray | int]]) -> np.ndarray:
    ab = np.zeros((len(segments), 2), dtype=np.float64)
    for seg_idx, seg in enumerate(segments):
        indices = seg["indices"]
        tau = seg["tau"]
        z = init_t[indices, 2]
        if len(indices) == 1 or float(np.max(tau)) <= 1e-8:
            ab[seg_idx] = np.array([0.0, float(z[0])], dtype=np.float64)
            continue
        a, b = np.polyfit(tau, z, deg=1)
        ab[seg_idx] = np.array([float(a), float(b)], dtype=np.float64)
    return ab


def unpack_state(flat_state: np.ndarray, num_frames: int, segments: list[dict[str, np.ndarray | int]]) -> tuple[np.ndarray, np.ndarray]:
    xy = flat_state[: 2 * num_frames].reshape(num_frames, 2)
    ab = flat_state[2 * num_frames :].reshape(len(segments), 2)
    z = np.zeros(num_frames, dtype=np.float64)
    for seg_idx, seg in enumerate(segments):
        indices = seg["indices"]
        tau = seg["tau"]
        a, b = ab[seg_idx]
        z[indices] = a * tau + b
    t = np.column_stack([xy[:, 0], xy[:, 1], z])
    return t, ab


def pose_residuals(
    flat_state: np.ndarray,
    obs_rows: list[dict[str, float]],
    segments: list[dict[str, np.ndarray | int]],
    camera: CameraModel,
    shape: SphereShape,
    contact_indices: set[int],
    weak_contact_indices: set[int],
    mask_weight: float,
    temp_weight: float,
    z_temp_weight: float,
    z_boundary_weight: float,
    z_slope_weight: float,
    contact_weight: float,
    center_weight: float,
    size_weight: float,
) -> np.ndarray:
    t, ab = unpack_state(flat_state, len(obs_rows), segments)
    residuals: list[float] = []

    for idx, row in enumerate(obs_rows):
        pred = shape.project(t[idx], camera)

        pred_contour = circle_contour_points(pred, num_points=24)
        obs_contour = row["contour_points"]
        residuals.extend((mask_weight * contour_chamfer_residuals(pred_contour, obs_contour, scale=12.0)).tolist())

        residuals.append(center_weight * (pred["u"] - row["u"]))
        residuals.append(center_weight * (pred["v"] - row["v"]))

        residuals.append(size_weight * (pred["diameter_px"] - row["mask_width"]))
        residuals.append(size_weight * (pred["diameter_px"] - row["mask_height"]))
        residuals.append(size_weight * (pred["diameter_px"] - row["mask_size"]))
        residuals.append(size_weight * ((pred["area_px"] - row["mask_area"]) / 1000.0))

        # Shared-camera contact: keep the projected ball bottom close to the contact floor line.
        if idx in contact_indices:
            residuals.append(contact_weight * ((pred["bottom_v"] - camera.floor_v) / 20.0))
        elif idx in weak_contact_indices:
            residuals.append(0.5 * contact_weight * ((pred["bottom_v"] - camera.floor_v) / 20.0))

        residuals.append(0.30 * max(0.0, 0.35 - t[idx, 2]))

    for idx in range(1, len(t) - 1):
        accel = t[idx + 1] - 2.0 * t[idx] + t[idx - 1]
        residuals.extend((temp_weight * accel).tolist())
        residuals.append(z_temp_weight * accel[2])

    for seg_idx in range(len(segments) - 1):
        end_idx = int(segments[seg_idx]["end"])
        next_start_idx = int(segments[seg_idx + 1]["start"])
        residuals.append(z_boundary_weight * (t[next_start_idx, 2] - t[end_idx, 2]))
        residuals.append(z_slope_weight * (ab[seg_idx + 1, 0] - ab[seg_idx, 0]))

    for a, _b in ab:
        residuals.append(0.12 * a)

    if len(t) >= 2:
        residuals.extend((0.15 * (t[1] - t[0])).tolist())
        residuals.extend((0.15 * (t[-1] - t[-2])).tolist())

    return np.array(residuals, dtype=np.float64)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_outputs(out_dir: Path, times: np.ndarray, init_t: np.ndarray, opt_t: np.ndarray, residual_px: np.ndarray) -> None:
    fig = plt.figure(figsize=(8.5, 6.5), dpi=140)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(opt_t[:, 0], opt_t[:, 2], opt_t[:, 1], linewidth=2.2, color="#4c72b0")
    ax.scatter(opt_t[:, 0], opt_t[:, 2], opt_t[:, 1], c=times, cmap="plasma", s=14)
    ax.set_xlabel("X_cam (m)")
    ax.set_ylabel("Z_cam (m)")
    ax.set_zlabel("Y_cam (m)")
    ax.set_title("Basketball shared-camera pose baseline")
    ax.view_init(elev=22, azim=-58)
    fig.tight_layout()
    fig.savefig(out_dir / "ball_pose6d_sharedcam_plot.png")
    plt.close(fig)

    fig, axes = plt.subplots(4, 1, figsize=(9, 8.5), dpi=140, sharex=True)
    labels = [("X_cam", 0), ("Y_cam", 1), ("Z_cam", 2)]
    for ax, (name, col) in zip(axes[:3], labels):
        ax.plot(times, init_t[:, col], label=f"{name} init", color="#c7d2e8")
        ax.plot(times, opt_t[:, col], label=f"{name} opt", color="#4c72b0")
        ax.set_ylabel(f"{name} (m)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.25)
    axes[3].plot(times, residual_px, color="#dd8452", label="reprojection err")
    axes[3].set_ylabel("err (px)")
    axes[3].set_xlabel("time (s)")
    axes[3].legend(loc="upper right")
    axes[3].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "ball_pose6d_sharedcam_components.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Alternative basketball baseline in the GVHMR camera frame.")
    parser.add_argument("--sample-dir", type=Path, default=Path("samples/basketball_01"))
    parser.add_argument("--ball-radius-m", type=float, default=BALL_RADIUS_M)
    parser.add_argument("--mask-weight", type=float, default=0.018)
    parser.add_argument("--temp-weight", type=float, default=0.08)
    parser.add_argument("--z-temp-weight", type=float, default=0.22)
    parser.add_argument("--z-boundary-weight", type=float, default=3.5)
    parser.add_argument("--z-slope-weight", type=float, default=0.35)
    parser.add_argument("--contact-weight", type=float, default=10.0)
    parser.add_argument("--center-weight", type=float, default=0.04)
    parser.add_argument("--size-weight", type=float, default=0.02)
    args = parser.parse_args()

    sample_dir = args.sample_dir
    results_dir = sample_dir / "results"
    out_dir = results_dir / "pose6d_sharedcam"
    out_dir.mkdir(parents=True, exist_ok=True)

    obs_rows = load_observations(results_dir / "tracking" / "ball_trajectory.csv", results_dir / "segmentation" / "masks")
    contact_frames = read_contact_frames(results_dir / "events" / "visual_events.csv")
    audio_frames = read_alignment_frames(results_dir / "events" / "audio_visual_alignment.csv")

    frames = np.array([int(r["frame"]) for r in obs_rows], dtype=np.int32)
    times = np.array([r["time"] for r in obs_rows], dtype=np.float64)
    obs_bottoms = np.array([r["v"] + r["r"] for r in obs_rows], dtype=np.float64)
    frame_to_idx = {frame: idx for idx, frame in enumerate(frames.tolist())}
    contact_indices = {frame_to_idx[f] for f in contact_frames if f in frame_to_idx}
    audio_contact_indices = {frame_to_idx[f] for f in audio_frames if f in frame_to_idx}
    if not contact_indices:
        raise RuntimeError("No overlapping basketball contact frames for shared-camera baseline")

    camera = read_gvhmr_camera(results_dir / "gvhmr" / "result.pkl")
    camera.floor_v = float(np.median([obs_bottoms[idx] for idx in sorted(contact_indices)]))
    shape = SphereShape(args.ball_radius_m)

    init_t = build_init_translations(obs_rows, camera, shape)
    segments = build_segments(len(obs_rows), contact_indices, times)
    init_ab = fit_segment_lines(init_t, segments)
    init_state = np.concatenate([init_t[:, :2].reshape(-1), init_ab.reshape(-1)])

    result = least_squares(
        pose_residuals,
        init_state,
        method="trf",
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=250,
        args=(
            obs_rows,
            segments,
            camera,
            shape,
            contact_indices,
            audio_contact_indices,
            args.mask_weight,
            args.temp_weight,
            args.z_temp_weight,
            args.z_boundary_weight,
            args.z_slope_weight,
            args.contact_weight,
            args.center_weight,
            args.size_weight,
        ),
    )

    opt_t, _opt_ab = unpack_state(result.x, len(obs_rows), segments)
    opt_t[:, 2] = np.maximum(opt_t[:, 2], 0.35)

    out_rows: list[dict[str, object]] = []
    reproj_rows: list[dict[str, object]] = []
    residual_px: list[float] = []
    for idx, row in enumerate(obs_rows):
        pred = shape.project(opt_t[idx], camera)
        err_u = pred["u"] - row["u"]
        err_v = pred["v"] - row["v"]
        err_px = float(np.hypot(err_u, err_v))
        residual_px.append(err_px)
        out_rows.append(
            {
                "frame": int(row["frame"]),
                "time": f"{row['time']:.6f}",
                "tx": f"{opt_t[idx, 0]:.6f}",
                "ty": f"{opt_t[idx, 1]:.6f}",
                "tz": f"{opt_t[idx, 2]:.6f}",
                "qw": "1.000000",
                "qx": "0.000000",
                "qy": "0.000000",
                "qz": "0.000000",
                "radius_m": f"{shape.radius_m:.6f}",
                "coord_frame": "gvhmr_incam",
                "u_obs": f"{row['u']:.3f}",
                "v_obs": f"{row['v']:.3f}",
                "radius_obs_px": f"{row['r']:.3f}",
                "u_proj": f"{pred['u']:.3f}",
                "v_proj": f"{pred['v']:.3f}",
                "radius_proj_px": f"{pred['r_px']:.3f}",
                "bottom_proj_v": f"{pred['bottom_v']:.3f}",
                "floor_v": f"{camera.floor_v:.3f}",
                "residual_px": f"{err_px:.6f}",
                "contact_frame": int(idx in contact_indices),
                "audio_contact_frame": int(idx in audio_contact_indices),
            }
        )
        reproj_rows.append(
            {
                "frame": int(row["frame"]),
                "u_obs": f"{row['u']:.3f}",
                "v_obs": f"{row['v']:.3f}",
                "u_reproj": f"{pred['u']:.3f}",
                "v_reproj": f"{pred['v']:.3f}",
                "error_u": f"{err_u:.6f}",
                "error_v": f"{err_v:.6f}",
                "error_px": f"{err_px:.6f}",
            }
        )

    write_csv(
        out_dir / "ball_pose6d_sharedcam_trajectory.csv",
        out_rows,
        [
            "frame",
            "time",
            "tx",
            "ty",
            "tz",
            "qw",
            "qx",
            "qy",
            "qz",
            "radius_m",
            "coord_frame",
            "u_obs",
            "v_obs",
            "radius_obs_px",
            "u_proj",
            "v_proj",
            "radius_proj_px",
            "bottom_proj_v",
            "floor_v",
            "residual_px",
            "contact_frame",
            "audio_contact_frame",
        ],
    )
    write_csv(
        out_dir / "ball_pose6d_sharedcam_reprojection_comparison.csv",
        reproj_rows,
        ["frame", "u_obs", "v_obs", "u_reproj", "v_reproj", "error_u", "error_v", "error_px"],
    )

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), dpi=140, sharex=True)
    u_obs = np.array([float(r["u_obs"]) for r in reproj_rows])
    v_obs = np.array([float(r["v_obs"]) for r in reproj_rows])
    u_reproj = np.array([float(r["u_reproj"]) for r in reproj_rows])
    v_reproj = np.array([float(r["v_reproj"]) for r in reproj_rows])
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
    fig.savefig(out_dir / "ball_pose6d_sharedcam_reprojection_comparison.png")
    plt.close(fig)

    plot_outputs(out_dir, times, init_t, opt_t, np.array(residual_px, dtype=np.float64))

    print(f"sharedcam_csv: {out_dir / 'ball_pose6d_sharedcam_trajectory.csv'}")
    print(f"reproj_csv: {out_dir / 'ball_pose6d_sharedcam_reprojection_comparison.csv'}")
    print(f"plot_png: {out_dir / 'ball_pose6d_sharedcam_plot.png'}")
    print(f"components_png: {out_dir / 'ball_pose6d_sharedcam_components.png'}")
    print(f"reproj_png: {out_dir / 'ball_pose6d_sharedcam_reprojection_comparison.png'}")
    print(f"mean_reproj_px: {float(np.mean(residual_px)):.6f}")
    print(f"median_reproj_px: {float(np.median(residual_px)):.6f}")
    print(f"K_fullimg_fx: {camera.fx:.3f}")
    print(f"K_fullimg_cx: {camera.cx:.3f}")
    print(f"floor_v: {camera.floor_v:.3f}")
    print(f"optimizer_cost: {float(result.cost):.6f}")


if __name__ == "__main__":
    main()
