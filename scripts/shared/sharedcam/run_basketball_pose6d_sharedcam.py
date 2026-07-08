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
from support_geometry import (
    estimate_support_geometry_from_contacts,
    read_contact_frames,
    write_support_geometry_json,
)

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



def read_radius_estimates(path: Path) -> dict[int, float]:
    """Load per-frame radius estimates from SAM 3D Objects."""
    radius_by_frame = {}
    if not path.exists():
        return radius_by_frame
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(row["frame"])
            radius_str = row.get("estimated_radius_m", "")
            if radius_str:
                try:
                    radius_by_frame[frame] = float(radius_str)
                except ValueError:
                    pass
    return radius_by_frame


def read_object_observations(path: Path, radius_estimates: dict[int, float] | None = None) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            cx = row.get("center_x", "")
            cy = row.get("center_y", "")
            radius = row.get("enclosing_radius_px", "")
            if not cx or not cy or not radius:
                continue
            bbox_w = float(row.get("bbox_w_px", 0.0) or 0.0)
            bbox_h = float(row.get("bbox_h_px", 0.0) or 0.0)
            frame = int(row["frame"])
            rows.append(
                {
                    "frame": float(frame),
                    "time": float(row["time"]),
                    "u": float(cx),
                    "v": float(cy),
                    "r": float(radius),
                    "mask_area": float(row.get("mask_area_px", 0.0) or 0.0),
                    "mask_left": float(row.get("bbox_x1", 0.0) or 0.0),
                    "mask_right": float(row.get("bbox_x2", 0.0) or 0.0),
                    "mask_top": float(row.get("bbox_y1", 0.0) or 0.0),
                    "mask_bottom": float(row.get("bbox_y2", 0.0) or 0.0),
                    "mask_width": bbox_w,
                    "mask_height": bbox_h,
                    "mask_size": 0.5 * (bbox_w + bbox_h),
                    "observation_conf": float(row.get("observation_conf", 0.0) or 0.0),
                    "radius_est_m": radius_estimates.get(frame) if radius_estimates else None,
                }
            )
    if not rows:
        raise RuntimeError(f"No valid object observation rows in {path}")
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



def load_observations_from_object_layer(path: Path, masks_dir: Path) -> list[dict[str, float]]:
    obs_rows = read_object_observations(path)
    silhouettes = read_mask_silhouettes(masks_dir)
    out: list[dict[str, float]] = []
    for row in obs_rows:
        frame = int(row["frame"])
        if frame not in silhouettes:
            continue
        merged = dict(row)
        merged.update(silhouettes[frame])
        out.append(merged)
    if not out:
        raise RuntimeError("No valid rows from object observation layer")
    return out



def read_contact_frames_from_candidates(path: Path, contact_type: str = "floor_contact_event") -> list[int]:
    frames: list[int] = []
    if not path.exists():
        return frames
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("contact_type") != contact_type:
                continue
            frames.append(int(float(row["frame"])))
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


def read_object_depth(path: Path) -> tuple[dict[int, float], dict[int, float]]:
    """Per-frame metric object depth (aligned) + per-frame confidence from
    run_depth_anything_v3. depth_conf folds in fit corr + extrapolation, so unreliable
    frames (e.g. the football ball extrapolating past the body) get ~0 weight."""
    depth: dict[int, float] = {}
    conf: dict[int, float] = {}
    if not path.exists():
        return depth, conf
    with path.open() as f:
        for row in csv.DictReader(f):
            val = row.get("object_z_aligned_m", "")
            if not val:
                continue
            try:
                z = float(val)
            except ValueError:
                continue
            if np.isfinite(z):
                fid = int(row["frame"])
                depth[fid] = z
                try:
                    conf[fid] = float(row.get("depth_conf", "") or 0.0)
                except ValueError:
                    conf[fid] = 0.0
    return depth, conf


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


def build_init_translations(
    obs_rows: list[dict[str, float]],
    camera: CameraModel,
    shape: SphereShape,
    depth_obs: np.ndarray | None = None,
) -> np.ndarray:
    """Initial 3D translations. Depth comes from the sphere size cue (Z = f*R/r)
    unless per-frame metric depth (`depth_obs`) is supplied, in which case that
    object-agnostic depth seeds Z and X,Y are back-projected from the center."""
    translations = []
    for idx, row in enumerate(obs_rows):
        if depth_obs is not None and np.isfinite(depth_obs[idx]):
            z = max(float(depth_obs[idx]), 1e-6)
            x = (row["u"] - camera.cx) * z / camera.fx
            y = (row["v"] - camera.cy) * z / camera.fy
        else:
            z = camera.fx * shape.radius_m / max(row["r"], 1e-6)
            x = (row["u"] - camera.cx) * z / camera.fx
            r_px = camera.fx * shape.radius_m / max(z, 1e-6)
            v_center_from_floor = camera.floor_v - r_px
            y = (v_center_from_floor - camera.cy) * z / camera.fy
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
    xy = flat_state[: num_frames * 2].reshape(num_frames, 2)
    ab = flat_state[num_frames * 2 :].reshape(len(segments), 2)
    t = np.zeros((num_frames, 3), dtype=np.float64)
    t[:, :2] = xy
    for seg_idx, seg in enumerate(segments):
        indices = seg["indices"]
        tau = seg["tau"]
        a, b = ab[seg_idx]
        t[indices, 2] = a * tau + b
    return t, ab


def pose_residuals(
    flat_state: np.ndarray,
    obs_rows: list[dict[str, float]],
    camera: CameraModel,
    shape: SphereShape,
    segments: list[dict[str, np.ndarray | int]],
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
    depth_obs: np.ndarray | None = None,
    depth_weight: float = 0.0,
    depth_conf: np.ndarray | None = None,
) -> np.ndarray:
    t, ab = unpack_state(flat_state, len(obs_rows), segments)
    residuals: list[float] = []

    for idx, row in enumerate(obs_rows):
        pred = shape.project(t[idx], camera)

        # Mask-chamfer and size residuals assume a sphere of known radius; they are
        # disabled (weight 0) in the depthv3 path and skipped entirely so rows without
        # silhouette/mask fields (object-agnostic inputs) don't KeyError. The center
        # reprojection below is shape-independent and always active.
        if mask_weight > 0.0:
            pred_contour = circle_contour_points(pred, num_points=24)
            obs_contour = row["contour_points"]
            residuals.extend((mask_weight * contour_chamfer_residuals(pred_contour, obs_contour, scale=12.0)).tolist())

        residuals.append(center_weight * (pred["u"] - row["u"]))
        residuals.append(center_weight * (pred["v"] - row["v"]))

        if size_weight > 0.0:
            residuals.append(size_weight * (pred["diameter_px"] - row["mask_width"]))
            residuals.append(size_weight * (pred["diameter_px"] - row["mask_height"]))
            residuals.append(size_weight * (pred["diameter_px"] - row["mask_size"]))
            residuals.append(size_weight * ((pred["area_px"] - row["mask_area"]) / 1000.0))

        # Per-frame metric depth observation (Depth Anything 3, aligned to GVHMR).
        # Scaled by per-frame depth_conf so low-confidence frames (anti-correlated fit or
        # object extrapolating past the body range) barely pull the depth.
        if depth_weight > 0.0 and depth_obs is not None and np.isfinite(depth_obs[idx]):
            w = depth_weight * (float(depth_conf[idx]) if depth_conf is not None else 1.0)
            residuals.append(w * (t[idx, 2] - float(depth_obs[idx])))

        if idx in contact_indices:
            residuals.append(contact_weight * ((pred["bottom_v"] - camera.floor_v) / 20.0))
        elif idx in weak_contact_indices:
            residuals.append(0.5 * contact_weight * ((pred["bottom_v"] - camera.floor_v) / 20.0))

        residuals.append(0.30 * max(0.0, 0.35 - t[idx, 2]))

    for idx in range(1, len(t) - 1):
        accel_xy = t[idx + 1, :2] - 2.0 * t[idx, :2] + t[idx - 1, :2]
        residuals.extend((temp_weight * accel_xy).tolist())

    for seg in segments:
        indices = seg["indices"]
        if len(indices) >= 3:
            zvals = t[indices, 2]
            accel_z = zvals[2:] - 2.0 * zvals[1:-1] + zvals[:-2]
            residuals.extend((z_temp_weight * accel_z).tolist())

    for seg_idx in range(len(segments) - 1):
        cur_seg = segments[seg_idx]
        next_seg = segments[seg_idx + 1]
        end_idx = int(cur_seg["end"])
        next_start_idx = int(next_seg["start"])
        residuals.append(z_boundary_weight * (t[next_start_idx, 2] - t[end_idx, 2]))
        residuals.append(z_slope_weight * (ab[seg_idx + 1, 0] - ab[seg_idx, 0]))

    if len(t) >= 2:
        residuals.extend((0.15 * (t[1, :2] - t[0, :2])).tolist())
        residuals.extend((0.15 * (t[-1, :2] - t[-2, :2])).tolist())

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
    parser.add_argument("--z-boundary-weight", type=float, default=0.10)
    parser.add_argument("--z-slope-weight", type=float, default=0.05)
    parser.add_argument("--contact-weight", type=float, default=10.0)
    parser.add_argument("--center-weight", type=float, default=0.04)
    parser.add_argument("--size-weight", type=float, default=0.02)
    parser.add_argument(
        "--depth-source",
        choices=["sphere", "depthv3"],
        default="sphere",
        help="sphere: legacy Z=f*R/r size cue (known sphere). depthv3: per-frame metric "
        "depth from Depth Anything 3 (object-agnostic), disabling the sphere size/mask terms.",
    )
    parser.add_argument("--depth-csv", type=Path, default=None,
                        help="object_depth.csv from run_depth_anything_v3 (default results/depth/object_depth.csv)")
    parser.add_argument("--depth-weight", type=float, default=1.0,
                        help="Weight on the DA3 metric-depth residual (depthv3 mode).")
    parser.add_argument(
        "--object-observation-csv",
        type=Path,
        default=None,
        help="Optional generic object observation table. If present, sharedcam uses it instead of the old basketball-only tracking csv.",
    )
    parser.add_argument(
        "--radius-estimates-csv",
        type=Path,
        default=None,
        help="Optional per-frame radius estimates from SAM 3D Objects. If provided, uses estimated radius instead of fixed --ball-radius-m.",
    )
    args = parser.parse_args()

    sample_dir = args.sample_dir
    results_dir = sample_dir / "results"
    # Keep the sphere baseline and the DA3 metric-depth result in separate dirs.
    out_dir = results_dir / ("pose6d_sharedcam_depthv3" if args.depth_source == "depthv3" else "pose6d_sharedcam")
    out_dir.mkdir(parents=True, exist_ok=True)

    object_obs_csv = args.object_observation_csv or (results_dir / "object_observations" / "object_observations.csv")
    radius_est_csv = args.radius_estimates_csv or (results_dir / "object_observations" / "radius_estimates.csv")

    radius_estimates = read_radius_estimates(radius_est_csv)
    if radius_estimates:
        print(f"Loaded {len(radius_estimates)} radius estimates from {radius_est_csv}")
        mean_radius = np.mean(list(radius_estimates.values()))
        print(f"  Mean estimated radius: {mean_radius:.6f} m (vs. default {args.ball_radius_m:.6f} m)")
    
    masks_dir = results_dir / "segmentation" / "masks"
    have_masks = masks_dir.exists() and any(masks_dir.glob("*_mask.png"))
    if object_obs_csv.exists():
        if have_masks:
            obs_rows = load_observations_from_object_layer(object_obs_csv, masks_dir)
        else:
            # Object-agnostic path (e.g. box-tracked objects with no masks): the
            # silhouette-dependent residuals are off in depthv3 mode anyway.
            if args.depth_source != "depthv3":
                raise RuntimeError(
                    f"No masks in {masks_dir}; sphere depth needs silhouettes. "
                    "Use --depth-source depthv3 for mask-free objects."
                )
            obs_rows = read_object_observations(object_obs_csv, read_radius_estimates(radius_est_csv))
    else:
        obs_rows = load_observations(results_dir / "tracking" / "ball_trajectory.csv", masks_dir)
    contact_candidate_csv = results_dir / "contact_candidates" / "contact_candidates_labeled.csv"
    contact_frames = read_contact_frames_from_candidates(contact_candidate_csv, contact_type="floor_contact_event")
    if not contact_frames:
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
    support = estimate_support_geometry_from_contacts(frames, obs_bottoms, contact_frames)
    camera.floor_v = support.floor_v
    shape = SphereShape(args.ball_radius_m)

    # Depth source: sphere size cue (default) or per-frame DA3 metric depth.
    depth_obs = None
    mask_weight = args.mask_weight
    size_weight = args.size_weight
    depth_weight = 0.0
    if args.depth_source == "depthv3":
        depth_csv = args.depth_csv or (results_dir / "depth" / "object_depth.csv")
        depth_by_frame, conf_by_frame = read_object_depth(depth_csv)
        if not depth_by_frame:
            raise RuntimeError(
                f"--depth-source depthv3 but no usable depth in {depth_csv}. "
                "Run scripts.shared.depth.run_depth_anything_v3 first."
            )
        depth_obs = np.array([depth_by_frame.get(int(r["frame"]), np.nan) for r in obs_rows], dtype=np.float64)
        # per-frame confidence (fit corr x extrapolation penalty); unreliable depth -> ~0 weight
        depth_conf_obs = np.array([conf_by_frame.get(int(r["frame"]), 0.0) for r in obs_rows], dtype=np.float64)
        n_have = int(np.isfinite(depth_obs).sum())
        mean_conf = float(np.nanmean(depth_conf_obs[np.isfinite(depth_obs)])) if n_have else 0.0
        # Disable sphere-specific terms; rely on DA3 depth + center + smoothness + contact.
        mask_weight = 0.0
        size_weight = 0.0
        depth_weight = args.depth_weight
        print(f"depth_source: depthv3  ({n_have}/{len(obs_rows)} frames with metric depth, "
              f"depth_weight={depth_weight}, mean_conf={mean_conf:.3f})")
    else:
        depth_conf_obs = None
        print("depth_source: sphere (Z = f*R/r)")

    init_t = build_init_translations(obs_rows, camera, shape, depth_obs=depth_obs)
    segments = build_segments(len(obs_rows), contact_indices, times)
    init_ab = fit_segment_lines(init_t, segments)
    init_state = np.concatenate([init_t[:, :2].reshape(-1), init_ab.reshape(-1)], axis=0)

    result = least_squares(
        pose_residuals,
        init_state,
        method="trf",
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=250,
        args=(
            obs_rows,
            camera,
            shape,
            segments,
            contact_indices,
            audio_contact_indices,
            mask_weight,
            args.temp_weight,
            args.z_temp_weight,
            args.z_boundary_weight,
            args.z_slope_weight,
            args.contact_weight,
            args.center_weight,
            size_weight,
            depth_obs,
            depth_weight,
            depth_conf_obs,
        ),
    )

    opt_t, _opt_ab = unpack_state(result.x, len(obs_rows), segments)
    opt_t[:, 2] = np.maximum(opt_t[:, 2], 0.35)
    depth_outlier_mask = np.zeros(len(obs_rows), dtype=bool)

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
                "depth_outlier_corrected": int(depth_outlier_mask[idx]),
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
            "depth_outlier_corrected",
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

    write_support_geometry_json(out_dir / "support_geometry.json", support)
    plot_outputs(out_dir, times, init_t, opt_t, np.array(residual_px, dtype=np.float64))

    num_depth_outlier_corrected = int(depth_outlier_mask.sum())
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
    print(f"support_type: {support.support_type}  source: {support.source}  confidence: {support.confidence:.3f}")
    print(f"optimizer_cost: {float(result.cost):.6f}")
    print(f"depth_outlier_corrected_frames: {num_depth_outlier_corrected}")


if __name__ == "__main__":
    main()
