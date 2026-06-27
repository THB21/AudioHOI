#!/usr/bin/env python3
"""Opening-frame M18 correction optimized for 2D video overlay fit."""
from __future__ import annotations

import argparse
import csv
import math
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

REPO = Path(__file__).resolve().parents[5]
GENERIC_RENDER = REPO / "scripts" / "shared" / "generic_contact_pipeline" / "components" / "render"
for p in (REPO, GENERIC_RENDER):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import fit_mug_articraft_keyframe_pose as base  # noqa: E402
import render_mug_articraft_rigid_mesh_vlm as rigid  # noqa: E402


CONTACT_EVENTS = {
    "hand_handle_grasp": 1.0,
    "hand_handle_grasp_depth_misaligned": 0.55,
    "handle_contact_point_misaligned": 0.35,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def ff(row: dict[str, str] | None, key: str, default: float = math.nan) -> float:
    if not row:
        return default
    try:
        return float(row.get(key, "") or default)
    except (TypeError, ValueError):
        return default


def read_pose(path: Path) -> dict[int, dict[str, float]]:
    out = {}
    for r in read_csv(path):
        fr = int(float(r["frame"]))
        row = {
            "frame": fr,
            "time": ff(r, "time", (fr - 1) / 24.0),
            "x": ff(r, "x"),
            "y": ff(r, "y"),
            "z": ff(r, "z"),
            "yaw": ff(r, "yaw"),
            "pitch": ff(r, "pitch"),
            "roll": ff(r, "roll"),
            "scale": ff(r, "scale", 1.0),
        }
        row["yaw_deg"] = ff(r, "yaw_deg", math.degrees(row["yaw"]))
        row["pitch_deg"] = ff(r, "pitch_deg", math.degrees(row["pitch"]))
        row["roll_deg"] = ff(r, "roll_deg", math.degrees(row["roll"]))
        out[fr] = row
    return out


def read_phase(path: Path, col: str) -> dict[int, float]:
    out = {}
    for r in read_csv(path):
        fr = int(float(r["frame"]))
        out[fr] = ff(r, col, ff(r, "m17_phase_rad", ff(r, "phase_rad", 0.0)))
    return out


def load_k(sample: Path) -> np.ndarray:
    with (sample / "results" / "gvhmr" / "result.pkl").open("rb") as f:
        data = pickle.load(f)
    K = np.asarray(data["K_fullimg"], dtype=float)
    return K[0] if K.ndim == 3 else K


def pose_vec(row: dict[str, float], delta: np.ndarray | None = None) -> np.ndarray:
    out = np.array([row[k] for k in ["x", "y", "z", "yaw", "pitch", "roll", "scale"]], dtype=float)
    if delta is not None:
        out[:3] += delta
    return out


def project(points: np.ndarray, K: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    z = np.maximum(pts[:, 2], 1e-6)
    return np.column_stack([
        K[0, 0] * pts[:, 0] / z + K[0, 2],
        K[1, 1] * pts[:, 1] / z + K[1, 2],
    ])


def body_uv_metrics(
    pose: dict[str, float],
    phase: float,
    delta: np.ndarray,
    body_vertices: np.ndarray,
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    cam = base.transform(pose_vec(pose, delta), body_vertices @ rigid.rot_y(phase).T)
    uv = project(cam, K)
    bbox = np.array([uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()], dtype=float)
    center = np.array([(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5], dtype=float)
    return center, bbox


def contact_uv(
    pose: dict[str, float],
    phase: float,
    delta: np.ndarray,
    contact_row: dict[str, str],
    K: np.ndarray,
) -> np.ndarray:
    local = np.array([
        ff(contact_row, "mug_local_x"),
        ff(contact_row, "mug_local_y"),
        ff(contact_row, "mug_local_z"),
    ], dtype=float)
    cam = base.transform(pose_vec(pose, delta), local[None, :] @ rigid.rot_y(phase).T)
    return project(cam, K)[0]


def obs_body(row: dict[str, str]) -> tuple[np.ndarray, np.ndarray] | None:
    if str(row.get("body_available", "")).strip() != "1":
        return None
    center = np.array([ff(row, "body_center_x"), ff(row, "body_center_y")], dtype=float)
    bbox = np.array([
        ff(row, "body_bbox_x1"),
        ff(row, "body_bbox_y1"),
        ff(row, "body_bbox_x2"),
        ff(row, "body_bbox_y2"),
    ], dtype=float)
    if not (np.isfinite(center).all() and np.isfinite(bbox).all()):
        return None
    return center, bbox


def optimize(
    frames: list[int],
    poses: dict[int, dict[str, float]],
    phases: dict[int, float],
    observations: dict[int, dict[str, str]],
    contacts: dict[int, dict[str, str]],
    body_vertices: np.ndarray,
    K: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    n = len(frames)
    frame_to_idx = {fr: i for i, fr in enumerate(frames)}
    fit_frames = [fr for fr in frames if args.start_frame <= fr <= args.end_frame]

    def residual(flat: np.ndarray) -> np.ndarray:
        deltas = flat.reshape(n, 3)
        res: list[float] = []
        keep_scale = math.sqrt(args.w_keep) / max(args.max_xy_delta_m, 1e-6)
        z_keep_scale = math.sqrt(args.w_z_keep) / max(args.max_z_delta_m, 1e-6)
        res.extend((keep_scale * deltas[:, :2]).reshape(-1).tolist())
        res.extend((z_keep_scale * deltas[:, 2]).reshape(-1).tolist())

        for fr in fit_frames:
            obs = obs_body(observations.get(fr, {}))
            if obs is not None:
                i = frame_to_idx[fr]
                center_obs, bbox_obs = obs
                center_pred, bbox_pred = body_uv_metrics(poses[fr], phases[fr], deltas[i], body_vertices, K)
                conf = max(0.15, min(1.0, ff(observations.get(fr, {}), "observation_conf", 0.6)))
                res.extend((math.sqrt(args.w_body_center * conf) * (center_pred - center_obs) / args.sigma_center_px).tolist())
                res.extend((math.sqrt(args.w_body_bbox * conf) * (bbox_pred - bbox_obs) / args.sigma_bbox_px).tolist())

            c = contacts.get(fr, {})
            event = str(c.get("object_contact_event", "")).strip()
            if (
                str(c.get("nearest_articraft_part", "")).strip() == "handle_loop"
                and event in CONTACT_EVENTS
            ):
                i = frame_to_idx[fr]
                pred = contact_uv(poses[fr], phases[fr], deltas[i], c, K)
                target = np.array([ff(c, "contact_u"), ff(c, "contact_v")], dtype=float)
                if np.isfinite(target).all():
                    wt = CONTACT_EVENTS[event] * max(0.25, min(1.0, ff(c, "contact_conf", 0.5)))
                    res.extend((math.sqrt(args.w_contact * wt) * (pred - target) / args.sigma_contact_px).tolist())

        smooth_scale = math.sqrt(args.w_smooth) / max(args.max_xy_delta_m, 1e-6)
        for i in range(1, n - 1):
            acc = deltas[i + 1] - 2.0 * deltas[i] + deltas[i - 1]
            res.extend((smooth_scale * acc).tolist())

        post_start = args.end_frame + args.fade_frames
        post_scale = math.sqrt(args.w_post) / max(args.max_xy_delta_m, 1e-6)
        for i, fr in enumerate(frames):
            if fr >= post_start:
                res.extend((post_scale * deltas[i]).tolist())
        return np.asarray(res, dtype=float)

    x0 = np.zeros((n, 3), dtype=float).reshape(-1)
    lb = np.tile(np.array([-args.max_xy_delta_m, -args.max_xy_delta_m, -args.max_z_delta_m]), n)
    ub = np.tile(np.array([args.max_xy_delta_m, args.max_xy_delta_m, args.max_z_delta_m]), n)
    opt = least_squares(residual, x0, bounds=(lb, ub), loss="soft_l1", f_scale=1.0, max_nfev=220)
    return opt.x.reshape(n, 3)


def run(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    sample = args.sample_dir.resolve()
    out_dir = args.out_dir or sample / "results" / "mug_m18_opening_2d_video_correction"
    out_dir.mkdir(parents=True, exist_ok=True)

    pose_csv = args.pose_csv or sample / "results" / "pipe" / "anchored_pose_obs.csv"
    phase_csv = args.phase_csv or sample / "results" / "renders" / "M17_phase_corrected" / "corrected_handle_phase.csv"
    obs_csv = args.object_observation_csv or sample / "results" / "object_observations" / "object_observations.csv"
    contact_csv = args.contact_csv or sample / "results" / "mug_articraft_contact_points" / "mug_articraft_contact_points.csv"

    poses = read_pose(pose_csv)
    phases = read_phase(phase_csv, args.phase_col)
    observations = {int(float(r["frame"])): r for r in read_csv(obs_csv)}
    contacts = {int(float(r["frame"])): r for r in read_csv(contact_csv)}
    meshes = rigid.load_articraft_meshes(args.mesh_root)
    body_vertices = meshes["body_shell"][0]
    K = load_k(sample)
    frames = sorted(set(poses) & set(phases))
    deltas = optimize(frames, poses, phases, observations, contacts, body_vertices, K, args)
    frame_to_idx = {fr: i for i, fr in enumerate(frames)}

    pose_out = out_dir / "mug_m18_opening_2d_video_pose.csv"
    traj_out = out_dir / "mug_m18_opening_2d_video_trajectory.csv"
    report_out = out_dir / "mug_m18_opening_2d_video_report.csv"
    summary_out = out_dir / "mug_m18_opening_2d_video_summary.txt"

    pose_rows = []
    traj_rows = []
    report_rows = []
    for fr in frames:
        p = poses[fr]
        d = deltas[frame_to_idx[fr]]
        obs = obs_body(observations.get(fr, {}))
        body_before = body_after = math.nan
        contact_before = contact_after = math.nan
        if args.start_frame <= fr <= args.end_frame and obs is not None:
            center_obs, bbox_obs = obs
            c0, b0 = body_uv_metrics(p, phases[fr], np.zeros(3), body_vertices, K)
            c1, b1 = body_uv_metrics(p, phases[fr], d, body_vertices, K)
            body_before = float(np.linalg.norm(c0 - center_obs))
            body_after = float(np.linalg.norm(c1 - center_obs))
        c = contacts.get(fr, {})
        event = str(c.get("object_contact_event", "")).strip()
        if args.start_frame <= fr <= args.end_frame and event in CONTACT_EVENTS and str(c.get("nearest_articraft_part", "")).strip() == "handle_loop":
            target = np.array([ff(c, "contact_u"), ff(c, "contact_v")], dtype=float)
            if np.isfinite(target).all():
                contact_before = float(np.linalg.norm(contact_uv(p, phases[fr], np.zeros(3), c, K) - target))
                contact_after = float(np.linalg.norm(contact_uv(p, phases[fr], d, c, K) - target))
        corrected = dict(p)
        corrected["x"] = p["x"] + float(d[0])
        corrected["y"] = p["y"] + float(d[1])
        corrected["z"] = p["z"] + float(d[2])
        pose_rows.append({
            "frame": fr,
            "time": f"{p['time']:.6f}",
            "x": f"{corrected['x']:.9f}",
            "y": f"{corrected['y']:.9f}",
            "z": f"{corrected['z']:.9f}",
            "yaw": f"{p['yaw']:.9f}",
            "yaw_deg": f"{p['yaw_deg']:.6f}",
            "pitch": f"{p['pitch']:.9f}",
            "pitch_deg": f"{p['pitch_deg']:.6f}",
            "roll": f"{p['roll']:.9f}",
            "roll_deg": f"{p['roll_deg']:.6f}",
            "scale": f"{p['scale']:.9f}",
        })
        row = {
            "frame": fr,
            "time": f"{p['time']:.6f}",
            "delta_x": f"{d[0]:.9f}",
            "delta_y": f"{d[1]:.9f}",
            "delta_z": f"{d[2]:.9f}",
            "delta_norm_m": f"{float(np.linalg.norm(d)):.9f}",
            "body_center_err_before_px": f"{body_before:.6f}" if np.isfinite(body_before) else "",
            "body_center_err_after_px": f"{body_after:.6f}" if np.isfinite(body_after) else "",
            "contact_err_before_px": f"{contact_before:.6f}" if np.isfinite(contact_before) else "",
            "contact_err_after_px": f"{contact_after:.6f}" if np.isfinite(contact_after) else "",
        }
        traj_rows.append(row)
        if np.isfinite(body_before) or np.isfinite(contact_before):
            report_rows.append(row)

    fields = ["frame", "time", "x", "y", "z", "yaw", "yaw_deg", "pitch", "pitch_deg", "roll", "roll_deg", "scale"]
    with pose_out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(pose_rows)
    with traj_out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(traj_rows[0].keys()))
        w.writeheader()
        w.writerows(traj_rows)
    with report_out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(traj_rows[0].keys()))
        w.writeheader()
        w.writerows(report_rows)

    body_b = [ff(r, "body_center_err_before_px") for r in report_rows]
    body_a = [ff(r, "body_center_err_after_px") for r in report_rows]
    cont_b = [ff(r, "contact_err_before_px") for r in report_rows]
    cont_a = [ff(r, "contact_err_after_px") for r in report_rows]
    body_b = [v for v in body_b if np.isfinite(v)]
    body_a = [v for v in body_a if np.isfinite(v)]
    cont_b = [v for v in cont_b if np.isfinite(v)]
    cont_a = [v for v in cont_a if np.isfinite(v)]
    norms = np.linalg.norm(deltas, axis=1)
    with summary_out.open("w") as f:
        f.write(f"num_frames: {len(frames)}\n")
        f.write(f"start_frame: {args.start_frame}\n")
        f.write(f"end_frame: {args.end_frame}\n")
        if body_b:
            f.write(f"mean_body_center_err_before_px: {float(np.mean(body_b)):.3f}\n")
            f.write(f"mean_body_center_err_after_px: {float(np.mean(body_a)):.3f}\n")
            f.write(f"max_body_center_err_after_px: {float(np.max(body_a)):.3f}\n")
        if cont_b:
            f.write(f"mean_contact_err_before_px: {float(np.mean(cont_b)):.3f}\n")
            f.write(f"mean_contact_err_after_px: {float(np.mean(cont_a)):.3f}\n")
            f.write(f"max_contact_err_after_px: {float(np.max(cont_a)):.3f}\n")
        f.write(f"mean_delta_from_m18_m: {float(np.mean(norms)):.6f}\n")
        f.write(f"max_delta_from_m18_m: {float(np.max(norms)):.6f}\n")
        for name in [
            "w_body_center", "w_body_bbox", "w_contact", "w_keep", "w_z_keep",
            "w_smooth", "w_post", "sigma_center_px", "sigma_bbox_px", "sigma_contact_px",
            "max_xy_delta_m", "max_z_delta_m",
        ]:
            f.write(f"{name}: {getattr(args, name)}\n")
    return pose_out, report_out, summary_out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir", type=Path, default=Path("samples_known_object/02_mug"))
    ap.add_argument("--pose-csv", type=Path, default=None)
    ap.add_argument("--phase-csv", type=Path, default=None)
    ap.add_argument("--object-observation-csv", type=Path, default=None)
    ap.add_argument("--contact-csv", type=Path, default=None)
    ap.add_argument("--mesh-root", type=Path, default=Path("samples_known_object/02_mug/articraft/materialized_mug_mesh"))
    ap.add_argument("--phase-col", type=str, default="m17_phase_rad")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--start-frame", type=int, default=1)
    ap.add_argument("--end-frame", type=int, default=24)
    ap.add_argument("--fade-frames", type=int, default=14)
    ap.add_argument("--w-body-center", type=float, default=6.0)
    ap.add_argument("--w-body-bbox", type=float, default=3.0)
    ap.add_argument("--w-contact", type=float, default=0.75)
    ap.add_argument("--w-keep", type=float, default=1.5)
    ap.add_argument("--w-z-keep", type=float, default=8.0)
    ap.add_argument("--w-smooth", type=float, default=8.0)
    ap.add_argument("--w-post", type=float, default=14.0)
    ap.add_argument("--sigma-center-px", type=float, default=5.0)
    ap.add_argument("--sigma-bbox-px", type=float, default=7.5)
    ap.add_argument("--sigma-contact-px", type=float, default=12.0)
    ap.add_argument("--max-xy-delta-m", type=float, default=0.075)
    ap.add_argument("--max-z-delta-m", type=float, default=0.035)
    args = ap.parse_args()
    pose, report, summary = run(args)
    print(f"[m18_opening_2d] pose -> {pose}")
    print(f"[m18_opening_2d] report -> {report}")
    print(f"[m18_opening_2d] summary -> {summary}")


if __name__ == "__main__":
    main()
