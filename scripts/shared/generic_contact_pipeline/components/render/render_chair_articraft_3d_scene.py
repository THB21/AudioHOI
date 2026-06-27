#!/usr/bin/env python3
"""Render final chair articulated 6D pose in camera 3D and side Y-Z views."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import pickle
import shutil
import subprocess
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
from scipy.spatial.transform import Rotation

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

import smplx


BODY_EDGES = [
    (0, 1), (1, 4), (4, 7), (7, 10),
    (0, 2), (2, 5), (5, 8), (8, 11),
    (0, 3), (3, 6), (6, 9), (9, 12),
    (12, 13), (13, 16), (16, 18), (18, 20),
    (12, 14), (14, 17), (17, 19), (19, 21),
    (12, 15),
]

SEGMENT_COLORS = {
    "orange_front_leg": "#f5a623",
    "blue_rear_leg": "#2997d8",
    "top_rail_observation": "#f03020",
    "seat_front_observation": "#d84cff",
    "seat_depth_line": "#b97cff",
    "board_bottom_edge": "#e36b2c",
    "front_floor_support": "#d6d600",
    "rear_floor_support": "#a0b800",
    "seat_hinge_observation": "#d84cff",
}

SEGMENT_BGR = {
    "orange_front_leg": (0, 165, 255),
    "blue_rear_leg": (255, 160, 0),
    "top_rail_observation": (0, 0, 255),
    "seat_front_observation": (255, 0, 255),
    "seat_depth_line": (180, 120, 255),
    "board_bottom_edge": (0, 100, 255),
    "front_floor_support": (0, 220, 220),
    "rear_floor_support": (0, 180, 170),
    "seat_hinge_observation": (255, 0, 255),
}


def import_stage3():
    path = Path(__file__).resolve().parent / "fit_chair_metric_6d_pose.py"
    spec = importlib.util.spec_from_file_location("chair_fit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fit = import_stage3()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows in {path}")
    return rows


def ff(row: dict[str, str], key: str, default: float = math.nan) -> float:
    val = row.get(key, "")
    if val in ("", None):
        return default
    return float(val)


def pose_params(row: dict[str, str]) -> np.ndarray:
    keys = ["rx_delta", "ry_delta", "rz_delta", "tx", "ty", "tz", "rear_joint_angle", "seat_joint_angle"]
    return np.asarray([ff(row, k, 0.0) for k in keys], dtype=float)


def rotation_matrix(params: np.ndarray) -> np.ndarray:
    return Rotation.from_rotvec(params[:3]).as_matrix() @ fit.BASE_LOCAL_TO_CAM


def euler_xyz_deg(params: np.ndarray) -> np.ndarray:
    # Camera-frame object orientation. Report as roll/pitch/yaw for 6D auditing.
    return Rotation.from_matrix(rotation_matrix(params)).as_euler("xyz", degrees=True)


def rotate_x_about(points: np.ndarray, origin: np.ndarray, angle: float) -> np.ndarray:
    rot = Rotation.from_rotvec(np.array([float(angle), 0.0, 0.0])).as_matrix()
    return (points - origin) @ rot.T + origin


def local_rear_origin(segs: dict[str, tuple[str, str, np.ndarray, np.ndarray]]) -> np.ndarray:
    pts = []
    for sid in ("rear_leg_left", "rear_leg_right"):
        if sid in segs:
            pts.append(segs[sid][2])
    if pts:
        return np.mean(np.vstack(pts), axis=0)
    return np.array([0.0, 0.040, 0.548], dtype=float)


def local_seat_origin(segs: dict[str, tuple[str, str, np.ndarray, np.ndarray]]) -> np.ndarray:
    if "seat_hinge_edge" in segs:
        _part, _role, a, b = segs["seat_hinge_edge"]
        return (a + b) * 0.5
    return np.array([0.0, 0.090, 0.440], dtype=float)


def articulate_segment_points(
    sid: str,
    part: str,
    params: np.ndarray,
    pts_local: np.ndarray,
    segs: dict[str, tuple[str, str, np.ndarray, np.ndarray]],
) -> np.ndarray:
    rear_origin = local_rear_origin(segs)
    seat_origin = local_seat_origin(segs)
    rear_angle = float(params[6]) if len(params) > 6 else 0.0
    seat_angle = float(params[7]) if len(params) > 7 else 0.0
    if part == "rear_leg" or part == "rear_stretcher" or sid in {"rear_feet_line", "rear_lower_stretcher"}:
        return rotate_x_about(pts_local, rear_origin, rear_angle)
    if part == "side_stretcher" or "side_lower_stretcher" in sid:
        out = pts_local.copy()
        rear_idx = int(np.argmax(out[:, 1]))
        out[rear_idx : rear_idx + 1] = rotate_x_about(out[rear_idx : rear_idx + 1], rear_origin, rear_angle)
        return out
    if part in {"seat", "seat_slat"} or sid == "seat_hinge_edge":
        return rotate_x_about(pts_local, seat_origin, -seat_angle)
    return pts_local


def chair_segments_cam(params: np.ndarray, segs: dict[str, tuple[str, str, np.ndarray, np.ndarray]]) -> list[tuple[str, str, np.ndarray]]:
    out = []
    for sid, (part, role, a, b) in segs.items():
        local = articulate_segment_points(sid, part, params, np.vstack([a, b]), segs)
        cam = fit.transform(params, local)
        out.append((sid, role, cam))
    return out


def project_cam(points_cam: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points_cam, dtype=float)
    z = pts[:, 2]
    valid = np.isfinite(pts).all(axis=1) & (z > 1e-5)
    zc = np.maximum(z, 1e-5)
    uv = np.column_stack(
        [
            K[0, 0] * pts[:, 0] / zc + K[0, 2],
            K[1, 1] * pts[:, 1] / zc + K[1, 2],
        ]
    )
    return uv, valid


def cam_to_worldlike(points: np.ndarray) -> np.ndarray:
    out = points.copy()
    out[..., 1] *= -1.0
    return out


def figure_to_bgr(fig: plt.Figure) -> np.ndarray:
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)


def finalize_video(tmp: Path, out: Path) -> None:
    ffmpeg = shutil.which("ffmpeg") or "/usr/bin/ffmpeg"
    try:
        subprocess.run([ffmpeg, "-y", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        subprocess.run([ffmpeg, "-y", "-i", str(tmp), "-c:v", "libopenh264", "-pix_fmt", "yuv420p", str(out)], check=True, capture_output=True, text=True)
    tmp.unlink(missing_ok=True)


def read_human_result(path: Path) -> dict[str, np.ndarray]:
    with path.open("rb") as f:
        data = pickle.load(f)
    params = data["smpl_params_incam"]
    return {
        "body_pose": np.asarray(params["body_pose"], dtype=np.float32),
        "betas": np.asarray(params["betas"], dtype=np.float32),
        "global_orient": np.asarray(params["global_orient"], dtype=np.float32),
        "transl": np.asarray(params["transl"], dtype=np.float32),
        "K_fullimg": np.asarray(data["K_fullimg"], dtype=np.float32),
    }


def build_body_outputs(body_models_root: Path, human: dict[str, np.ndarray], vertex_stride: int) -> tuple[np.ndarray, np.ndarray]:
    model = smplx.create(
        str(body_models_root),
        model_type="smplx",
        gender="neutral",
        ext="npz",
        use_pca=False,
        flat_hand_mean=True,
        num_betas=10,
        batch_size=human["transl"].shape[0],
    )
    with torch.inference_mode():
        output = model(
            body_pose=torch.from_numpy(human["body_pose"]),
            betas=torch.from_numpy(human["betas"]),
            global_orient=torch.from_numpy(human["global_orient"]),
            transl=torch.from_numpy(human["transl"]),
            return_verts=True,
        )
    joints = output.joints.detach().cpu().numpy().astype(np.float32)
    verts = output.vertices.detach().cpu().numpy().astype(np.float32)[:, ::vertex_stride, :]
    return joints, verts


def palm_centers(joints_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left_ids = [20, 25, 28, 31, 34]
    right_ids = [21, 40, 43, 46, 49]
    return joints_cam[left_ids].mean(axis=0), joints_cam[right_ids].mean(axis=0)


def load_contacts(path: Path) -> dict[int, list[dict[str, str]]]:
    if not path.exists():
        return {}
    out: dict[int, list[dict[str, str]]] = {}
    for row in read_rows(path):
        if row.get("contact_active") != "1":
            continue
        out.setdefault(int(row["frame"]), []).append(row)
    return out


def contact_anchor_points(params: np.ndarray, rows: list[dict[str, str]]) -> list[tuple[str, np.ndarray]]:
    pts = []
    for row in rows:
        local = np.asarray([ff(row, "chair_local_x"), ff(row, "chair_local_y"), ff(row, "chair_local_z")], dtype=float)
        if not np.all(np.isfinite(local)):
            continue
        cam = fit.transform(params, local[None, :])[0]
        pts.append((row.get("hand_side", ""), cam))
    return pts


def contact_gap_entries(params: np.ndarray, rows: list[dict[str, str]], joints_frame: np.ndarray | None) -> list[tuple[str, np.ndarray, np.ndarray, float]]:
    if joints_frame is None:
        return []
    left, right = palm_centers(joints_frame)
    hand_by_side = {"left_hand": left, "right_hand": right}
    out = []
    for hand_side, anchor in contact_anchor_points(params, rows):
        hand = hand_by_side.get(hand_side)
        if hand is None:
            continue
        out.append((hand_side, anchor, hand, float(np.linalg.norm(hand - anchor))))
    return out


def frame_path(sample: Path, frame: int) -> Path:
    for name in (f"{frame:05d}.png", f"{frame:06d}.png", f"{frame:04d}.png"):
        p = sample / "frames" / name
        if p.exists():
            return p
    raise FileNotFoundError(f"No extracted frame for frame {frame} under {sample / 'frames'}")


def draw_text(img: np.ndarray, text: str, xy: tuple[int, int], color: tuple[int, int, int], scale: float = 0.48) -> None:
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (20, 20, 20), 3, cv2.LINE_AA)
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_line_2d(img: np.ndarray, a: np.ndarray, b: np.ndarray, color: tuple[int, int, int], thickness: int = 3) -> None:
    p0 = tuple(np.round(a).astype(int))
    p1 = tuple(np.round(b).astype(int))
    cv2.line(img, p0, p1, (20, 20, 20), thickness + 3, cv2.LINE_AA)
    cv2.line(img, p0, p1, color, thickness, cv2.LINE_AA)


def draw_chair_projection_2d(
    img: np.ndarray,
    params: np.ndarray,
    segs: dict[str, tuple[str, str, np.ndarray, np.ndarray]],
    contacts: list[dict[str, str]],
    K: np.ndarray,
) -> None:
    for _, role, cam in chair_segments_cam(params, segs):
        color = SEGMENT_BGR.get(role)
        if color is None:
            continue
        uv, valid = project_cam(cam, K)
        if not np.all(valid):
            continue
        thick = 4 if "leg" in role or "rail" in role else 3
        draw_line_2d(img, uv[0], uv[1], color, thick)
        for pt in uv:
            cv2.circle(img, tuple(np.round(pt).astype(int)), 5, (20, 20, 20), -1, cv2.LINE_AA)
            cv2.circle(img, tuple(np.round(pt).astype(int)), 3, color, -1, cv2.LINE_AA)

    for hand_side, anchor in contact_anchor_points(params, contacts):
        uv, valid = project_cam(anchor[None, :], K)
        if valid[0]:
            p = tuple(np.round(uv[0]).astype(int))
            cv2.circle(img, p, 10, (20, 20, 20), -1, cv2.LINE_AA)
            cv2.circle(img, p, 7, (0, 255, 255), -1, cv2.LINE_AA)
            draw_text(img, hand_side.replace("_hand", ""), (p[0] + 7, p[1] - 7), (0, 255, 255), 0.42)


def draw_human_projection_2d(
    img: np.ndarray,
    joints_frame: np.ndarray,
    K: np.ndarray,
    contacts: list[dict[str, str]],
    params: np.ndarray,
) -> None:
    body_uv, body_valid = project_cam(joints_frame[:22], K)
    for a, b in BODY_EDGES:
        if a < len(body_valid) and b < len(body_valid) and body_valid[a] and body_valid[b]:
            cv2.line(
                img,
                tuple(np.round(body_uv[a]).astype(int)),
                tuple(np.round(body_uv[b]).astype(int)),
                (210, 120, 255),
                2,
                cv2.LINE_AA,
            )
    for uv, valid in zip(body_uv, body_valid):
        if valid:
            cv2.circle(img, tuple(np.round(uv).astype(int)), 3, (90, 70, 180), -1, cv2.LINE_AA)

    left, right = palm_centers(joints_frame)
    palm_by_side = [("left_hand", left, (70, 180, 70)), ("right_hand", right, (230, 200, 70))]
    for name, palm, color in palm_by_side:
        uv, valid = project_cam(palm[None, :], K)
        if valid[0]:
            p = tuple(np.round(uv[0]).astype(int))
            cv2.circle(img, p, 11, (20, 20, 20), -1, cv2.LINE_AA)
            cv2.circle(img, p, 8, color, -1, cv2.LINE_AA)
            draw_text(img, name.replace("_hand", " palm"), (p[0] + 7, p[1] + 15), color, 0.42)

    for hand_side, anchor, hand, gap in contact_gap_entries(params, contacts, joints_frame):
        auv, av = project_cam(anchor[None, :], K)
        huv, hv = project_cam(hand[None, :], K)
        if av[0] and hv[0]:
            cv2.line(
                img,
                tuple(np.round(auv[0]).astype(int)),
                tuple(np.round(huv[0]).astype(int)),
                (15, 15, 15),
                2,
                cv2.LINE_AA,
            )
            mid = ((auv[0] + huv[0]) * 0.5).astype(int)
            draw_text(img, f"{gap:.2f}m", (int(mid[0]) + 4, int(mid[1]) - 4), (255, 255, 255), 0.4)


def collect_bounds(
    pose_rows: list[dict[str, str]],
    segs: dict[str, tuple[str, str, np.ndarray, np.ndarray]],
    joints: np.ndarray | None,
    verts: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    pts = []
    for row in pose_rows:
        params = pose_params(row)
        for _, _, cam in chair_segments_cam(params, segs):
            pts.append(cam)
    if joints is not None:
        pts.append(joints[:, :22, :].reshape(-1, 3))
    if verts is not None:
        pts.append(verts.reshape(-1, 3))
    all_cam = np.vstack(pts)
    world = cam_to_worldlike(all_cam)
    return world.min(axis=0), world.max(axis=0)


def draw_chair_3d(ax, params: np.ndarray, segs: dict[str, tuple[str, str, np.ndarray, np.ndarray]], contacts: list[dict[str, str]]) -> None:
    for _, role, cam in chair_segments_cam(params, segs):
        w = cam_to_worldlike(cam)
        color = SEGMENT_COLORS.get(role, "#444444")
        ax.plot(w[:, 0], w[:, 2], w[:, 1], color=color, linewidth=3.0 if "leg" in role or "rail" in role else 2.2, alpha=0.96)
        ax.scatter(w[:, 0], w[:, 2], w[:, 1], s=18, color=color, edgecolors="#202020", linewidths=0.4, depthshade=False)
    for hand, cam in contact_anchor_points(params, contacts):
        w = cam_to_worldlike(cam[None, :])[0]
        ax.scatter([w[0]], [w[2]], [w[1]], s=90, color="#ffea00", edgecolors="#202020", linewidths=0.8, depthshade=False)
        ax.text(w[0], w[2], w[1], hand.replace("_hand", ""), color="#202020", fontsize=8)
    draw_pose_axes_3d(ax, params)


def draw_pose_axes_3d(ax, params: np.ndarray, axis_len: float = 0.18) -> None:
    origin = params[3:6]
    R = rotation_matrix(params)
    axes = [
        ("+X", np.array([axis_len, 0.0, 0.0]), "#e53935"),
        ("+Y", np.array([0.0, axis_len, 0.0]), "#43a047"),
        ("+Z", np.array([0.0, 0.0, axis_len]), "#1e88e5"),
    ]
    ow = cam_to_worldlike(origin[None, :])[0]
    for label, local_axis, color in axes:
        end = origin + R @ local_axis
        ew = cam_to_worldlike(end[None, :])[0]
        ax.plot([ow[0], ew[0]], [ow[2], ew[2]], [ow[1], ew[1]], color=color, linewidth=3.0, alpha=0.95)
        ax.text(ew[0], ew[2], ew[1], label, color=color, fontsize=8)


def draw_human_3d(ax, joints_frame: np.ndarray, verts_frame: np.ndarray | None, contacts: list[dict[str, str]]) -> None:
    if verts_frame is not None:
        v = cam_to_worldlike(verts_frame)
        ax.scatter(v[:, 0], v[:, 2], v[:, 1], s=4, color="#9c89f2", alpha=0.18, linewidths=0, depthshade=False)
    body = cam_to_worldlike(joints_frame[:22])
    for a, b in BODY_EDGES:
        seg = body[[a, b]]
        ax.plot(seg[:, 0], seg[:, 2], seg[:, 1], color="#6b54d2", linewidth=2.2, alpha=0.92)
    ax.scatter(body[:, 0], body[:, 2], body[:, 1], s=14, color="#e4dbff", edgecolors="#6b54d2", linewidths=0.4, depthshade=False)
    left, right = palm_centers(joints_frame)
    for name, cam, color in [("left", left, "#46b446"), ("right", right, "#46c8e6")]:
        w = cam_to_worldlike(cam[None, :])[0]
        ax.scatter([w[0]], [w[2]], [w[1]], s=70, color=color, edgecolors="#202020", linewidths=0.7, depthshade=False)
        if contacts:
            ax.text(w[0], w[2], w[1], name, color="#202020", fontsize=8)


def draw_contact_gap_lines_3d(ax, params: np.ndarray, contacts: list[dict[str, str]], joints_frame: np.ndarray | None) -> None:
    for hand_side, anchor, hand, gap in contact_gap_entries(params, contacts, joints_frame):
        aw = cam_to_worldlike(anchor[None, :])[0]
        hw = cam_to_worldlike(hand[None, :])[0]
        ax.plot([aw[0], hw[0]], [aw[2], hw[2]], [aw[1], hw[1]], color="#111111", linewidth=1.5, linestyle="--", alpha=0.8)
        mid = (aw + hw) * 0.5
        ax.text(mid[0], mid[2], mid[1], f"{hand_side[:1]} gap={gap:.2f}m", color="#111111", fontsize=8)


def draw_contact_gap_lines_side_yz(ax, params: np.ndarray, contacts: list[dict[str, str]], joints_frame: np.ndarray | None) -> None:
    for hand_side, anchor, hand, gap in contact_gap_entries(params, contacts, joints_frame):
        aw = cam_to_worldlike(anchor[None, :])[0]
        hw = cam_to_worldlike(hand[None, :])[0]
        ax.plot([aw[2], hw[2]], [aw[1], hw[1]], color="#111111", linewidth=1.4, linestyle="--", alpha=0.8, zorder=9)
        mid = (aw + hw) * 0.5
        ax.text(mid[2], mid[1], f"{hand_side[:1]} {gap:.2f}m", color="#111111", fontsize=8)


def setup_3d_axes(ax, lo: np.ndarray, hi: np.ndarray, title: str) -> None:
    span = np.maximum(hi - lo, 0.1)
    margin = np.maximum(span * 0.12, 0.08)
    lo = lo - margin
    hi = hi + margin
    ax.set_xlim(float(lo[0]), float(hi[0]))
    ax.set_ylim(float(lo[2]), float(hi[2]))
    ax.set_zlim(float(lo[1]), float(hi[1]))
    ax.set_xlabel("X_cam (m)")
    ax.set_ylabel("Z_cam (m)")
    ax.set_zlabel("Y up (m)")
    ax.set_title(title)
    ax.view_init(elev=18, azim=-62)
    ax.grid(True, alpha=0.18)


def render_camera3d(
    pose_rows: list[dict[str, str]],
    segs: dict[str, tuple[str, str, np.ndarray, np.ndarray]],
    contacts_by_frame: dict[int, list[dict[str, str]]],
    out_dir: Path,
    width: int,
    height: int,
    fps: float,
    joints: np.ndarray | None = None,
    verts: np.ndarray | None = None,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "camera3d.tmp.mp4"
    mp4 = out_dir / "camera3d.mp4"
    preview = out_dir / "camera3d_preview.png"
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    lo, hi = collect_bounds(pose_rows, segs, joints, verts)
    preview_written = False
    traj = []
    for idx, row in enumerate(pose_rows):
        fr = int(row["frame"])
        params = pose_params(row)
        center = cam_to_worldlike(params[3:6][None, :])[0]
        traj.append(center)
        fig = plt.figure(figsize=(width / 100, height / 100), dpi=100, facecolor="#f7f8fb")
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor("#f7f8fb")
        setup_3d_axes(ax, lo, hi, "Chair camera 3D" + (" with human" if joints is not None else " object only"))
        t = np.asarray(traj)
        ax.plot(t[:, 0], t[:, 2], t[:, 1], color="#9aa4b2", linewidth=1.4, alpha=0.55)
        draw_chair_3d(ax, params, segs, contacts_by_frame.get(fr, []))
        if joints is not None:
            draw_human_3d(ax, joints[idx], None if verts is None else verts[idx], contacts_by_frame.get(fr, []))
            draw_contact_gap_lines_3d(ax, params, contacts_by_frame.get(fr, []), joints[idx])
        fig.text(0.04, 0.04, f"frame {fr:03d}  phase={row.get('source', '')}", fontsize=9, color="#555555")
        canvas = figure_to_bgr(fig)
        plt.close(fig)
        writer.write(canvas)
        if not preview_written:
            cv2.imwrite(str(preview), canvas)
            preview_written = True
    writer.release()
    finalize_video(tmp, mp4)
    return preview, mp4


def render_side_yz(
    pose_rows: list[dict[str, str]],
    segs: dict[str, tuple[str, str, np.ndarray, np.ndarray]],
    contacts_by_frame: dict[int, list[dict[str, str]]],
    out_dir: Path,
    width: int,
    height: int,
    fps: float,
    joints: np.ndarray | None = None,
    verts: np.ndarray | None = None,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "side_yz.tmp.mp4"
    mp4 = out_dir / "side_yz.mp4"
    preview = out_dir / "side_yz_preview.png"
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    lo, hi = collect_bounds(pose_rows, segs, joints, verts)
    span = np.maximum(hi - lo, 0.1)
    margin = np.maximum(span * 0.12, 0.08)
    preview_written = False
    for idx, row in enumerate(pose_rows):
        fr = int(row["frame"])
        params = pose_params(row)
        fig = plt.figure(figsize=(width / 100, height / 100), dpi=100, facecolor="#f7f8fb")
        ax = fig.add_subplot(111)
        ax.set_facecolor("#f7f8fb")
        ax.set_xlim(float(lo[2] - margin[2]), float(hi[2] + margin[2]))
        ax.set_ylim(float(lo[1] - margin[1]), float(hi[1] + margin[1]))
        ax.set_xlabel("Z_cam (depth, m)")
        ax.set_ylabel("Y up (m)")
        ax.set_title("Chair side Y-Z" + (" with human" if joints is not None else " object only"))
        ax.grid(True, alpha=0.18)
        for _, role, cam in chair_segments_cam(params, segs):
            w = cam_to_worldlike(cam)
            color = SEGMENT_COLORS.get(role, "#444444")
            ax.plot(w[:, 2], w[:, 1], color=color, linewidth=3.0 if "leg" in role or "rail" in role else 2.2, alpha=0.96)
            ax.scatter(w[:, 2], w[:, 1], s=16, color=color, edgecolors="#202020", linewidths=0.3, zorder=4)
        for hand, cam in contact_anchor_points(params, contacts_by_frame.get(fr, [])):
            w = cam_to_worldlike(cam[None, :])[0]
            ax.scatter([w[2]], [w[1]], s=75, color="#ffea00", edgecolors="#202020", linewidths=0.8, zorder=6)
            ax.text(w[2], w[1], hand.replace("_hand", ""), color="#202020", fontsize=8)
        draw_pose_axes_side_yz(ax, params)
        if joints is not None:
            if verts is not None:
                v = cam_to_worldlike(verts[idx])
                ax.scatter(v[:, 2], v[:, 1], s=4, color="#9c89f2", alpha=0.18, linewidths=0)
            body = cam_to_worldlike(joints[idx, :22])
            for a, b in BODY_EDGES:
                seg = body[[a, b]]
                ax.plot(seg[:, 2], seg[:, 1], color="#6b54d2", linewidth=2.2, alpha=0.92)
            ax.scatter(body[:, 2], body[:, 1], s=14, color="#e4dbff", edgecolors="#6b54d2", linewidths=0.4, zorder=4)
            left, right = palm_centers(joints[idx])
            for name, cam, color in [("left", left, "#46b446"), ("right", right, "#46c8e6")]:
                w = cam_to_worldlike(cam[None, :])[0]
                ax.scatter([w[2]], [w[1]], s=70, color=color, edgecolors="#202020", linewidths=0.7, zorder=7)
            draw_contact_gap_lines_side_yz(ax, params, contacts_by_frame.get(fr, []), joints[idx])
        fig.text(0.04, 0.04, f"frame {fr:03d}", fontsize=9, color="#555555")
        canvas = figure_to_bgr(fig)
        plt.close(fig)
        writer.write(canvas)
        if not preview_written:
            cv2.imwrite(str(preview), canvas)
            preview_written = True
    writer.release()
    finalize_video(tmp, mp4)
    return preview, mp4


def draw_pose_axes_side_yz(ax, params: np.ndarray, axis_len: float = 0.18) -> None:
    origin = params[3:6]
    R = rotation_matrix(params)
    axes = [
        ("+X", np.array([axis_len, 0.0, 0.0]), "#e53935"),
        ("+Y", np.array([0.0, axis_len, 0.0]), "#43a047"),
        ("+Z", np.array([0.0, 0.0, axis_len]), "#1e88e5"),
    ]
    ow = cam_to_worldlike(origin[None, :])[0]
    for label, local_axis, color in axes:
        end = origin + R @ local_axis
        ew = cam_to_worldlike(end[None, :])[0]
        ax.plot([ow[2], ew[2]], [ow[1], ew[1]], color=color, linewidth=2.6, alpha=0.95, zorder=8)
        ax.text(ew[2], ew[1], label, color=color, fontsize=8)


def write_6d_diagnostics(pose_rows: list[dict[str, str]], phase_path: Path, out_path: Path) -> None:
    phases = {}
    if phase_path.exists():
        phases = {int(r["frame"]): r.get("phase", "") for r in read_rows(phase_path)}
    fields = [
        "frame", "time", "phase",
        "tx", "ty", "tz",
        "roll_x_deg", "pitch_y_deg", "yaw_z_deg",
        "qx", "qy", "qz", "qw",
        "source",
    ]
    rows = []
    for row in pose_rows:
        params = pose_params(row)
        roll, pitch, yaw = euler_xyz_deg(params)
        fr = int(row["frame"])
        rows.append(
            {
                "frame": str(fr),
                "time": row.get("time", ""),
                "phase": phases.get(fr, ""),
                "tx": f"{params[3]:.9f}",
                "ty": f"{params[4]:.9f}",
                "tz": f"{params[5]:.9f}",
                "roll_x_deg": f"{roll:.6f}",
                "pitch_y_deg": f"{pitch:.6f}",
                "yaw_z_deg": f"{yaw:.6f}",
                "qx": row.get("qx", ""),
                "qy": row.get("qy", ""),
                "qz": row.get("qz", ""),
                "qw": row.get("qw", ""),
                "source": row.get("source", ""),
            }
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_contact_gap_diagnostics(
    pose_rows: list[dict[str, str]],
    contacts_by_frame: dict[int, list[dict[str, str]]],
    joints: np.ndarray | None,
    out_path: Path,
) -> None:
    if joints is None:
        return
    fields = ["frame", "time", "hand_side", "anchor_x", "anchor_y", "anchor_z", "hand_x", "hand_y", "hand_z", "gap_m", "gap_xy_m", "gap_z_m"]
    rows = []
    for idx, row in enumerate(pose_rows):
        fr = int(row["frame"])
        params = pose_params(row)
        for hand_side, anchor, hand, gap in contact_gap_entries(params, contacts_by_frame.get(fr, []), joints[idx]):
            delta = hand - anchor
            rows.append(
                {
                    "frame": str(fr),
                    "time": row.get("time", ""),
                    "hand_side": hand_side,
                    "anchor_x": f"{anchor[0]:.9f}",
                    "anchor_y": f"{anchor[1]:.9f}",
                    "anchor_z": f"{anchor[2]:.9f}",
                    "hand_x": f"{hand[0]:.9f}",
                    "hand_y": f"{hand[1]:.9f}",
                    "hand_z": f"{hand[2]:.9f}",
                    "gap_m": f"{gap:.9f}",
                    "gap_xy_m": f"{float(np.linalg.norm(delta[:2])):.9f}",
                    "gap_z_m": f"{delta[2]:.9f}",
                }
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render_video_overlay(
    sample: Path,
    pose_rows: list[dict[str, str]],
    segs: dict[str, tuple[str, str, np.ndarray, np.ndarray]],
    contacts_by_frame: dict[int, list[dict[str, str]]],
    out_dir: Path,
    name: str,
    fps: float,
    K_all: np.ndarray,
    joints: np.ndarray | None = None,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / f"{Path(name).stem}.tmp.mp4"
    mp4 = out_dir / name
    preview = out_dir / f"{Path(name).stem}_preview.png"

    first = cv2.imread(str(frame_path(sample, int(pose_rows[0]["frame"]))))
    if first is None:
        raise RuntimeError(f"Could not read first frame for overlay: {pose_rows[0]['frame']}")
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    preview_written = False

    for idx, row in enumerate(pose_rows):
        fr = int(row["frame"])
        img = cv2.imread(str(frame_path(sample, fr)))
        if img is None:
            raise RuntimeError(f"Could not read frame {fr}")
        params = pose_params(row)
        k_idx = min(max(fr - 1, 0), len(K_all) - 1) if K_all.ndim == 3 else 0
        K = K_all[k_idx] if K_all.ndim == 3 else K_all
        frame_contacts = contacts_by_frame.get(fr, [])
        draw_chair_projection_2d(img, params, segs, frame_contacts, K)
        if joints is not None:
            j_idx = min(max(fr - 1, 0), len(joints) - 1)
            draw_human_projection_2d(img, joints[j_idx], K, frame_contacts, params)
        draw_text(img, f"frame {fr:03d}", (18, 34), (0, 255, 0), 0.75)
        writer.write(img)
        if not preview_written:
            cv2.imwrite(str(preview), img)
            preview_written = True

    writer.release()
    finalize_video(tmp, mp4)
    return preview, mp4


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample-dir", type=Path, default=Path("samples_known_object/05_chair"))
    ap.add_argument("--pose-csv", type=Path, default=None)
    ap.add_argument("--segments-csv", type=Path, default=None, help="Chair local semantic segments to render. Defaults to results/chair_semantic_local_points/chair_semantic_local_segments.csv.")
    ap.add_argument("--contacts-csv", type=Path, default=None, help="Contact CSV. Defaults to pose CSV directory/object_contact_points.csv.")
    ap.add_argument("--phase-csv", type=Path, default=None, help="Phase CSV. Defaults to pose CSV directory/object_phase.csv.")
    ap.add_argument("--body-model-root", type=Path, default=Path("third-party/GVHMR/inputs/checkpoints/body_models"))
    ap.add_argument("--out-root", type=Path, default=None)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--vertex-stride", type=int, default=16)
    args = ap.parse_args()

    sample = args.sample_dir
    pose_csv = args.pose_csv or sample / "results/final_result/object_pose.csv"
    out_root = args.out_root or sample / "results/renders/final_result"
    pose_dir = pose_csv.parent
    segments_csv = args.segments_csv or sample / "results/chair_semantic_local_points/chair_semantic_local_segments.csv"
    contacts_csv = args.contacts_csv or pose_dir / "object_contact_points.csv"
    phase_csv = args.phase_csv or pose_dir / "object_phase.csv"
    pose_rows = read_rows(pose_csv)
    segs = fit.load_segments(segments_csv)
    contacts = load_contacts(contacts_csv)
    diagnostics_csv = out_root / "object_6d_diagnostics.csv"
    write_6d_diagnostics(pose_rows, phase_csv, diagnostics_csv)

    object_out = out_root / "object_only"
    human_out = out_root / "with_human"
    obj_cam_png, obj_cam_mp4 = render_camera3d(pose_rows, segs, contacts, object_out, args.width, args.height, args.fps)
    obj_side_png, obj_side_mp4 = render_side_yz(pose_rows, segs, contacts, object_out, args.width, args.height, args.fps)

    human_path = sample / "results/gvhmr/result.pkl"
    human = read_human_result(human_path) if human_path.exists() else None
    if human is None:
        raise RuntimeError(f"Missing GVHMR result needed for camera intrinsics: {human_path}")
    K_all = np.asarray(human["K_fullimg"], dtype=float)
    obj_overlay_png, obj_overlay_mp4 = render_video_overlay(sample, pose_rows, segs, contacts, object_out, "overlay.mp4", args.fps, K_all)
    outputs: dict[str, object] = {
        "object_pose_csv": str(pose_csv),
        "segments_csv": str(segments_csv),
        "contacts_csv": str(contacts_csv),
        "phase_csv": str(phase_csv),
        "object_6d_diagnostics_csv": str(diagnostics_csv),
        "object_only": {
            "overlay": str(obj_overlay_mp4),
            "overlay_preview": str(obj_overlay_png),
            "camera3d": str(obj_cam_mp4),
            "camera3d_preview": str(obj_cam_png),
            "side_yz": str(obj_side_mp4),
            "side_yz_preview": str(obj_side_png),
        },
        "with_human": {},
    }
    if human is not None:
        if len(pose_rows) != human["transl"].shape[0]:
            raise RuntimeError(f"Pose/human frame mismatch: {len(pose_rows)} vs {human['transl'].shape[0]}")
        joints, verts = build_body_outputs(args.body_model_root, human, args.vertex_stride)
        gap_csv = out_root / "contact_3d_gaps.csv"
        write_contact_gap_diagnostics(pose_rows, contacts, joints, gap_csv)
        hum_overlay_png, hum_overlay_mp4 = render_video_overlay(sample, pose_rows, segs, contacts, human_out, "contact_overlay.mp4", args.fps, K_all, joints)
        hum_cam_png, hum_cam_mp4 = render_camera3d(pose_rows, segs, contacts, human_out, args.width, args.height, args.fps, joints, verts)
        hum_side_png, hum_side_mp4 = render_side_yz(pose_rows, segs, contacts, human_out, args.width, args.height, args.fps, joints, verts)
        outputs["with_human"] = {
            "overlay": str(hum_overlay_mp4),
            "overlay_preview": str(hum_overlay_png),
            "camera3d": str(hum_cam_mp4),
            "camera3d_preview": str(hum_cam_png),
            "side_yz": str(hum_side_mp4),
            "side_yz_preview": str(hum_side_png),
            "contact_3d_gaps_csv": str(gap_csv),
        }
    (out_root / "stage5_3d_render_summary.json").write_text(json.dumps(outputs, indent=2) + "\n")
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
