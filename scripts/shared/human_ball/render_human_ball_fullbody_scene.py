#!/usr/bin/env python3
"""Render a full-body human + basketball scene in world or overlay view."""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

import smplx

BALL_COLOR = "#db7a20"
BODY_COLOR = "#8f7ae5"
BODY_EDGE_COLOR = "#4f3ca1"
BODY_POINT_COLOR = "#b6a6f4"
BALL_RADIUS_M = 0.12


def figure_to_bgr(fig: plt.Figure) -> np.ndarray:
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)


def read_ball_pose(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "frame": int(row["frame"]),
                    "time": float(row["time"]),
                    "x": float(row["tx"]),
                    "y": float(row["ty"]),
                    "z": float(row["tz"]),
                    "r": float(row["radius_m"]),
                    "u_proj": float(row["u_proj"]),
                    "v_proj": float(row["v_proj"]),
                    "radius_proj_px": float(row["radius_proj_px"]),
                    "bottom_proj_v": float(row["bottom_proj_v"]) if row.get("bottom_proj_v") else np.nan,
                    "floor_v": float(row["floor_v"]) if row.get("floor_v") else np.nan,
                    "coord_frame": row.get("coord_frame", ""),
                    "contact_frame": int(row.get("contact_frame", 0) or 0),
                }
            )
    if not rows:
        raise RuntimeError(f"No ball pose rows found in {path}")
    return rows


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


def build_body_outputs(
    body_models_root: Path,
    human_params: dict[str, np.ndarray],
    vertex_stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model = smplx.create(
        str(body_models_root),
        model_type="smplx",
        gender="neutral",
        ext="npz",
        use_pca=False,
        flat_hand_mean=True,
        num_betas=10,
        batch_size=human_params["transl"].shape[0],
    )

    body_pose = torch.from_numpy(human_params["body_pose"])
    betas = torch.from_numpy(human_params["betas"])
    global_orient = torch.from_numpy(human_params["global_orient"])
    transl = torch.from_numpy(human_params["transl"])

    with torch.inference_mode():
        output = model(
            body_pose=body_pose,
            betas=betas,
            global_orient=global_orient,
            transl=transl,
            return_verts=True,
        )

    joints = output.joints.detach().cpu().numpy().astype(np.float32)
    vertices = output.vertices.detach().cpu().numpy().astype(np.float32)
    sampled_vertices = vertices[:, ::vertex_stride, :]
    return joints, sampled_vertices, vertices


def human_cam_to_worldlike(points: np.ndarray) -> np.ndarray:
    """Convert camera-space human points to a y-up plotting convention."""
    out = points.copy()
    out[..., 1] *= -1.0
    return out


def compute_floor_alignment_offset(ball_rows: list[dict[str, float]], vertices: np.ndarray) -> float:
    """Estimate a shared floor by aligning the human mesh support to ball contact support."""
    verts_world = human_cam_to_worldlike(vertices)
    human_floor_samples = verts_world[:, :, 1].min(axis=1)
    human_floor = float(np.quantile(human_floor_samples, 0.20))

    contact_bottoms = [row["y"] - row["r"] for row in ball_rows if int(row.get("contact_frame", 0)) == 1]
    if not contact_bottoms:
        contact_bottoms = [row["y"] - row["r"] for row in ball_rows]
    ball_floor = float(np.median(np.asarray(contact_bottoms, dtype=np.float32)))
    return ball_floor - human_floor


def infer_ball_world_camera(ball_rows: list[dict[str, float]]) -> dict[str, float]:
    xs = np.asarray([row["x"] / max(row["z"], 1e-6) for row in ball_rows], dtype=np.float64)
    ys = np.asarray([row["y"] / max(row["z"], 1e-6) for row in ball_rows], dtype=np.float64)
    us = np.asarray([row["u_proj"] for row in ball_rows], dtype=np.float64)
    vs = np.asarray([row["v_proj"] for row in ball_rows], dtype=np.float64)
    rs = np.asarray([row["radius_proj_px"] for row in ball_rows], dtype=np.float64)
    zs = np.asarray([row["z"] for row in ball_rows], dtype=np.float64)

    A_u = np.stack([xs, np.ones_like(xs)], axis=1)
    fx, cx = np.linalg.lstsq(A_u, us, rcond=None)[0]

    A_v = np.stack([-ys, np.ones_like(ys)], axis=1)
    fy, floor_v = np.linalg.lstsq(A_v, vs, rcond=None)[0]

    fx_r = float(np.median(rs * zs / BALL_RADIUS_M))
    if np.isfinite(fx_r) and fx_r > 0:
        fx = 0.5 * (float(fx) + fx_r)
        fy = 0.5 * (float(fy) + fx_r)

    return {
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "floor_v": float(floor_v),
    }


def infer_sharedcam_camera(ball_rows: list[dict[str, float]]) -> dict[str, float]:
    xs = np.asarray([row["x"] / max(row["z"], 1e-6) for row in ball_rows], dtype=np.float64)
    ys = np.asarray([row["y"] / max(row["z"], 1e-6) for row in ball_rows], dtype=np.float64)
    us = np.asarray([row["u_proj"] for row in ball_rows], dtype=np.float64)
    vs = np.asarray([row["v_proj"] for row in ball_rows], dtype=np.float64)
    rs = np.asarray([row["radius_proj_px"] for row in ball_rows], dtype=np.float64)
    zs = np.asarray([row["z"] for row in ball_rows], dtype=np.float64)
    floor_samples = np.asarray(
        [row["floor_v"] for row in ball_rows if np.isfinite(row.get("floor_v", np.nan))],
        dtype=np.float64,
    )

    A_u = np.stack([xs, np.ones_like(xs)], axis=1)
    fx, cx = np.linalg.lstsq(A_u, us, rcond=None)[0]

    A_v = np.stack([ys, np.ones_like(ys)], axis=1)
    fy, cy = np.linalg.lstsq(A_v, vs, rcond=None)[0]

    fx_r = float(np.median(rs * zs / BALL_RADIUS_M))
    if np.isfinite(fx_r) and fx_r > 0:
        fx = 0.5 * (float(fx) + fx_r)
        fy = 0.5 * (float(fy) + fx_r)

    floor_v = float(np.median(floor_samples)) if floor_samples.size else float(cy)
    return {
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "floor_v": floor_v,
    }


def convert_ball_rows_to_worldlike(ball_rows: list[dict[str, float]]) -> tuple[list[dict[str, float]], dict[str, float]]:
    if not ball_rows or ball_rows[0].get("coord_frame") != "gvhmr_incam":
        return ball_rows, infer_ball_world_camera(ball_rows)

    camera = infer_sharedcam_camera(ball_rows)
    converted: list[dict[str, float]] = []
    for row in ball_rows:
        bottom_v = row["bottom_proj_v"]
        if not np.isfinite(bottom_v):
            bottom_v = row["v_proj"] + row["radius_proj_px"]
        height = BALL_RADIUS_M + ((camera["floor_v"] - bottom_v) * row["z"] / max(camera["fy"], 1e-6))
        new_row = dict(row)
        new_row["y"] = float(max(BALL_RADIUS_M, height))
        converted.append(new_row)
    return converted, infer_ball_world_camera(converted)


def project_points(points_cam: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = np.clip(points_cam[:, 2], 1e-5, None)
    u = K[0, 0] * (points_cam[:, 0] / z) + K[0, 2]
    v = K[1, 1] * (points_cam[:, 1] / z) + K[1, 2]
    valid = points_cam[:, 2] > 1e-5
    return np.stack([u, v], axis=1), valid


def project_world_points(points_world: np.ndarray, camera: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    z = np.clip(points_world[:, 2], 1e-5, None)
    u = camera["cx"] + camera["fx"] * (points_world[:, 0] / z)
    v = camera["floor_v"] - camera["fy"] * (points_world[:, 1] / z)
    valid = points_world[:, 2] > 1e-5
    return np.stack([u, v], axis=1), valid


def write_joint_csv(
    out_path: Path,
    ball_rows: list[dict[str, float]],
    joints: np.ndarray,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "frame",
        "time",
        "pelvis_x",
        "pelvis_y",
        "pelvis_z",
        "left_wrist_x",
        "left_wrist_y",
        "left_wrist_z",
        "right_wrist_x",
        "right_wrist_y",
        "right_wrist_z",
        "ball_x",
        "ball_y",
        "ball_z",
        "left_wrist_ball_dist",
        "right_wrist_ball_dist",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, ball in enumerate(ball_rows):
            pelvis = joints[idx, 0]
            lw = joints[idx, 20]
            rw = joints[idx, 21]
            ball_xyz = np.array([ball["x"], ball["y"], ball["z"]], dtype=np.float32)
            writer.writerow(
                {
                    "frame": ball["frame"],
                    "time": f"{ball['time']:.6f}",
                    "pelvis_x": f"{pelvis[0]:.6f}",
                    "pelvis_y": f"{pelvis[1]:.6f}",
                    "pelvis_z": f"{pelvis[2]:.6f}",
                    "left_wrist_x": f"{lw[0]:.6f}",
                    "left_wrist_y": f"{lw[1]:.6f}",
                    "left_wrist_z": f"{lw[2]:.6f}",
                    "right_wrist_x": f"{rw[0]:.6f}",
                    "right_wrist_y": f"{rw[1]:.6f}",
                    "right_wrist_z": f"{rw[2]:.6f}",
                    "ball_x": f"{ball['x']:.6f}",
                    "ball_y": f"{ball['y']:.6f}",
                    "ball_z": f"{ball['z']:.6f}",
                    "left_wrist_ball_dist": f"{float(np.linalg.norm(lw - ball_xyz)):.6f}",
                    "right_wrist_ball_dist": f"{float(np.linalg.norm(rw - ball_xyz)):.6f}",
                }
            )


def setup_axes(
    width: int,
    height: int,
    ball_rows: list[dict[str, float]],
    joints: np.ndarray,
    sampled_vertices: np.ndarray,
    floor_offset_y: float = 0.0,
) -> tuple[plt.Figure, plt.Axes]:
    ball_xyz = np.array([[r["x"], r["y"], r["z"]] for r in ball_rows], dtype=np.float32)
    all_pts = np.concatenate(
        [
            ball_xyz,
            joints[:, :22, :].reshape(-1, 3),
            sampled_vertices.reshape(-1, 3),
        ],
        axis=0,
    )
    all_pts = human_cam_to_worldlike(all_pts)
    all_pts[:, 1] += floor_offset_y
    xs = all_pts[:, 0]
    ys = all_pts[:, 1]
    zs = all_pts[:, 2]

    margin_x = max(0.10, 0.18 * float(xs.max() - xs.min()))
    margin_y = max(0.10, 0.18 * float(ys.max() - ys.min()))
    margin_z = max(0.10, 0.18 * float(zs.max() - zs.min()))

    fig = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100, facecolor="#f6f7fb")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#f6f7fb")
    ax.set_title("Full-body human + basketball shared scene", pad=14)
    ax.set_xlim(float(xs.min() - margin_x), float(xs.max() + margin_x))
    ax.set_ylim(float(zs.min() - margin_z), float(zs.max() + margin_z))
    ax.set_zlim(float(min(0.0, ys.min() - margin_y * 0.25)), float(ys.max() + margin_y))
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_zlabel("Y (m)")
    ax.set_box_aspect((1.45, 2.35, 1.55))
    ax.view_init(elev=21, azim=-62)
    ax.grid(True, alpha=0.18)
    return fig, ax


def draw_body_mesh(ax: plt.Axes, verts_frame: np.ndarray, floor_offset_y: float = 0.0) -> None:
    verts_frame = human_cam_to_worldlike(verts_frame)
    verts_frame[:, 1] += floor_offset_y
    ax.scatter(
        verts_frame[:, 0],
        verts_frame[:, 2],
        verts_frame[:, 1],
        s=4.0,
        color=BODY_POINT_COLOR,
        alpha=0.22,
        depthshade=False,
        linewidths=0,
    )


def render_scene(
    ball_rows: list[dict[str, float]],
    joints: np.ndarray,
    sampled_vertices: np.ndarray,
    renders_dir: Path,
    fps: float,
    width: int,
    height: int,
    out_name: str,
    preview_name: str,
) -> tuple[Path, Path]:
    renders_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = renders_dir / out_name
    png_path = renders_dir / preview_name
    writer = cv2.VideoWriter(str(mp4_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {mp4_path}")

    preview_written = False
    body_root_traj = joints[:, 0, :]
    ball_traj = np.array([[r["x"], r["y"], r["z"]] for r in ball_rows], dtype=np.float32)
    floor_offset_y = compute_floor_alignment_offset(ball_rows, sampled_vertices)
    body_root_traj_world = human_cam_to_worldlike(body_root_traj.copy())
    body_root_traj_world[:, 1] += floor_offset_y

    for idx, ball in enumerate(ball_rows):
        fig, ax = setup_axes(width, height, ball_rows, joints, sampled_vertices, floor_offset_y=floor_offset_y)

        ax.plot(ball_traj[:, 0], ball_traj[:, 2], ball_traj[:, 1], color="#d1d7e1", linewidth=1.2, alpha=0.55)
        ax.plot(
            body_root_traj_world[:, 0],
            body_root_traj_world[:, 2],
            body_root_traj_world[:, 1],
            color="#c2b6f2",
            linewidth=1.2,
            alpha=0.45,
        )

        ax.plot(ball_traj[: idx + 1, 0], ball_traj[: idx + 1, 2], ball_traj[: idx + 1, 1], color="#4c72b0", linewidth=2.5, alpha=0.96)
        ax.plot(
            body_root_traj_world[: idx + 1, 0],
            body_root_traj_world[: idx + 1, 2],
            body_root_traj_world[: idx + 1, 1],
            color=BODY_COLOR,
            linewidth=1.8,
            alpha=0.75,
        )

        draw_body_mesh(ax, sampled_vertices[idx], floor_offset_y=floor_offset_y)

        ball_size = max(90.0, 220.0 * (ball["r"] / 0.12))
        ax.scatter(
            [ball["x"]],
            [ball["z"]],
            [ball["y"]],
            s=ball_size,
            color=BALL_COLOR,
            edgecolors="#1f1f1f",
            linewidths=1.1,
            depthshade=False,
        )

        pelvis = human_cam_to_worldlike(joints[idx : idx + 1, 0:1, :])[0, 0]
        pelvis[1] += floor_offset_y
        ax.plot(
            [pelvis[0], ball["x"]],
            [pelvis[2], ball["z"]],
            [pelvis[1], ball["y"]],
            color="#7f8c8d",
            linewidth=1.0,
            alpha=0.45,
            linestyle="--",
        )

        lw = joints[idx, 20]
        rw = joints[idx, 21]
        left_dist = float(np.linalg.norm(lw - ball_traj[idx]))
        right_dist = float(np.linalg.norm(rw - ball_traj[idx]))

        fig.text(
            0.05,
            0.05,
            (
                f"frame {ball['frame']:03d}   t={ball['time']:.2f}s   "
                f"L wrist-ball={left_dist:.3f}m   R wrist-ball={right_dist:.3f}m   "
                f"floor offset={floor_offset_y:+.3f}m"
            ),
            fontsize=10,
            color="#555555",
        )

        canvas = figure_to_bgr(fig)
        plt.close(fig)
        writer.write(canvas)
        if not preview_written:
            cv2.imwrite(str(png_path), canvas)
            preview_written = True

    writer.release()
    return png_path, mp4_path


def draw_overlay_body(
    frame: np.ndarray,
    verts_frame: np.ndarray,
    camera: dict[str, float],
    floor_offset_y: float,
) -> np.ndarray:
    canvas = frame.copy()
    verts_world = human_cam_to_worldlike(verts_frame)
    verts_world[:, 1] += floor_offset_y

    verts_uv, verts_valid = project_world_points(verts_world, camera)
    for uv in verts_uv[verts_valid]:
        x = int(round(uv[0]))
        y = int(round(uv[1]))
        if 0 <= x < canvas.shape[1] and 0 <= y < canvas.shape[0]:
            cv2.circle(canvas, (x, y), 1, (244, 228, 194), -1, lineType=cv2.LINE_AA)
    return canvas


def render_overlay(
    sample_dir: Path,
    ball_rows: list[dict[str, float]],
    joints: np.ndarray,
    sampled_vertices: np.ndarray,
    renders_dir: Path,
    fps: float,
    out_name: str,
    preview_name: str,
) -> tuple[Path, Path]:
    frames_dir = sample_dir / "frames"
    first_frame = cv2.imread(str(frames_dir / "00001.png"))
    if first_frame is None:
        raise RuntimeError(f"Could not read frames from {frames_dir}")
    height, width = first_frame.shape[:2]

    renders_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = renders_dir / out_name
    png_path = renders_dir / preview_name
    writer = cv2.VideoWriter(str(mp4_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {mp4_path}")

    floor_offset_y = compute_floor_alignment_offset(ball_rows, sampled_vertices)
    ball_camera = infer_ball_world_camera(ball_rows)
    preview_written = False
    for idx, ball in enumerate(ball_rows):
        frame_path = frames_dir / f"{ball['frame']:05d}.png"
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise RuntimeError(f"Missing frame {frame_path}")

        canvas = draw_overlay_body(frame, sampled_vertices[idx], ball_camera, floor_offset_y)
        center_uv, valid = project_world_points(
            np.asarray([[ball["x"], ball["y"], ball["z"]]], dtype=np.float32),
            ball_camera,
        )
        if not valid[0]:
            writer.write(canvas)
            continue
        center = tuple(np.round(center_uv[0]).astype(int))
        radius = max(3, int(round(ball_camera["fx"] * ball["r"] / max(ball["z"], 1e-6))))

        ball_layer = canvas.copy()
        cv2.circle(ball_layer, center, radius, (32, 122, 219), -1, lineType=cv2.LINE_AA)
        cv2.circle(ball_layer, center, radius, (28, 28, 28), 2, lineType=cv2.LINE_AA)
        cv2.addWeighted(ball_layer, 0.38, canvas, 0.62, 0.0, dst=canvas)
        cv2.circle(canvas, center, radius, (32, 122, 219), 2, lineType=cv2.LINE_AA)

        lw = joints[idx, 20]
        rw = joints[idx, 21]
        ball_xyz = np.array([ball["x"], ball["y"], ball["z"]], dtype=np.float32)
        left_dist = float(np.linalg.norm(lw - ball_xyz))
        right_dist = float(np.linalg.norm(rw - ball_xyz))
        cv2.putText(
            canvas,
            f"frame {ball['frame']:03d}  t={ball['time']:.2f}s  L={left_dist:.3f}m  R={right_dist:.3f}m",
            (24, height - 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (32, 32, 32),
            2,
            lineType=cv2.LINE_AA,
        )

        writer.write(canvas)
        if not preview_written:
            cv2.imwrite(str(png_path), canvas)
            preview_written = True

    writer.release()
    return png_path, mp4_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-dir", type=Path, default=Path("samples/basketball_01"))
    parser.add_argument("--body-model-root", type=Path, default=Path("third-party/GVHMR/inputs/checkpoints/body_models"))
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--vertex-stride", type=int, default=12)
    parser.add_argument("--out", type=str, default="human_ball_fullbody_scene.mp4")
    parser.add_argument("--view", type=str, choices=("world", "overlay"), default="world")
    parser.add_argument("--ball-branch", type=str, choices=("pose6d", "pose6d_sharedcam"), default="pose6d")
    args = parser.parse_args()

    sample_dir = args.sample_dir
    results_dir = sample_dir / "results"
    renders_dir = results_dir / "renders"
    joint_dir = results_dir / "joint"

    if args.ball_branch == "pose6d":
        ball_csv = results_dir / "pose6d" / "ball_pose6d_trajectory.csv"
    else:
        ball_csv = results_dir / "pose6d_sharedcam" / "ball_pose6d_sharedcam_trajectory.csv"
    ball_rows_raw = read_ball_pose(ball_csv)
    ball_rows, _ball_world_camera = convert_ball_rows_to_worldlike(ball_rows_raw)
    human_params = read_human_result(results_dir / "gvhmr" / "result.pkl")
    if len(ball_rows) != human_params["transl"].shape[0]:
        raise RuntimeError("Ball/human frame count mismatch")

    joints, sampled_vertices, _ = build_body_outputs(args.body_model_root, human_params, args.vertex_stride)
    write_joint_csv(joint_dir / "human_ball_body_joints.csv", ball_rows, joints)

    if args.view == "world":
        png_path, mp4_path = render_scene(
            ball_rows=ball_rows,
            joints=joints,
            sampled_vertices=sampled_vertices,
            renders_dir=renders_dir,
            fps=args.fps,
            width=args.width,
            height=args.height,
            out_name=args.out,
            preview_name="human_ball_fullbody_scene_preview.png",
        )
    else:
        png_path, mp4_path = render_overlay(
            sample_dir=sample_dir,
            ball_rows=ball_rows,
            joints=joints,
            sampled_vertices=sampled_vertices,
            renders_dir=renders_dir,
            fps=args.fps,
            out_name=args.out,
            preview_name="human_ball_fullbody_overlay_preview.png",
        )
    print(f"joints_csv: {joint_dir / 'human_ball_body_joints.csv'}")
    print(f"preview_png: {png_path}")
    print(f"scene_mp4: {mp4_path}")


if __name__ == "__main__":
    main()
