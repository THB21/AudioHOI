#!/usr/bin/env python3
"""Direct shared-camera rendering for the pose6d_sharedcam basketball branch.

This renderer treats both the basketball and GVHMR human output as living in the
same full-image camera coordinate system with the same K_fullimg intrinsics.

It intentionally does not convert the ball back into the older floor-anchored
"world-height" representation.
"""

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

BALL_BGR = (32, 122, 219)
BODY_POINT_BGR = (194, 228, 244)
BODY_LINE_BGR = (104, 78, 214)
BALL_RADIUS_M = 0.12

# Readable body skeleton over the first 22 SMPL-X body joints.
BODY_EDGES = [
    (0, 1), (1, 4), (4, 7), (7, 10),
    (0, 2), (2, 5), (5, 8), (8, 11),
    (0, 3), (3, 6), (6, 9), (9, 12),
    (12, 13), (13, 16), (16, 18), (18, 20),
    (12, 14), (14, 17), (17, 19), (19, 21),
    (12, 15),
]


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
                    "contact_frame": int(row.get("contact_frame", 0) or 0),
                }
            )
    if not rows:
        raise RuntimeError(f"No ball rows found in {path}")
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
) -> tuple[np.ndarray, np.ndarray]:
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
    with torch.inference_mode():
        output = model(
            body_pose=torch.from_numpy(human_params["body_pose"]),
            betas=torch.from_numpy(human_params["betas"]),
            global_orient=torch.from_numpy(human_params["global_orient"]),
            transl=torch.from_numpy(human_params["transl"]),
            return_verts=True,
        )
    joints = output.joints.detach().cpu().numpy().astype(np.float32)
    sampled_vertices = output.vertices.detach().cpu().numpy().astype(np.float32)[:, ::vertex_stride, :]
    return joints, sampled_vertices


def cam_to_worldlike(points: np.ndarray) -> np.ndarray:
    out = points.copy()
    out[..., 1] *= -1.0
    return out


def project_points(points_cam: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = np.clip(points_cam[:, 2], 1e-6, None)
    u = K[0, 0] * (points_cam[:, 0] / z) + K[0, 2]
    v = K[1, 1] * (points_cam[:, 1] / z) + K[1, 2]
    valid = points_cam[:, 2] > 1e-6
    return np.stack([u, v], axis=1), valid


def project_ball(ball: dict[str, float], K: np.ndarray) -> tuple[tuple[int, int], int] | None:
    point = np.asarray([[ball["x"], ball["y"], ball["z"]]], dtype=np.float32)
    uv, valid = project_points(point, K)
    if not valid[0]:
        return None
    center = tuple(np.round(uv[0]).astype(int))
    radius = int(round(K[0, 0] * ball["r"] / max(ball["z"], 1e-6)))
    return center, max(3, radius)



def draw_body_skeleton_overlay(frame: np.ndarray, joints_cam: np.ndarray, K: np.ndarray) -> None:
    joints_uv, valid = project_points(joints_cam[:22], K)
    for a, b in BODY_EDGES:
        if not (valid[a] and valid[b]):
            continue
        pa = tuple(np.round(joints_uv[a]).astype(int))
        pb = tuple(np.round(joints_uv[b]).astype(int))
        cv2.line(frame, pa, pb, BODY_LINE_BGR, 3, cv2.LINE_AA)
    for uv in joints_uv[valid]:
        pt = tuple(np.round(uv).astype(int))
        cv2.circle(frame, pt, 3, BODY_POINT_BGR, -1, lineType=cv2.LINE_AA)


def draw_ball_sprite(frame: np.ndarray, center: tuple[int, int], radius: int) -> None:
    overlay = frame.copy()
    cx, cy = center
    cv2.circle(overlay, center, radius, BALL_BGR, thickness=-1, lineType=cv2.LINE_AA)
    cv2.circle(overlay, center, radius, (34, 34, 34), thickness=max(1, radius // 10), lineType=cv2.LINE_AA)
    cv2.line(overlay, (cx - radius, cy), (cx + radius, cy), (34, 34, 34), max(1, radius // 12), cv2.LINE_AA)
    cv2.line(overlay, (cx, cy - radius), (cx, cy + radius), (34, 34, 34), max(1, radius // 12), cv2.LINE_AA)
    cv2.ellipse(overlay, center, (radius, max(2, int(radius * 0.45))), 45, 0, 360, (34, 34, 34), max(1, radius // 12), cv2.LINE_AA)
    cv2.ellipse(overlay, center, (radius, max(2, int(radius * 0.45))), -45, 0, 360, (34, 34, 34), max(1, radius // 12), cv2.LINE_AA)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.circle(mask, center, radius + 1, 255, thickness=-1, lineType=cv2.LINE_AA)
    mask_f = (mask.astype(np.float32) / 255.0)[:, :, None] * 0.90
    frame[:] = (frame.astype(np.float32) * (1.0 - mask_f) + overlay.astype(np.float32) * mask_f).astype(np.uint8)


def render_overlay_ball_only(
    sample_dir: Path,
    ball_rows: list[dict[str, float]],
    K: np.ndarray,
    out_dir: Path,
    fps: float,
) -> tuple[Path, Path]:
    frames_dir = sample_dir / "frames"
    first = cv2.imread(str(frames_dir / "00001.png"))
    if first is None:
        raise RuntimeError(f"Could not read frames from {frames_dir}")
    h, w = first.shape[:2]

    mp4_path = out_dir / "overlay.mp4"
    png_path = out_dir / "overlay_preview.png"
    writer = cv2.VideoWriter(str(mp4_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {mp4_path}")

    traj: list[tuple[int, int]] = []
    preview_written = False
    for ball in ball_rows:
        frame = cv2.imread(str(frames_dir / f"{ball['frame']:05d}.png"))
        if frame is None:
            continue
        proj = project_ball(ball, K)
        if proj is None:
            writer.write(frame)
            continue
        center, radius = proj
        traj.append(center)
        if len(traj) >= 2:
            cv2.polylines(frame, [np.asarray(traj, dtype=np.int32)], False, (93, 126, 188), 2, cv2.LINE_AA)
        draw_ball_sprite(frame, center, radius)
        cv2.putText(
            frame,
            f"frame {ball['frame']:03d}  t={ball['time']:.2f}s  y_cam={ball['y']:.3f}",
            (24, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            (22, 22, 22),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)
        if not preview_written:
            cv2.imwrite(str(png_path), frame)
            preview_written = True

    writer.release()
    return png_path, mp4_path


def render_overlay_with_human(
    sample_dir: Path,
    ball_rows: list[dict[str, float]],
    sampled_vertices: np.ndarray,
    joints: np.ndarray,
    K_fullimg: np.ndarray,
    out_dir: Path,
    fps: float,
) -> tuple[Path, Path]:
    frames_dir = sample_dir / "frames"
    first = cv2.imread(str(frames_dir / "00001.png"))
    if first is None:
        raise RuntimeError(f"Could not read frames from {frames_dir}")
    h, w = first.shape[:2]

    mp4_path = out_dir / "overlay.mp4"
    png_path = out_dir / "overlay_preview.png"
    writer = cv2.VideoWriter(str(mp4_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {mp4_path}")

    preview_written = False
    traj: list[tuple[int, int]] = []
    for idx, ball in enumerate(ball_rows):
        frame = cv2.imread(str(frames_dir / f"{ball['frame']:05d}.png"))
        if frame is None:
            continue

        K = K_fullimg[idx]

        verts_uv, verts_valid = project_points(sampled_vertices[idx], K)
        for uv in verts_uv[verts_valid]:
            x = int(round(uv[0]))
            y = int(round(uv[1]))
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(frame, (x, y), 1, BODY_POINT_BGR, -1, lineType=cv2.LINE_AA)
        draw_body_skeleton_overlay(frame, joints[idx], K)

        proj = project_ball(ball, K)
        if proj is not None:
            center, radius = proj
            traj.append(center)
            if len(traj) >= 2:
                cv2.polylines(frame, [np.asarray(traj, dtype=np.int32)], False, (93, 126, 188), 2, cv2.LINE_AA)
            draw_ball_sprite(frame, center, radius)

        lw = joints[idx, 20]
        rw = joints[idx, 21]
        ball_xyz = np.asarray([ball["x"], ball["y"], ball["z"]], dtype=np.float32)
        left_dist = float(np.linalg.norm(lw - ball_xyz))
        right_dist = float(np.linalg.norm(rw - ball_xyz))
        cv2.putText(
            frame,
            f"frame {ball['frame']:03d}  t={ball['time']:.2f}s  L={left_dist:.3f}m  R={right_dist:.3f}m",
            (24, h - 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (22, 22, 22),
            2,
            cv2.LINE_AA,
        )

        writer.write(frame)
        if not preview_written:
            cv2.imwrite(str(png_path), frame)
            preview_written = True

    writer.release()
    return png_path, mp4_path


def render_camera3d(
    ball_rows: list[dict[str, float]],
    joints: np.ndarray | None,
    sampled_vertices: np.ndarray | None,
    out_dir: Path,
    fps: float,
    width: int,
    height: int,
    with_human: bool,
) -> tuple[Path, Path]:
    mp4_path = out_dir / "camera3d.mp4"
    png_path = out_dir / "camera3d_preview.png"
    writer = cv2.VideoWriter(str(mp4_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {mp4_path}")

    ball_xyz_cam = np.asarray([[r["x"], r["y"], r["z"]] for r in ball_rows], dtype=np.float32)
    ball_xyz = cam_to_worldlike(ball_xyz_cam)
    all_pts = [ball_xyz]
    if with_human and sampled_vertices is not None:
        all_pts.append(cam_to_worldlike(sampled_vertices.reshape(-1, 3)))
    all_pts = np.concatenate(all_pts, axis=0)

    xs = all_pts[:, 0]
    ys = all_pts[:, 1]
    zs = all_pts[:, 2]
    margin_x = max(0.10, 0.20 * float(xs.max() - xs.min()))
    margin_y = max(0.10, 0.20 * float(ys.max() - ys.min()))
    margin_z = max(0.10, 0.20 * float(zs.max() - zs.min()))

    preview_written = False
    for idx, ball in enumerate(ball_rows):
        fig = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100, facecolor="#f7f8fb")
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor("#f7f8fb")
        ax.set_xlim(float(xs.min() - margin_x), float(xs.max() + margin_x))
        ax.set_ylim(float(zs.min() - margin_z), float(zs.max() + margin_z))
        ax.set_zlim(float(ys.min() - margin_y), float(ys.max() + margin_y))
        ax.set_xlabel("X_cam (m)")
        ax.set_ylabel("Z_cam (m)")
        ax.set_zlabel("Y_worldlike (up, m)")
        ax.set_title("Shared-camera 3D scene (worldlike view)")
        ax.view_init(elev=18, azim=-62)
        ax.grid(True, alpha=0.18)

        ax.plot(ball_xyz[:, 0], ball_xyz[:, 2], ball_xyz[:, 1], color="#d1d7e1", linewidth=1.2, alpha=0.55)
        ax.plot(ball_xyz[: idx + 1, 0], ball_xyz[: idx + 1, 2], ball_xyz[: idx + 1, 1], color="#4c72b0", linewidth=2.4, alpha=0.95)
        ball_now = ball_xyz[idx]
        ax.scatter([ball_now[0]], [ball_now[2]], [ball_now[1]], s=180, color="#db7a20", edgecolors="#1f1f1f", linewidths=1.0, depthshade=False)

        if with_human and sampled_vertices is not None and joints is not None:
            verts = cam_to_worldlike(sampled_vertices[idx])
            ax.scatter(verts[:, 0], verts[:, 2], verts[:, 1], s=5.0, color="#9c89f2", alpha=0.22, depthshade=False, linewidths=0)
            body_j = cam_to_worldlike(joints[idx, :22, :])
            for a, b in BODY_EDGES:
                seg = body_j[[a, b]]
                ax.plot(seg[:, 0], seg[:, 2], seg[:, 1], color="#6b54d2", linewidth=2.6, alpha=0.95)
            ax.scatter(body_j[:, 0], body_j[:, 2], body_j[:, 1], s=18, color="#e4dbff", edgecolors="#6b54d2", linewidths=0.4, depthshade=False)
            pelvis_traj = cam_to_worldlike(joints[:, 0, :])
            pelvis = pelvis_traj[idx]
            ax.plot(
                pelvis_traj[: idx + 1, 0],
                pelvis_traj[: idx + 1, 2],
                pelvis_traj[: idx + 1, 1],
                color="#7b61d8",
                linewidth=1.8,
                alpha=0.55,
            )
            ax.scatter(
                [pelvis[0]],
                [pelvis[2]],
                [pelvis[1]],
                s=110,
                color="#6b54d2",
                edgecolors="#ffffff",
                linewidths=0.9,
                depthshade=False,
            )

        y_worldlike = -ball["y"]
        fig.text(0.05, 0.05, f"frame {ball['frame']:03d}   t={ball['time']:.2f}s   y_worldlike={y_worldlike:.3f}m", fontsize=10, color="#555555")
        canvas = figure_to_bgr(fig)
        plt.close(fig)
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
    parser.add_argument(
        "--ball-csv",
        type=Path,
        default=None,
        help="Path to the ball trajectory CSV to render. Defaults to pose6d_sharedcam baseline.",
    )
    parser.add_argument(
        "--render-tag",
        type=str,
        default="pose6d_sharedcam_direct",
        help="Subdirectory name under results/renders for this render batch.",
    )
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--vertex-stride", type=int, default=12)
    parser.add_argument("--with-human", action="store_true")
    args = parser.parse_args()

    sample_dir = args.sample_dir
    results_dir = sample_dir / "results"
    out_root = results_dir / "renders" / args.render_tag
    ball_out = out_root / "ball"
    human_out = out_root / "with_human"
    ball_out.mkdir(parents=True, exist_ok=True)
    human_out.mkdir(parents=True, exist_ok=True)

    ball_csv = args.ball_csv or (results_dir / "pose6d_sharedcam" / "ball_pose6d_sharedcam_trajectory.csv")
    ball_rows = read_ball_pose(ball_csv)
    human = read_human_result(results_dir / "gvhmr" / "result.pkl")
    if len(ball_rows) != human["transl"].shape[0]:
        raise RuntimeError("Ball/human frame count mismatch")

    # Ball-only outputs always get written.
    ball_overlay_png, ball_overlay_mp4 = render_overlay_ball_only(sample_dir, ball_rows, human["K_fullimg"][0], ball_out, args.fps)
    ball_cam3d_png, ball_cam3d_mp4 = render_camera3d(ball_rows, None, None, ball_out, args.fps, args.width, args.height, with_human=False)
    print(f"ball_csv: {ball_csv}")
    print(f"ball_overlay_preview: {ball_overlay_png}")
    print(f"ball_overlay_mp4: {ball_overlay_mp4}")
    print(f"ball_camera3d_preview: {ball_cam3d_png}")
    print(f"ball_camera3d_mp4: {ball_cam3d_mp4}")

    if args.with_human:
        joints, sampled_vertices = build_body_outputs(args.body_model_root, human, args.vertex_stride)
        human_overlay_png, human_overlay_mp4 = render_overlay_with_human(
            sample_dir,
            ball_rows,
            sampled_vertices,
            joints,
            human["K_fullimg"],
            human_out,
            args.fps,
        )
        human_cam3d_png, human_cam3d_mp4 = render_camera3d(
            ball_rows,
            joints,
            sampled_vertices,
            human_out,
            args.fps,
            args.width,
            args.height,
            with_human=True,
        )
        print(f"human_overlay_preview: {human_overlay_png}")
        print(f"human_overlay_mp4: {human_overlay_mp4}")
        print(f"human_camera3d_preview: {human_cam3d_png}")
        print(f"human_camera3d_mp4: {human_cam3d_mp4}")


if __name__ == "__main__":
    main()
