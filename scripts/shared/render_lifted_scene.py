#!/usr/bin/env python3
"""Render true 3D scenes directly from the lifted basketball trajectory.

This renderer consumes only `results/lifting/ball_3d_lifted_trajectory.csv` and
offers two views:

1. `world`: a free 3D trajectory plot.
2. `camera`: a synthetic camera-space scene using the lifted X/Y/Z state.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg


def read_lifted(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "frame": int(row["frame"]),
                    "time": float(row["time"]),
                    "X": float(row["X"]),
                    "Y": float(row["Y"]),
                    "Z": float(row["Z"]),
                    "radius_px": float(row.get("radius_px", 12.0)),
                }
            )
    if not rows:
        raise RuntimeError(f"No lifted rows found in {path}")
    return rows


def figure_to_bgr(fig: plt.Figure) -> np.ndarray:
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)


def setup_axes(width: int, height: int, rows: list[dict[str, float]]) -> tuple[plt.Figure, plt.Axes]:
    xs = np.array([r["X"] for r in rows], dtype=np.float32)
    ys = np.array([r["Y"] for r in rows], dtype=np.float32)
    zs = np.array([r["Z"] for r in rows], dtype=np.float32)

    margin_x = max(0.05, 0.15 * float(xs.max() - xs.min()))
    margin_y = max(0.05, 0.15 * float(ys.max() - ys.min()))
    margin_z = max(0.05, 0.15 * float(zs.max() - zs.min()))

    fig = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100, facecolor="#f5f6f8")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#f5f6f8")
    ax.set_title("Basketball lifted 3D scene", pad=14)

    ax.set_xlim(float(xs.min() - margin_x), float(xs.max() + margin_x))
    ax.set_ylim(float(zs.min() - margin_z), float(zs.max() + margin_z))
    ax.set_zlim(0.0, float(max(ys.max() + margin_y, 0.25)))
    ax.set_xlabel("X (side, m)")
    ax.set_ylabel("Z (depth, m)")
    ax.set_zlabel("Y (height, m)")
    ax.set_box_aspect((2.0, 2.2, 1.2))
    ax.view_init(elev=24, azim=-58)
    ax.grid(True, alpha=0.22)

    x_floor, z_floor = np.meshgrid(
        np.linspace(float(xs.min() - margin_x), float(xs.max() + margin_x), 2),
        np.linspace(float(zs.min() - margin_z), float(zs.max() + margin_z), 2),
    )
    y_floor = np.zeros_like(x_floor)
    ax.plot_surface(x_floor, z_floor, y_floor, color="#e6eaef", alpha=0.30, shade=False, linewidth=0)

    return fig, ax


def render_scene(sample_dir: Path, fps: float, width: int, height: int, out_name: str) -> tuple[Path, Path]:
    results_dir = sample_dir / "results"
    lifting_dir = results_dir / "lifting"
    renders_dir = results_dir / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)
    rows = read_lifted(lifting_dir / "ball_3d_lifted_trajectory.csv")

    mp4_path = renders_dir / out_name
    png_path = renders_dir / "lifted_scene_preview.png"
    writer = cv2.VideoWriter(str(mp4_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {mp4_path}")

    colors = plt.cm.viridis(np.linspace(0.10, 0.92, len(rows)))
    preview_written = False

    for idx, row in enumerate(rows):
        fig, ax = setup_axes(width, height, rows)

        # Full trajectory in the background.
        ax.plot(
            [r["X"] for r in rows],
            [r["Z"] for r in rows],
            [r["Y"] for r in rows],
            color="#b9c4d8",
            linewidth=1.4,
            alpha=0.55,
        )

        # Highlight only the path up to the current frame.
        ax.plot(
            [r["X"] for r in rows[: idx + 1]],
            [r["Z"] for r in rows[: idx + 1]],
            [r["Y"] for r in rows[: idx + 1]],
            color="#4c72b0",
            linewidth=2.6,
            alpha=0.95,
        )
        ax.scatter(
            [row["X"]],
            [row["Z"]],
            [row["Y"]],
            s=220,
            color=colors[idx],
            edgecolors="#1f1f1f",
            linewidths=1.2,
            depthshade=False,
        )

        # Shadow point on the floor directly below the ball.
        ax.scatter(
            [row["X"]],
            [row["Z"]],
            [0.0],
            s=100,
            color="#555555",
            alpha=0.28,
            depthshade=False,
        )

        fig.text(
            0.08,
            0.06,
            f"frame {row['frame']:03d}   t={row['time']:.2f}s   X={row['X']:.3f}m   Y={row['Y']:.3f}m   Z={row['Z']:.3f}m",
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


def rotation_x(theta_deg: float) -> np.ndarray:
    theta = np.deg2rad(theta_deg)
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=np.float32,
    )


def make_projection(
    point_world: np.ndarray,
    cam_pos: np.ndarray,
    rot_wc: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> tuple[float, float] | None:
    point_cam = rot_wc @ (point_world - cam_pos)
    if point_cam[2] <= 1e-4:
        return None
    u = cx + fx * (point_cam[0] / point_cam[2])
    v = cy - fy * (point_cam[1] / point_cam[2])
    return float(u), float(v)


def draw_floor_grid(
    canvas: np.ndarray,
    cam_pos: np.ndarray,
    rot_wc: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> None:
    h, w = canvas.shape[:2]

    # Light gym-like background.
    canvas[:] = (226, 232, 239)
    cv2.rectangle(canvas, (0, int(h * 0.48)), (w, h), (198, 182, 158), thickness=-1)
    cv2.rectangle(canvas, (0, int(h * 0.30)), (w, int(h * 0.48)), (46, 56, 76), thickness=-1)

    grid_color = (108, 98, 84)
    axis_color = (70, 64, 56)

    # Draw world-space floor grid (Y=0) projected into the synthetic camera.
    x_lines = np.linspace(-1.5, 1.5, 7)
    z_lines = np.linspace(2.2, 4.8, 8)
    for x in x_lines:
        pts = []
        for z in np.linspace(2.2, 4.8, 18):
            p = make_projection(np.array([x, 0.0, z], dtype=np.float32), cam_pos, rot_wc, fx, fy, cx, cy)
            if p is not None:
                pts.append((int(round(p[0])), int(round(p[1]))))
        if len(pts) >= 2:
            cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, grid_color, 1, cv2.LINE_AA)

    for z in z_lines:
        pts = []
        for x in np.linspace(-1.5, 1.5, 24):
            p = make_projection(np.array([x, 0.0, z], dtype=np.float32), cam_pos, rot_wc, fx, fy, cx, cy)
            if p is not None:
                pts.append((int(round(p[0])), int(round(p[1]))))
        if len(pts) >= 2:
            cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, grid_color, 1, cv2.LINE_AA)

    # Draw the near floor line a bit darker.
    near_pts = []
    for x in np.linspace(-1.5, 1.5, 40):
        p = make_projection(np.array([x, 0.0, 2.2], dtype=np.float32), cam_pos, rot_wc, fx, fy, cx, cy)
        if p is not None:
            near_pts.append((int(round(p[0])), int(round(p[1]))))
    if len(near_pts) >= 2:
        cv2.polylines(canvas, [np.array(near_pts, dtype=np.int32)], False, axis_color, 2, cv2.LINE_AA)


def draw_ball_sprite(
    canvas: np.ndarray,
    center: tuple[int, int],
    radius: int,
    color: tuple[int, int, int],
    alpha: float = 0.92,
) -> None:
    if radius <= 2:
        return
    overlay = canvas.copy()
    cx, cy = center

    cv2.circle(overlay, center, radius, color, thickness=-1, lineType=cv2.LINE_AA)
    cv2.circle(overlay, center, radius, (34, 34, 34), thickness=max(1, radius // 10), lineType=cv2.LINE_AA)
    cv2.line(overlay, (cx - radius, cy), (cx + radius, cy), (34, 34, 34), max(1, radius // 12), cv2.LINE_AA)
    cv2.line(overlay, (cx, cy - radius), (cx, cy + radius), (34, 34, 34), max(1, radius // 12), cv2.LINE_AA)
    cv2.ellipse(overlay, center, (radius, max(2, int(radius * 0.45))), 45, 0, 360, (34, 34, 34), max(1, radius // 12), cv2.LINE_AA)
    cv2.ellipse(overlay, center, (radius, max(2, int(radius * 0.45))), -45, 0, 360, (34, 34, 34), max(1, radius // 12), cv2.LINE_AA)
    cv2.circle(
        overlay,
        (cx - max(1, int(radius * 0.32)), cy - max(1, int(radius * 0.34))),
        max(2, int(radius * 0.18)),
        (240, 246, 250),
        thickness=-1,
        lineType=cv2.LINE_AA,
    )

    mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
    cv2.circle(mask, center, radius + 1, 255, thickness=-1, lineType=cv2.LINE_AA)
    mask_f = (mask.astype(np.float32) / 255.0)[:, :, None] * alpha
    canvas[:] = (canvas.astype(np.float32) * (1.0 - mask_f) + overlay.astype(np.float32) * mask_f).astype(np.uint8)


def render_camera_scene(sample_dir: Path, fps: float, width: int, height: int, out_name: str) -> tuple[Path, Path]:
    results_dir = sample_dir / "results"
    lifting_dir = results_dir / "lifting"
    renders_dir = results_dir / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)
    rows = read_lifted(lifting_dir / "ball_3d_lifted_trajectory.csv")

    mp4_path = renders_dir / out_name
    png_path = renders_dir / "lifted_scene_camera_preview.png"
    writer = cv2.VideoWriter(str(mp4_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {mp4_path}")

    # Camera chosen to mimic the original basketball clip: centered, slightly elevated,
    # and pitched down toward the floor. This keeps the output a true XYZ render while
    # making the viewpoint feel close to the source video.
    fx = fy = width * 0.90
    cx = width * 0.50
    cy = height * 0.50
    cam_pos = np.array([0.0, 1.28, 0.0], dtype=np.float32)
    rot_wc = rotation_x(-22.0)

    traj_uv: list[tuple[int, int]] = []
    preview_written = False
    colors = plt.cm.plasma(np.linspace(0.08, 0.92, len(rows)))

    for idx, row in enumerate(rows):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        draw_floor_grid(frame, cam_pos, rot_wc, fx, fy, cx, cy)

        point_world = np.array([row["X"], row["Y"], row["Z"]], dtype=np.float32)
        proj = make_projection(point_world, cam_pos, rot_wc, fx, fy, cx, cy)
        if proj is None:
            continue
        u, v = proj
        traj_uv.append((int(round(u)), int(round(v))))

        # Historical path in camera view.
        if len(traj_uv) >= 2:
            cv2.polylines(frame, [np.array(traj_uv, dtype=np.int32)], False, (93, 126, 188), 3, cv2.LINE_AA)

        # Shadow projected straight down onto the floor.
        shadow_proj = make_projection(np.array([row["X"], 0.0, row["Z"]], dtype=np.float32), cam_pos, rot_wc, fx, fy, cx, cy)
        if shadow_proj is not None:
            su, sv = int(round(shadow_proj[0])), int(round(shadow_proj[1]))
            shadow_radius = max(8, int(round((row["radius_px"] * 0.72) / max(row["Z"], 1.0))))
            shadow_layer = frame.copy()
            cv2.ellipse(
                shadow_layer,
                (su, sv),
                (int(shadow_radius * 1.45), max(4, int(shadow_radius * 0.48))),
                0,
                0,
                360,
                (58, 58, 58),
                thickness=-1,
                lineType=cv2.LINE_AA,
            )
            cv2.addWeighted(shadow_layer, 0.18, frame, 0.82, 0.0, frame)

        # Draw the basketball using the lifted depth to scale it in the camera view.
        radius = max(7, int(round((row["radius_px"] * 3.0) / max(row["Z"], 1.0))))
        color = tuple(int(c * 255) for c in colors[idx][:3])[::-1]
        draw_ball_sprite(frame, (int(round(u)), int(round(v))), radius, color=color, alpha=0.94)

        cv2.putText(
            frame,
            f"frame {row['frame']:03d}  t={row['time']:.2f}s",
            (32, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (32, 32, 32),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"X={row['X']:.3f}m  Y={row['Y']:.3f}m  Z={row['Z']:.3f}m",
            (32, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (74, 74, 74),
            2,
            cv2.LINE_AA,
        )

        writer.write(frame)
        if not preview_written:
            cv2.imwrite(str(png_path), frame)
            preview_written = True

    writer.release()
    return png_path, mp4_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-dir", type=Path, default=Path("samples/basketball_01"))
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--out", type=str, default="lifted_scene.mp4")
    parser.add_argument("--view", choices=("world", "camera"), default="world")
    args = parser.parse_args()

    if args.view == "world":
        png_path, mp4_path = render_scene(args.sample_dir, args.fps, args.width, args.height, args.out)
    else:
        png_path, mp4_path = render_camera_scene(args.sample_dir, args.fps, args.width, args.height, args.out)
    print(f"preview_png: {png_path}")
    print(f"scene_mp4: {mp4_path}")


if __name__ == "__main__":
    main()
