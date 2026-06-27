#!/usr/bin/env python3
"""Render current radius-free mug Articraft result with object-only/with-human layout."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import render_mug_articraft_camera3d as m16
import render_pose6d_scene as scene

LEFT_HAND_EDGES = [
    (20, 25), (25, 26), (26, 27),
    (20, 28), (28, 29), (29, 30),
    (20, 31), (31, 32), (32, 33),
    (20, 34), (34, 35), (35, 36),
    (20, 37), (37, 38), (38, 39),
]
RIGHT_HAND_EDGES = [
    (21, 40), (40, 41), (41, 42),
    (21, 43), (43, 44), (44, 45),
    (21, 46), (46, 47), (47, 48),
    (21, 49), (49, 50), (50, 51),
    (21, 52), (52, 53), (53, 54),
]
LEFT_PALM_IDS = [20, 25, 28, 31, 34]
RIGHT_PALM_IDS = [21, 40, 43, 46, 49]


def read_plugin_csv(path: Path) -> dict[int, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open() as f:
        return {int(float(r["frame"])): r for r in csv.DictReader(f)}


def ff(row: dict[str, str] | None, key: str, default: float = math.nan) -> float:
    if not row:
        return default
    try:
        return float(row.get(key, "") or default)
    except (TypeError, ValueError):
        return default


def contact_target_cam(row: dict[str, str] | None, fallback: np.ndarray | None = None) -> np.ndarray | None:
    if not row:
        return fallback
    target = np.array([
        ff(row, "object_contact_world_x"),
        ff(row, "object_contact_world_y"),
        ff(row, "object_contact_world_z"),
    ], dtype=float)
    if np.isfinite(target).all():
        return target
    return fallback


def grasp_basis(palm: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forward = target - palm
    norm = float(np.linalg.norm(forward))
    forward = forward / norm if norm > 1e-6 else np.array([0.0, 0.0, 1.0])
    camera_up = np.array([0.0, -1.0, 0.0])
    side = np.cross(forward, camera_up)
    if np.linalg.norm(side) < 1e-6:
        side = np.array([1.0, 0.0, 0.0])
    side = side / np.linalg.norm(side)
    up = np.cross(side, forward)
    up = up / max(np.linalg.norm(up), 1e-6)
    return forward, side, up


def synthetic_grasp_curves(palm: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    forward, side, up = grasp_basis(palm, target)
    palm_center = palm - 0.030 * forward
    palm_patch = np.array([
        palm_center + 0.030 * side + 0.020 * up,
        palm_center - 0.030 * side + 0.020 * up,
        palm_center - 0.035 * side - 0.022 * up,
        palm_center + 0.035 * side - 0.022 * up,
        palm_center + 0.030 * side + 0.020 * up,
    ])
    spreads = np.array([-0.050, -0.025, 0.000, 0.025, 0.050])
    heights = np.array([0.020, 0.026, 0.030, 0.026, 0.018])
    starts = np.array([palm_center + s * side + h * up for s, h in zip(spreads, heights)])
    targets = np.array([
        target + s * side + h * up - 0.010 * forward
        for s, h in zip(spreads * 0.45, heights * 0.20)
    ])
    curves = []
    for s, t in zip(starts, targets):
        mid = 0.45 * s + 0.55 * t - 0.045 * forward + 0.020 * up
        curves.append(np.array([
            (1.0 - u) ** 2 * s + 2.0 * (1.0 - u) * u * mid + u ** 2 * t
            for u in np.linspace(0.0, 1.0, 18)
        ]))
    thumb_start = palm_center - 0.055 * side - 0.010 * up
    thumb_target = target + 0.040 * side - 0.006 * up
    thumb_mid = 0.50 * thumb_start + 0.50 * thumb_target - 0.060 * forward - 0.010 * up
    curves.append(np.array([
        (1.0 - u) ** 2 * thumb_start + 2.0 * (1.0 - u) * u * thumb_mid + u ** 2 * thumb_target
        for u in np.linspace(0.0, 1.0, 18)
    ]))
    return palm_patch, curves


def plugin_state(fr: int, gate_rows: dict[int, dict[str, str]], release_frame: int) -> tuple[str, bool, bool, bool]:
    row = gate_rows.get(fr, {})
    direct = str(row.get("passes_direct_anchor_gate", "")).strip() == "1"
    near = str(row.get("passes_palm_gate", "")).strip() == "1"
    handle = str(row.get("nearest_articraft_part", "")).strip() == "handle_loop"
    released = fr >= release_frame
    return (
        "TABLE_RELEASED" if released else ("DIRECT_GRASP" if direct else ("PALM_NEAR_GRASP" if near else ("INFERRED_MUG_CONTACT_GRASP" if handle else "NO_GRASP"))),
        direct,
        near,
        released,
    )

def prepare_mesh_sequence(poses, phases, meshes, frames):
    mesh_cam = {}
    centers = []
    all_bounds = []
    for fr in frames:
        p = poses[fr]
        ph = phases[fr]
        frame_parts = {}
        for name, (verts, _edges) in meshes.items():
            cam = m16.transform_mesh(p, verts, ph)
            frame_parts[name] = cam
            if name in {"body_shell", "rim_ring", "bottom_disk", "handle_loop"}:
                all_bounds.append(cam[:: max(1, len(cam) // 120)])
        mesh_cam[fr] = frame_parts
        centers.append(m16.base.transform(p, np.zeros((1, 3), dtype=float))[0])
    return mesh_cam, np.asarray(centers), all_bounds


def bounds_for(mesh_bounds, centers_cam, joints=None, sampled_vertices=None):
    all_pts = list(mesh_bounds) + [centers_cam]
    if joints is not None:
        all_pts.append(joints[:, :55, :].reshape(-1, 3))
    if sampled_vertices is not None:
        all_pts.append(sampled_vertices.reshape(-1, 3))
    world = m16.cam_to_worldlike(np.concatenate(all_pts, axis=0))
    mn = np.nanmin(world, axis=0)
    mx = np.nanmax(world, axis=0)
    margin = np.maximum(0.10, 0.18 * (mx - mn))
    return mn - margin, mx + margin


def draw_grasp_plugin_3d(ax, joints_frame, fr: int, palm_rows, gate_rows, release_frame: int, fallback_target=None):
    active = palm_rows.get(fr, {}).get("active_hand", "left_hand")
    palm_ids = LEFT_PALM_IDS if active == "left_hand" else RIGHT_PALM_IDS
    palm = joints_frame[palm_ids].mean(axis=0)
    target = contact_target_cam(gate_rows.get(fr), fallback_target)
    state, direct, near, released = plugin_state(fr, gate_rows, release_frame)
    enabled = target is not None and not released and state != "NO_GRASP"
    color = "#00ff73" if (direct or near) else "#ffb000"
    if enabled:
        palm_patch, curves = synthetic_grasp_curves(palm, target)
        palm_w = m16.cam_to_worldlike(palm_patch)
        ax.plot(palm_w[:, 0], palm_w[:, 2], palm_w[:, 1], color=color, linewidth=3.0, alpha=0.96)
        for curve in curves:
            cw = m16.cam_to_worldlike(curve)
            ax.plot(cw[:, 0], cw[:, 2], cw[:, 1], color=color, linewidth=3.4, alpha=0.98)
        tw = m16.cam_to_worldlike(target[None, :])[0]
        ax.scatter([tw[0]], [tw[2]], [tw[1]], s=95, color="#00e0ff", edgecolors="#111", linewidths=0.4, depthshade=False)
    pw = m16.cam_to_worldlike(palm[None, :])[0]
    ax.scatter([pw[0]], [pw[2]], [pw[1]], s=80, color="#00ff73", edgecolors="#111", linewidths=0.4, depthshade=False)
    return state


def draw_human(ax, joints, sampled_vertices, idx, fr=None, palm_rows=None, gate_rows=None, release_frame=150, fallback_target=None):
    verts = m16.cam_to_worldlike(sampled_vertices[idx])
    ax.scatter(verts[:, 0], verts[:, 2], verts[:, 1], s=4.0, color="#9c89f2", alpha=0.18, depthshade=False, linewidths=0)
    body_j = m16.cam_to_worldlike(joints[idx, :22, :])
    for a, b in scene.BODY_EDGES:
        seg = body_j[[a, b]]
        ax.plot(seg[:, 0], seg[:, 2], seg[:, 1], color="#6b54d2", linewidth=2.4, alpha=0.95)
    ax.scatter(body_j[:, 0], body_j[:, 2], body_j[:, 1], s=18, color="#e4dbff", edgecolors="#6b54d2", linewidths=0.4, depthshade=False)
    for edges, color in [(LEFT_HAND_EDGES, "#31c978"), (RIGHT_HAND_EDGES, "#86d7a6")]:
        for a, b in edges:
            if a >= joints.shape[1] or b >= joints.shape[1]:
                continue
            seg = m16.cam_to_worldlike(joints[idx, [a, b], :])
            ax.plot(seg[:, 0], seg[:, 2], seg[:, 1], color=color, linewidth=2.8, alpha=0.98)
        ids = sorted({v for edge in edges for v in edge if v < joints.shape[1]})
        pts = m16.cam_to_worldlike(joints[idx, ids, :])
        ax.scatter(pts[:, 0], pts[:, 2], pts[:, 1], s=24, color=color, edgecolors="#0b3b22", linewidths=0.3, depthshade=False)
    if fr is not None and palm_rows is not None and gate_rows is not None:
        return draw_grasp_plugin_3d(ax, joints[idx], fr, palm_rows, gate_rows, release_frame, fallback_target)
    return ""



def load_K(sample: Path) -> np.ndarray:
    with (sample / "results" / "gvhmr" / "result.pkl").open("rb") as f:
        data = pickle.load(f)
    K = np.asarray(data["K_fullimg"], dtype=float)
    return K[0] if K.ndim == 3 else K


def project_points(points: np.ndarray, K: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    z = np.maximum(pts[:, 2], 1e-6)
    u = K[0, 0] * pts[:, 0] / z + K[0, 2]
    v = K[1, 1] * pts[:, 1] / z + K[1, 2]
    return np.stack([u, v], axis=1)


def draw_mesh_overlay(img: np.ndarray, mesh_parts: dict[str, np.ndarray], meshes_solid, K: np.ndarray, *_args, **_kwargs):
    """Solid face renderer using painter's algorithm + diffuse shading."""
    m16.rigid.draw_solid_overlay_all_parts(img, mesh_parts, meshes_solid, K)


def draw_grasp_plugin_overlay(img: np.ndarray, joints_frame: np.ndarray, K: np.ndarray, fr: int, palm_rows, gate_rows, release_frame: int):
    active = palm_rows.get(fr, {}).get("active_hand", "left_hand")
    palm_ids = LEFT_PALM_IDS if active == "left_hand" else RIGHT_PALM_IDS
    palm = joints_frame[palm_ids].mean(axis=0)
    target = contact_target_cam(gate_rows.get(fr))
    state, direct, near, released = plugin_state(fr, gate_rows, release_frame)
    color = (80, 255, 80) if (direct or near) else (0, 190, 255)
    palm_uv = project_points(palm[None, :], K)[0]
    if np.all(np.isfinite(palm_uv)):
        cv2.circle(img, tuple(np.round(palm_uv).astype(int)), 7, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(img, tuple(np.round(palm_uv).astype(int)), 5, (80, 255, 80), -1, cv2.LINE_AA)
    if target is not None and not released and state != "NO_GRASP":
        palm_patch, curves = synthetic_grasp_curves(palm, target)
        for curve in [palm_patch] + curves:
            uv = project_points(curve, K)
            pts = np.round(uv[np.all(np.isfinite(uv), axis=1)]).astype(np.int32)
            if len(pts) >= 2:
                cv2.polylines(img, [pts], False, color, 2, cv2.LINE_AA)
        tuv = project_points(target[None, :], K)[0]
        if np.all(np.isfinite(tuv)):
            cv2.circle(img, tuple(np.round(tuv).astype(int)), 7, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(img, tuple(np.round(tuv).astype(int)), 5, (0, 225, 255), -1, cv2.LINE_AA)
    cv2.putText(img, state, (24, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, state, (24, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.68, color, 2, cv2.LINE_AA)


def draw_human_overlay(img: np.ndarray, joints_frame: np.ndarray, K: np.ndarray, fr: int | None = None, palm_rows=None, gate_rows=None, release_frame: int = 150):
    uv = project_points(joints_frame[:22], K)
    for a, b in scene.BODY_EDGES:
        if a >= len(uv) or b >= len(uv):
            continue
        if np.all(np.isfinite(uv[[a, b]])):
            cv2.line(img, tuple(np.round(uv[a]).astype(int)), tuple(np.round(uv[b]).astype(int)), (210, 90, 255), 2, cv2.LINE_AA)
    for p in uv:
        if np.all(np.isfinite(p)):
            cv2.circle(img, tuple(np.round(p).astype(int)), 3, (245, 230, 255), -1, cv2.LINE_AA)
            cv2.circle(img, tuple(np.round(p).astype(int)), 3, (110, 80, 210), 1, cv2.LINE_AA)
    for edges in (LEFT_HAND_EDGES, RIGHT_HAND_EDGES):
        hand_ids = sorted({v for edge in edges for v in edge if v < len(joints_frame)})
        huv = project_points(joints_frame[hand_ids], K)
        id_to_uv = {jid: huv[i] for i, jid in enumerate(hand_ids)}
        for a, b in edges:
            if a in id_to_uv and b in id_to_uv and np.all(np.isfinite([id_to_uv[a], id_to_uv[b]])):
                cv2.line(img, tuple(np.round(id_to_uv[a]).astype(int)), tuple(np.round(id_to_uv[b]).astype(int)), (90, 230, 130), 2, cv2.LINE_AA)
        for p in huv:
            if np.all(np.isfinite(p)):
                cv2.circle(img, tuple(np.round(p).astype(int)), 3, (80, 245, 130), -1, cv2.LINE_AA)
    if fr is not None and palm_rows is not None and gate_rows is not None:
        draw_grasp_plugin_overlay(img, joints_frame, K, fr, palm_rows, gate_rows, release_frame)


def render_overlay_variant(args, out_dir: Path, frames, meshes, mesh_cam, K: np.ndarray, with_human=False, joints=None, palm_rows=None, gate_rows=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    mp4 = out_dir / "overlay.mp4"
    preview = out_dir / "overlay_preview.png"
    writer = None
    tmp = mp4.with_suffix(".tmp.mp4")
    preview_written = False
    for idx, fr in enumerate(frames):
        img_path = args.sample_dir / "frames" / f"{fr:05d}.png"
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Missing frame image: {img_path}")
        draw_mesh_overlay(img, mesh_cam[fr], meshes, K, args.body_edge_step, args.detail_edge_step)
        if with_human and joints is not None:
            jidx = min(max(fr - 1, 0), len(joints) - 1)
            draw_human_overlay(img, joints[jidx], K, fr=fr, palm_rows=palm_rows, gate_rows=gate_rows, release_frame=args.release_frame)
        cv2.putText(img, f"frame {fr:03d}  anchor-depth pose + Articraft mesh", (24, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 230, 30), 2, cv2.LINE_AA)
        if writer is None:
            h, w = img.shape[:2]
            writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError(f"Could not open writer: {tmp}")
        writer.write(img)
        if not preview_written:
            cv2.imwrite(str(preview), img)
            preview_written = True
    if writer is None:
        raise RuntimeError("No overlay frames rendered")
    writer.release()
    m16.finalize(tmp, mp4)
    return {"overlay": str(mp4), "overlay_preview": str(preview)}


def render_side_yz_variant(args, out_dir: Path, frames, meshes, mesh_cam, centers_cam, with_human=False, joints=None, palm_rows=None, gate_rows=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    mp4 = out_dir / "side_yz.mp4"
    preview = out_dir / "side_yz_preview.png"
    writer, tmp = m16.make_writer(mp4, args.fps, (args.width, args.height))
    centers_world = m16.cam_to_worldlike(centers_cam)
    all_pts = [centers_cam]
    for fr in frames[::max(1, len(frames)//60)]:
        for name in ["body_shell", "rim_ring", "bottom_disk", "handle_loop"]:
            if name in mesh_cam[fr]:
                pts = mesh_cam[fr][name]
                all_pts.append(pts[::max(1, len(pts)//80)])
    if with_human and joints is not None:
        all_pts.append(joints[:, :22, :].reshape(-1, 3))
    world_all = m16.cam_to_worldlike(np.concatenate(all_pts, axis=0))
    y_min, y_max = np.nanmin(world_all[:, 1]), np.nanmax(world_all[:, 1])
    z_min, z_max = np.nanmin(world_all[:, 2]), np.nanmax(world_all[:, 2])
    my = max(0.10, 0.16 * (y_max - y_min))
    mz = max(0.10, 0.16 * (z_max - z_min))
    preview_written = False
    for idx, fr in enumerate(frames):
        fig, ax = plt.subplots(figsize=(args.width/100, args.height/100), dpi=100, facecolor="#f7f8fb")
        ax.set_facecolor("#f7f8fb")
        ax.set_xlim(float(z_min - mz), float(z_max + mz))
        ax.set_ylim(float(y_min - my), float(y_max + my))
        ax.set_xlabel("Z_cam depth (m)")
        ax.set_ylabel("Y_worldlike up (m)")
        ax.set_title("Articraft mug side Y-Z" + (" with human" if with_human else " object only"))
        ax.grid(True, alpha=0.25)
        ax.plot(centers_world[:, 2], centers_world[:, 1], color="#aeb8c8", lw=1.0, alpha=0.35)
        ax.plot(centers_world[:idx+1, 2], centers_world[:idx+1, 1], color="#4c72b0", lw=2.2, alpha=0.95)
        for name in ["body_shell", "rim_ring", "bottom_disk", "handle_loop"]:
            if name not in mesh_cam[fr] or name not in meshes:
                continue
            ptsw = m16.cam_to_worldlike(mesh_cam[fr][name])
            _verts, edges = meshes[name]
            color = m16.PART_COLORS.get(name, "#333333")
            step = args.body_edge_step if name == "body_shell" else args.detail_edge_step
            for a, b in edges[::max(1, step)]:
                seg = ptsw[[a, b]]
                if np.all(np.isfinite(seg)):
                    ax.plot(seg[:, 2], seg[:, 1], color=color, lw=0.6 if name == "body_shell" else 1.4, alpha=0.50 if name == "body_shell" else 0.95)
        if with_human and joints is not None:
            jidx = min(max(fr - 1, 0), len(joints) - 1)
            jw = m16.cam_to_worldlike(joints[jidx, :22, :])
            for a, b in scene.BODY_EDGES:
                seg = jw[[a, b]]
                ax.plot(seg[:, 2], seg[:, 1], color="#6b54d2", lw=2.0, alpha=0.9)
            for edges in (LEFT_HAND_EDGES, RIGHT_HAND_EDGES):
                for a, b in edges:
                    if a >= joints.shape[1] or b >= joints.shape[1]:
                        continue
                    seg = m16.cam_to_worldlike(joints[jidx, [a, b], :])
                    ax.plot(seg[:, 2], seg[:, 1], color="#2bc275", lw=2.2, alpha=0.95)
            if palm_rows is not None and gate_rows is not None:
                active = palm_rows.get(fr, {}).get("active_hand", "left_hand")
                palm_ids = LEFT_PALM_IDS if active == "left_hand" else RIGHT_PALM_IDS
                palm = joints[jidx, palm_ids, :].mean(axis=0)
                target = contact_target_cam(gate_rows.get(fr))
                state, direct, near, released = plugin_state(fr, gate_rows, args.release_frame)
                if target is not None and not released and state != "NO_GRASP":
                    _patch, curves = synthetic_grasp_curves(palm, target)
                    color = "#00ff73" if (direct or near) else "#ffb000"
                    for curve in curves:
                        cw = m16.cam_to_worldlike(curve)
                        ax.plot(cw[:, 2], cw[:, 1], color=color, lw=2.5, alpha=0.95)
        fig.text(0.04, 0.045, f"frame {fr:03d}", fontsize=10, color="#333")
        canvas = m16.figure_to_bgr(fig)
        plt.close(fig)
        writer.write(canvas)
        if not preview_written:
            cv2.imwrite(str(preview), canvas)
            preview_written = True
    writer.release()
    m16.finalize(tmp, mp4)
    return {"side_yz": str(mp4), "side_yz_preview": str(preview)}

# Solid face colors for camera3d Poly3DCollection (matplotlib hex, RGB order)
_CAM3D_SOLID_COLORS = {
    'body_shell':  "#DECDB1",   # warm cream/sandy-beige
    'rim_ring':    "#C8B79B",
    'bottom_disk': "#BEAD91",
    'handle_loop': "#DECDB1",
}


def render_variant(args, out_dir: Path, frames, poses, phases, meshes, mesh_cam, centers_cam, mesh_bounds, meshes_solid=None, with_human=False, joints=None, sampled_vertices=None, palm_rows=None, gate_rows=None):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    out_dir.mkdir(parents=True, exist_ok=True)
    mp4 = out_dir / "camera3d.mp4"
    preview = out_dir / "camera3d_preview.png"
    centers_world = m16.cam_to_worldlike(centers_cam)
    lo, hi = bounds_for(mesh_bounds, centers_cam, joints if with_human else None, sampled_vertices if with_human else None)

    writer, tmp = m16.make_writer(mp4, args.fps, (args.width, args.height))
    preview_written = False
    for idx, fr in enumerate(frames):
        fig = plt.figure(figsize=(args.width / 100.0, args.height / 100.0), dpi=100, facecolor="#f7f8fb")
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor("#f7f8fb")
        ax.set_xlim(float(lo[0]), float(hi[0]))
        ax.set_ylim(float(lo[2]), float(hi[2]))
        ax.set_zlim(float(lo[1]), float(hi[1]))
        ax.set_xlabel("X_cam (m)")
        ax.set_ylabel("Z_cam depth (m)")
        ax.set_zlabel("Y_worldlike up (m)")
        title = "Articraft mug camera 3D with human" if with_human else "Articraft mug camera 3D object only"
        ax.set_title(title)
        ax.view_init(elev=args.elev, azim=args.azim)
        ax.grid(True, alpha=0.18)

        ax.plot(centers_world[:, 0], centers_world[:, 2], centers_world[:, 1], color="#aeb8c8", linewidth=1.2, alpha=0.42)
        ax.plot(centers_world[: idx + 1, 0], centers_world[: idx + 1, 2], centers_world[: idx + 1, 1], color="#4c72b0", linewidth=2.3, alpha=0.9)

        if meshes_solid is not None:
            # Solid face rendering via Poly3DCollection (painter's algorithm in matplotlib 3D)
            for name in ["body_shell", "bottom_disk", "rim_ring", "handle_loop"]:
                if name not in meshes_solid or name not in mesh_cam[fr]:
                    continue
                _sv, faces, _se = meshes_solid[name]
                pts_world = m16.cam_to_worldlike(mesh_cam[fr][name])
                # Matplotlib 3D axis mapping: plot(x, y=z_cam, z=y_worldlike)
                tris = pts_world[faces]               # Nx3x3
                verts_mpl = tris[:, :, [0, 2, 1]]    # reorder axes for matplotlib
                color_hex = _CAM3D_SOLID_COLORS.get(name, "#D4B896")
                coll = Poly3DCollection(verts_mpl, alpha=0.78, facecolor=color_hex, edgecolor='none')
                ax.add_collection3d(coll)
        else:
            for name in ["body_shell", "rim_ring", "bottom_disk", "handle_loop"]:
                if name not in meshes:
                    continue
                _verts, edges = meshes[name]
                pts_world = m16.cam_to_worldlike(mesh_cam[fr][name])
                edge_step = args.body_edge_step if name == "body_shell" else args.detail_edge_step
                m16.draw_mesh(ax, pts_world, edges, m16.PART_COLORS.get(name, "#333"), m16.PART_LW.get(name, 1.0), 0.92 if name != "body_shell" else 0.55, edge_step=edge_step)

        c = centers_world[idx]
        ax.scatter([c[0]], [c[2]], [c[1]], s=55, color="#db7a20", edgecolors="#1f1f1f", linewidths=0.8, depthshade=False)
        state = ""
        if with_human and joints is not None and sampled_vertices is not None:
            fallback = np.mean(mesh_cam[fr]["handle_loop"], axis=0) if "handle_loop" in mesh_cam[fr] else None
            jidx = min(max(fr - 1, 0), len(joints) - 1)
            state = draw_human(ax, joints, sampled_vertices, jidx, fr=fr, palm_rows=palm_rows, gate_rows=gate_rows, release_frame=args.release_frame, fallback_target=fallback)
        fig.text(0.04, 0.045, f"frame {fr:03d}  phase={math.degrees(phases[fr]):.1f} deg  {state}", fontsize=10, color="#333333")
        canvas = m16.figure_to_bgr(fig)
        plt.close(fig)
        writer.write(canvas)
        if not preview_written:
            cv2.imwrite(str(preview), canvas)
            preview_written = True
    writer.release()
    m16.finalize(tmp, mp4)
    return {"camera3d": str(mp4), "camera3d_preview": str(preview)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir", type=Path, default=Path("samples_known_object/02_mug"))
    ap.add_argument("--mesh-root", type=Path, default=Path("samples_known_object/02_mug/articraft/materialized_mug_mesh"))
    ap.add_argument("--pose-csv", type=Path, default=None)
    ap.add_argument("--phase-csv", type=Path, default=None)
    ap.add_argument("--out-root", type=Path, default=None)
    ap.add_argument("--body-model-root", type=Path, default=Path("third-party/GVHMR/inputs/checkpoints/body_models"))
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--elev", type=float, default=18.0)
    ap.add_argument("--azim", type=float, default=-62.0)
    ap.add_argument("--body-edge-step", type=int, default=3)
    ap.add_argument("--detail-edge-step", type=int, default=1)
    ap.add_argument("--vertex-stride", type=int, default=16)
    ap.add_argument("--phase-col", type=str, default=None, help="CSV column for phase (auto-detected if omitted).")
    ap.add_argument("--wireframe", action="store_true", help="Use wireframe camera3d (fast) instead of solid Poly3DCollection.")
    ap.add_argument("--release-frame", type=int, default=150, help="Frame after which the mug is table-released.")
    args = ap.parse_args()

    import csv as _csv
    sample = args.sample_dir
    pose_csv = args.pose_csv or (sample / "proxy" / "mug_body_only_cylinder_pose_anchor_depth_sequence.csv")
    if not pose_csv.exists():
        pose_csv = sample / "proxy" / "mug_body_only_cylinder_pose_table_static_sequence.csv"
    if not pose_csv.exists():
        pose_csv = sample / "proxy" / "mug_body_only_cylinder_pose_segmented_sequence.csv"
    phase_csv = args.phase_csv or (sample / "results" / "renders" / "M17_phase_corrected" / "corrected_handle_phase.csv")
    if not phase_csv.exists():
        phase_csv = sample / "results" / "renders" / "M14_joint_contact_handle_phase" / "handle_phase_joint_contact.csv"
    out_root = args.out_root or (sample / "results" / "renders" / "M18_anchor_depth_scene")

    # Auto-detect phase column: explicit arg > m17_phase_rad in header > default auto
    if args.phase_col:
        phase_col = args.phase_col
    else:
        with phase_csv.open() as _f:
            _hdr = next(_csv.reader(_f))
        phase_col = "m17_phase_rad" if "m17_phase_rad" in _hdr else None

    poses = m16.read_pose_sequence(pose_csv)
    phases = m16.read_phase_sequence(phase_csv, col=phase_col)
    meshes        = m16.rigid.load_articraft_meshes(args.mesh_root)        # edges for camera3d / side_yz
    meshes_solid  = m16.rigid.load_articraft_meshes_solid(args.mesh_root)  # faces for solid overlay
    frames = sorted(set(poses) & set(phases))
    mesh_cam, centers_cam, mesh_bounds = prepare_mesh_sequence(poses, phases, meshes, frames)
    K = load_K(sample)
    palm_rows = read_plugin_csv(sample / "results" / "hand_contact_proxy" / "mug_palm_proxy.csv")
    gate_rows = read_plugin_csv(sample / "results" / "hand_contact_proxy" / "palm_handle_verification.csv")

    outputs = {
        "pose_csv": str(pose_csv),
        "phase_csv": str(phase_csv),
        "phase_col": phase_col or "auto",
        "mesh_root": str(args.mesh_root),
        "mesh_parts": ["body_shell", "rim_ring", "bottom_disk", "handle_loop"],
        "num_frames": len(frames),
        "object_only": render_variant(args, out_root / "object_only", frames, poses, phases, meshes, mesh_cam, centers_cam, mesh_bounds, meshes_solid=None if args.wireframe else meshes_solid, with_human=False),
    }
    outputs["object_only"].update(render_overlay_variant(args, out_root / "object_only", frames, meshes_solid, mesh_cam, K, with_human=False))
    outputs["object_only"].update(render_side_yz_variant(args, out_root / "object_only", frames, meshes, mesh_cam, centers_cam, with_human=False))
    human = scene.read_human_result(sample / "results" / "gvhmr" / "result.pkl")
    joints, sampled_vertices = scene.build_body_outputs(args.body_model_root, human, args.vertex_stride)
    outputs["with_human"] = render_variant(
        args, out_root / "with_human", frames, poses, phases, meshes, mesh_cam, centers_cam, mesh_bounds,
        meshes_solid=None if args.wireframe else meshes_solid,
        with_human=True, joints=joints, sampled_vertices=sampled_vertices,
        palm_rows=palm_rows, gate_rows=gate_rows,
    )
    outputs["with_human"].update(render_overlay_variant(args, out_root / "with_human", frames, meshes_solid, mesh_cam, K, with_human=True, joints=joints, palm_rows=palm_rows, gate_rows=gate_rows))
    outputs["with_human"].update(render_side_yz_variant(args, out_root / "with_human", frames, meshes, mesh_cam, centers_cam, with_human=True, joints=joints, palm_rows=palm_rows, gate_rows=gate_rows))
    (out_root / "outputs.json").write_text(json.dumps(outputs, indent=2) + "\n")
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
