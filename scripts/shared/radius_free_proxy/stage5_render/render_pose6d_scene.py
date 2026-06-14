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
import json
import pickle
import shutil
import subprocess
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
LEFT_CONTACT_BGR = (70, 180, 70)
RIGHT_CONTACT_BGR = (70, 200, 230)
ACTIVE_CONTACT_BGR = (30, 30, 240)
FLOOR_LINE_BGR = (40, 180, 40)
FEET_LINE_BGR = (220, 120, 40)

# Readable body skeleton over the first 22 SMPL-X body joints.
BODY_EDGES = [
    (0, 1), (1, 4), (4, 7), (7, 10),
    (0, 2), (2, 5), (5, 8), (8, 11),
    (0, 3), (3, 6), (6, 9), (9, 12),
    (12, 13), (13, 16), (16, 18), (18, 20),
    (12, 14), (14, 17), (17, 19), (19, 21),
    (12, 15),
]


def find_ffmpeg() -> str | None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is not None:
        return ffmpeg
    candidates = [
        "/usr/bin/ffmpeg",
        "/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        str(Path.home() / "miniconda3" / "bin" / "ffmpeg"),
        str(Path.home() / "miniconda3" / "envs" / "audiohoi" / "bin" / "ffmpeg"),
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    return None


def pick_h264_encoder(preferred: str = "auto") -> str:
    if preferred != "auto":
        return preferred
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found; cannot write h264 output")
    try:
        result = subprocess.run(
            [ffmpeg, "-encoders"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("Failed to query ffmpeg encoders for h264 output") from exc
    encoders_text = result.stdout
    if "libx264" in encoders_text:
        return "libx264"
    if "libopenh264" in encoders_text:
        return "libopenh264"
    raise RuntimeError("No supported h264 encoder found (tried libx264, libopenh264)")


def make_video_writer(
    mp4_path: Path,
    fps: float,
    size: tuple[int, int],
    video_codec: str,
) -> tuple[cv2.VideoWriter, Path]:
    if video_codec == "mp4v":
        writer_path = mp4_path
    else:
        writer_path = mp4_path.with_name(f"{mp4_path.stem}.tmp_mp4v.mp4")
    writer = cv2.VideoWriter(str(writer_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {writer_path}")
    return writer, writer_path


def finalize_video_file(
    writer_path: Path,
    mp4_path: Path,
    video_codec: str,
    h264_encoder: str,
) -> Path:
    if video_codec == "mp4v":
        return mp4_path
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found; cannot finalize h264 output")
    tmp_h264_path = mp4_path.with_name(f"{mp4_path.stem}.tmp_h264.mp4")
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(writer_path),
        "-c:v",
        h264_encoder,
        "-pix_fmt",
        "yuv420p",
        str(tmp_h264_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else "unknown ffmpeg error"
        raise RuntimeError(f"ffmpeg h264 transcode failed for {mp4_path}: {stderr}") from exc
    if mp4_path.exists():
        mp4_path.unlink()
    tmp_h264_path.replace(mp4_path)
    if writer_path.exists():
        writer_path.unlink()
    return mp4_path



def figure_to_bgr(fig: plt.Figure) -> np.ndarray:
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)


def read_ball_pose(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    detected_human_part = "hand"
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            radius_m_raw = row.get("radius_m", "")
            radius_m = float(radius_m_raw) if radius_m_raw not in ("", None) else BALL_RADIUS_M
            contact_part_raw = str(row.get("contact_part", "") or "").strip().lower()
            active_part_raw = str(row.get("active_part", "") or "").strip().lower()
            if contact_part_raw in {"hand", "foot"}:
                detected_human_part = contact_part_raw
            elif active_part_raw.endswith("_hand"):
                detected_human_part = "hand"
            elif active_part_raw.endswith("_foot"):
                detected_human_part = "foot"
            rows.append(
                {
                    "frame": int(row["frame"]),
                    "time": float(row["time"]),
                    "x": float(row["tx"]),
                    "y": float(row["ty"]),
                    "z": float(row["tz"]),
                    "r": radius_m,
                    "contact_frame": int(row.get("contact_frame", 0) or 0),
                    "floor_v": float(row.get("floor_v", 0.0) or 0.0),
                    "bottom_proj_v": float(row.get("bottom_proj_v", 0.0) or 0.0),
                    "contact_part": row.get("contact_part", ""),
                    "contact_side": row.get("contact_side", ""),
                    "contact_label": row.get("contact_label", ""),
                    "active_part": row.get("active_part", ""),
                    "active_hand": row.get("active_hand", ""),
                    "active_foot": row.get("active_foot", ""),
                    "anchor_contact_event": int(row.get("anchor_contact_event", row.get("human_contact_event", 0)) or 0),
                    "human_contact_event": int(row.get("human_contact_event", row.get("anchor_contact_event", 0)) or 0),
                    "anchor_contact_state": int(row.get("anchor_contact_state", row.get("human_contact_state", 0)) or 0),
                    "human_contact_state": int(row.get("human_contact_state", row.get("anchor_contact_state", 0)) or 0),
                }
            )
    if not rows:
        raise RuntimeError(f"No ball rows found in {path}")
    for row in rows:
        row["default_human_part"] = detected_human_part
    return rows


def read_object_observations(path: Path) -> dict[int, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open() as f:
        return {int(row["frame"]): row for row in csv.DictReader(f)}


def parse_obs_float(row: dict[str, str] | None, key: str, default: float = np.nan) -> float:
    if row is None:
        return default
    value = row.get(key, "")
    if value in ("", None):
        return default
    try:
        return float(value)
    except ValueError:
        return default




def read_object_proxy(sample_dir: Path) -> dict[str, object]:
    proxy_path = sample_dir / "proxy" / "mug_proxy.json"
    if not proxy_path.exists():
        return {}
    try:
        return json.loads(proxy_path.read_text())
    except Exception:
        return {}


def proxy_part(proxy: dict[str, object] | None, name: str) -> dict[str, object]:
    if not isinstance(proxy, dict):
        return {}
    parts = proxy.get("parts")
    if not isinstance(parts, dict):
        return {}
    part = parts.get(name)
    return part if isinstance(part, dict) else {}


def proxy_contact_side(proxy: dict[str, object] | None, obs: dict[str, str] | None) -> str:
    if isinstance(proxy, dict):
        cr = proxy.get("contact_region")
        if isinstance(cr, dict):
            side = str(cr.get("side", "") or "").strip().lower()
            if side in {"left", "right"}:
                return side
    side = str((obs or {}).get("handle_side", "") or "").strip().lower()
    return side if side in {"left", "right"} else "right"


def proxy_float(part: dict[str, object], key: str, default: float) -> float:
    try:
        return float(part.get(key, default))
    except Exception:
        return default

def infer_object_kind(sample_dir: Path, object_obs: dict[int, dict[str, str]]) -> str:
    for row in object_obs.values():
        name = str(row.get("object_name", "") or "").strip().lower()
        if name:
            return name
    metadata_path = sample_dir / "metadata.json"
    if metadata_path.exists():
        import json
        try:
            return str(json.loads(metadata_path.read_text()).get("name", "") or "").strip().lower()
        except Exception:
            return ""
    return ""


def draw_mug_sprite(frame: np.ndarray, center: tuple[int, int], radius: int, obs: dict[str, str] | None, object_proxy: dict[str, object] | None = None) -> None:
    overlay = frame.copy()
    cx, cy = center
    body_x1 = parse_obs_float(obs, "body_bbox_x1")
    body_y1 = parse_obs_float(obs, "body_bbox_y1")
    body_x2 = parse_obs_float(obs, "body_bbox_x2")
    body_y2 = parse_obs_float(obs, "body_bbox_y2")
    if all(np.isfinite(v) for v in [body_x1, body_y1, body_x2, body_y2]):
        x1, y1, x2, y2 = [int(round(v)) for v in [body_x1, body_y1, body_x2, body_y2]]
    else:
        body = proxy_part(object_proxy, "body")
        body_h_norm = proxy_float(body, "height", 1.0)
        body_r_norm = proxy_float(body, "radius", 0.40)
        h = max(18, int(radius * 2.35 * body_h_norm))
        w = max(14, int(h * body_r_norm * 1.85))
        x1, y1, x2, y2 = cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    bw = max(8, x2 - x1)
    bh = max(12, y2 - y1)
    mug_fill = (214, 232, 238)
    mug_edge = (48, 72, 86)
    rim_fill = (238, 247, 249)
    shadow = (92, 117, 124)

    # Articraft-backed mug body: cylindrical side walls plus distinct rim/bottom.
    body_poly = np.asarray([
        [x1 + bw // 10, y1 + bh // 12],
        [x2 - bw // 10, y1 + bh // 12],
        [x2 - bw // 7, y2 - bh // 12],
        [x1 + bw // 7, y2 - bh // 12],
    ], dtype=np.int32)
    cv2.fillConvexPoly(overlay, body_poly, mug_fill, cv2.LINE_AA)
    cv2.polylines(overlay, [body_poly], True, mug_edge, 2, cv2.LINE_AA)
    rim_center = ((x1 + x2) // 2, y1 + max(3, bh // 10))
    rim_axes = (max(5, bw // 2), max(3, bh // 10))
    cv2.ellipse(overlay, rim_center, rim_axes, 0, 0, 360, rim_fill, -1, cv2.LINE_AA)
    cv2.ellipse(overlay, rim_center, rim_axes, 0, 0, 360, mug_edge, 2, cv2.LINE_AA)
    inner_axes = (max(3, int(rim_axes[0] * 0.72)), max(2, int(rim_axes[1] * 0.55)))
    cv2.ellipse(overlay, rim_center, inner_axes, 0, 0, 360, shadow, 1, cv2.LINE_AA)
    bottom_center = ((x1 + x2) // 2, y2 - max(3, bh // 12))
    cv2.ellipse(overlay, bottom_center, (max(4, bw // 3), max(2, bh // 16)), 0, 0, 360, mug_edge, 1, cv2.LINE_AA)

    side = proxy_contact_side(object_proxy, obs)
    sign = -1 if side == "left" else 1
    handle_color = (92, 155, 224)
    hx_outer = x1 - max(9, bw // 3) if sign < 0 else x2 + max(9, bw // 3)
    hx_inner = x1 + max(2, bw // 18) if sign < 0 else x2 - max(2, bw // 18)
    hy_top = y1 + int(0.34 * bh)
    hy_bot = y1 + int(0.70 * bh)
    hy_mid = (hy_top + hy_bot) // 2
    handle_pts = np.asarray([[hx_inner, hy_top], [hx_outer, hy_top], [hx_outer, hy_mid], [hx_outer, hy_bot], [hx_inner, hy_bot]], dtype=np.int32)
    cv2.polylines(overlay, [handle_pts], False, handle_color, max(3, bw // 10), cv2.LINE_AA)
    cv2.polylines(overlay, [handle_pts], False, mug_edge, 1, cv2.LINE_AA)
    cv2.circle(overlay, (hx_inner, hy_top), max(2, bw // 20), mug_edge, -1, cv2.LINE_AA)
    cv2.circle(overlay, (hx_inner, hy_bot), max(2, bw // 20), mug_edge, -1, cv2.LINE_AA)

    pad_x = max(20, bw // 2)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.rectangle(
        mask,
        (max(0, min(x1, hx_outer) - pad_x // 4), max(0, y1 - 18)),
        (min(frame.shape[1] - 1, max(x2, hx_outer) + pad_x // 4), min(frame.shape[0] - 1, y2 + 18)),
        255,
        -1,
    )
    mask_f = (mask.astype(np.float32) / 255.0)[:, :, None] * 0.86
    frame[:] = (frame.astype(np.float32) * (1.0 - mask_f) + overlay.astype(np.float32) * mask_f).astype(np.uint8)


def draw_object_sprite(frame: np.ndarray, ball: dict[str, float], K: np.ndarray, object_kind: str, obs: dict[str, str] | None, object_proxy: dict[str, object] | None = None) -> None:
    proj = project_ball(ball, K)
    if proj is None:
        return
    center, radius = proj
    if object_kind == "mug":
        draw_mug_sprite(frame, center, radius, obs, object_proxy)
    else:
        draw_object_sprite(frame, ball, K, object_kind, object_obs.get(int(ball["frame"])), object_proxy)


def draw_box_wire_3d(ax, center: tuple[float, float, float], size: tuple[float, float, float], color: str, linewidth: float = 2.0, alpha: float = 0.95) -> None:
    cx, cz, cy = center
    sx, sz, sy = size[0] / 2.0, size[1] / 2.0, size[2] / 2.0
    corners = np.asarray([
        [cx - sx, cz - sz, cy - sy], [cx + sx, cz - sz, cy - sy], [cx + sx, cz + sz, cy - sy], [cx - sx, cz + sz, cy - sy],
        [cx - sx, cz - sz, cy + sy], [cx + sx, cz - sz, cy + sy], [cx + sx, cz + sz, cy + sy], [cx - sx, cz + sz, cy + sy],
    ])
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for a, b in edges:
        seg = corners[[a, b]]
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=color, linewidth=linewidth, alpha=alpha)


def draw_mug_3d(ax, obj_cam: np.ndarray, obs: dict[str, str] | None, scale: float = 1.0, object_proxy: dict[str, object] | None = None) -> None:
    obj = cam_to_worldlike(obj_cam[None, :])[0]
    x, z, y = float(obj[0]), float(obj[2]), float(obj[1])
    body = proxy_part(object_proxy, "body")
    rim = proxy_part(object_proxy, "rim")
    bottom = proxy_part(object_proxy, "bottom")
    handle = proxy_part(object_proxy, "handle")
    total_height = 0.115 * scale
    if isinstance(object_proxy, dict):
        units = object_proxy.get("units")
        if isinstance(units, dict):
            total_height = max(0.04, float(units.get("total_height_m", 0.102))) * scale
    body_r = proxy_float(body, "radius", 0.392157) * total_height
    body_h = proxy_float(body, "height", 0.931373) * total_height
    rim_r = proxy_float(rim, "radius", 0.431373) * total_height
    bottom_r = proxy_float(bottom, "radius", body_r / max(total_height, 1e-6)) * total_height
    theta = np.linspace(0, 2 * np.pi, 48)
    # Articraft-style hollow cylindrical mug: body wall, rim, bottom.
    for yy, rr, color, lw, alpha in [
        (y - body_h / 2, bottom_r, "#314d55", 1.5, 0.72),
        (y + body_h / 2, rim_r, "#253d45", 2.2, 0.96),
    ]:
        ax.plot(x + rr * np.cos(theta), np.full_like(theta, z), yy + 0.35 * rr * np.sin(theta), color=color, linewidth=lw, alpha=alpha)
    for sx in [-1, 1]:
        ax.plot([x + sx * body_r, x + sx * body_r], [z, z], [y - body_h / 2, y + body_h / 2], color="#314d55", linewidth=2.0, alpha=0.9)
    ax.scatter([x], [z], [y], s=135, color="#d6e8ee", edgecolors="#25353a", linewidths=1.0, depthshade=False)

    side = proxy_contact_side(object_proxy, obs)
    sign = -1.0 if side == "left" else 1.0
    elements = handle.get("elements") if isinstance(handle, dict) else None
    if isinstance(elements, dict):
        for name, payload in elements.items():
            if not isinstance(payload, dict):
                continue
            c = payload.get("center_m", [0.0, 0.0, 0.0])
            size = payload.get("size_m", [0.01, 0.01, 0.01])
            try:
                cx_m, _cy_m, cz_m = [float(v) * scale for v in c]
                sx_m, sy_m, sz_m = [float(v) * scale for v in size]
            except Exception:
                continue
            # payload center_m is local metric [x, y, z_up] after mirroring; render axes are X, Zdepth, Yup.
            draw_box_wire_3d(ax, (x + cx_m, z, y + cz_m), (sx_m, 0.018 * scale, sz_m), "#1f77b4", linewidth=2.4 if name == "outer_grip" else 2.0, alpha=0.96)
    else:
        phi = np.linspace(-np.pi * 0.70, np.pi * 0.70, 28)
        hx = x + sign * (body_r + 0.020 * scale + 0.018 * scale * np.cos(phi))
        hz = np.full_like(phi, z)
        hy = y + 0.030 * scale * np.sin(phi)
        ax.plot(hx, hz, hy, color="#1f77b4", linewidth=3.0, alpha=0.95)

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


def build_palm_centers(joints_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left_ids = [20, 25, 28, 31, 34]
    right_ids = [21, 40, 43, 46, 49]
    left_palm = joints_cam[left_ids].mean(axis=0)
    right_palm = joints_cam[right_ids].mean(axis=0)
    return left_palm, right_palm


def build_foot_centers(joints_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return joints_cam[10], joints_cam[11]


def infer_contact_part(ball: dict[str, float], default_human_part: str = "hand") -> tuple[str, str | None]:
    contact_part = str(ball.get("contact_part", "") or "").strip().lower()
    contact_side = str(ball.get("contact_side", "") or "").strip().lower()
    active_part = str(ball.get("active_part", "") or "").strip().lower()

    # Rendering should show the candidate human part for the whole sequence,
    # including floor/support frames. Prefer active_part over contact_part so a
    # floor event can still display the relevant hand/foot candidate marker.
    if active_part.endswith("_hand"):
        side = active_part.split("_", 1)[0]
        return "hand", side if side in {"left", "right"} else None
    if active_part.endswith("_foot"):
        side = active_part.split("_", 1)[0]
        return "foot", side if side in {"left", "right"} else None

    active_foot = str(ball.get("active_foot", "") or "").strip().lower()
    if active_foot in {"left", "right"}:
        return "foot", active_foot
    active_hand = str(ball.get("active_hand", "") or "").strip().lower()
    if active_hand in {"left", "right"}:
        return "hand", active_hand

    if contact_part in {"hand", "foot"}:
        side = contact_side if contact_side in {"left", "right"} else None
        return contact_part, side
    if contact_part == "floor":
        return default_human_part, None
    return default_human_part, None


def choose_active_contact_proxy(joints_cam: np.ndarray, ball_xyz: np.ndarray, contact_part: str, active_side: str | None = None) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    if contact_part == "floor":
        return None, None, None
    if contact_part == "foot":
        left_proxy, right_proxy = build_foot_centers(joints_cam)
    else:
        left_proxy, right_proxy = build_palm_centers(joints_cam)
    if active_side == "left":
        return left_proxy, right_proxy, "left"
    if active_side == "right":
        return left_proxy, right_proxy, "right"
    return left_proxy, right_proxy, None


def should_draw_human_contact_proxy(ball: dict[str, float], contact_part: str) -> bool:
    return contact_part in {"hand", "foot"}


def should_highlight_anchor_contact(ball: dict[str, float]) -> bool:
    return bool(
        ball.get("anchor_contact_event", 0)
        or ball.get("human_contact_event", 0)
        or ball.get("contact_frame", 0)
    )


def draw_contact_markers_overlay(frame: np.ndarray, joints_cam: np.ndarray, ball_xyz: np.ndarray, K: np.ndarray, contact_part: str, active_side: str | None = None, highlight_active: bool = False) -> str | None:
    left_proxy, right_proxy, active_name = choose_active_contact_proxy(joints_cam, ball_xyz, contact_part, active_side)
    if left_proxy is None or right_proxy is None:
        return None
    proxy_points = np.stack([left_proxy, right_proxy], axis=0)
    proxy_uv, proxy_valid = project_points(proxy_points, K)

    if proxy_valid[0]:
        pt = tuple(np.round(proxy_uv[0]).astype(int))
        cv2.circle(frame, pt, 7, LEFT_CONTACT_BGR, -1, lineType=cv2.LINE_AA)
        cv2.circle(frame, pt, 9, (20, 20, 20), 1, lineType=cv2.LINE_AA)
    if proxy_valid[1]:
        pt = tuple(np.round(proxy_uv[1]).astype(int))
        cv2.circle(frame, pt, 7, RIGHT_CONTACT_BGR, -1, lineType=cv2.LINE_AA)
        cv2.circle(frame, pt, 9, (20, 20, 20), 1, lineType=cv2.LINE_AA)

    if active_name == "left":
        active_idx = 0
    elif active_name == "right":
        active_idx = 1
    else:
        return None
    if proxy_valid[active_idx]:
        if highlight_active:
            pt = tuple(np.round(proxy_uv[active_idx]).astype(int))
            cv2.circle(frame, pt, 12, ACTIVE_CONTACT_BGR, 2, lineType=cv2.LINE_AA)
        return active_name
    return None


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

def compute_feet_support_v(joints_cam: np.ndarray, K: np.ndarray) -> tuple[float | None, tuple[float | None, float | None]]:
    support_ids = [7, 8, 10, 11]
    pts = joints_cam[support_ids]
    uv, valid = project_points(pts, K)
    if not np.any(valid):
        return None, (None, None)
    left_vals = []
    right_vals = []
    if valid[0]:
        left_vals.append(float(uv[0, 1]))
    if valid[2]:
        left_vals.append(float(uv[2, 1]))
    if valid[1]:
        right_vals.append(float(uv[1, 1]))
    if valid[3]:
        right_vals.append(float(uv[3, 1]))
    left_v = max(left_vals) if left_vals else None
    right_v = max(right_vals) if right_vals else None
    vals = [v for v in [left_v, right_v] if v is not None]
    support_v = max(vals) if vals else None
    return support_v, (left_v, right_v)


def draw_support_lines_overlay(frame: np.ndarray, ball: dict[str, float], joints_cam: np.ndarray, K: np.ndarray) -> tuple[float | None, tuple[float | None, float | None]]:
    h, w = frame.shape[:2]
    floor_v = float(ball.get("floor_v", 0.0) or 0.0)
    if floor_v > 1e-6:
        y = int(round(floor_v))
        if 0 <= y < h:
            cv2.line(frame, (0, y), (w - 1, y), FLOOR_LINE_BGR, 2, cv2.LINE_AA)
            cv2.putText(frame, 'ball floor_v', (18, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, FLOOR_LINE_BGR, 1, cv2.LINE_AA)
    feet_support_v, (left_v, right_v) = compute_feet_support_v(joints_cam, K)
    if feet_support_v is not None:
        y = int(round(feet_support_v))
        if 0 <= y < h:
            cv2.line(frame, (0, y), (w - 1, y), FEET_LINE_BGR, 2, cv2.LINE_AA)
            cv2.putText(frame, 'feet support_v', (18, min(h - 12, y + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, FEET_LINE_BGR, 1, cv2.LINE_AA)
    return feet_support_v, (left_v, right_v)


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
    object_kind: str,
    object_obs: dict[int, dict[str, str]],
    fps: float,
    video_codec: str,
    h264_encoder: str,
    object_proxy: dict[str, object] | None = None,
) -> tuple[Path, Path]:
    frames_dir = sample_dir / "frames"
    first = cv2.imread(str(frames_dir / "00001.png"))
    if first is None:
        raise RuntimeError(f"Could not read frames from {frames_dir}")
    h, w = first.shape[:2]

    mp4_path = out_dir / "overlay.mp4"
    png_path = out_dir / "overlay_preview.png"
    writer, writer_path = make_video_writer(mp4_path, fps, (w, h), video_codec)

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
            f"frame {ball['frame']:03d}  t={ball['time']:.2f}s  x={ball['x']:.3f}  y={ball['y']:.3f}  z={ball['z']:.3f}",
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
    mp4_path = finalize_video_file(writer_path, mp4_path, video_codec, h264_encoder)
    return png_path, mp4_path


def render_overlay_with_human(
    sample_dir: Path,
    ball_rows: list[dict[str, float]],
    sampled_vertices: np.ndarray,
    joints: np.ndarray,
    K_fullimg: np.ndarray,
    out_dir: Path,
    object_kind: str,
    object_obs: dict[int, dict[str, str]],
    fps: float,
    video_codec: str,
    h264_encoder: str,
    object_proxy: dict[str, object] | None = None,
) -> tuple[Path, Path]:
    frames_dir = sample_dir / "frames"
    first = cv2.imread(str(frames_dir / "00001.png"))
    if first is None:
        raise RuntimeError(f"Could not read frames from {frames_dir}")
    h, w = first.shape[:2]

    mp4_path = out_dir / "overlay.mp4"
    png_path = out_dir / "overlay_preview.png"
    writer, writer_path = make_video_writer(mp4_path, fps, (w, h), video_codec)

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
        feet_support_v, (left_foot_v, right_foot_v) = draw_support_lines_overlay(frame, ball, joints[idx], K)

        ball_xyz = np.asarray([ball["x"], ball["y"], ball["z"]], dtype=np.float32)
        contact_part, active_side = infer_contact_part(ball, str(ball.get("default_human_part", "hand")))
        active_proxy_name = None
        if should_draw_human_contact_proxy(ball, contact_part):
            active_proxy_name = draw_contact_markers_overlay(
                frame,
                joints[idx],
                ball_xyz,
                K,
                contact_part,
                active_side,
                highlight_active=should_highlight_anchor_contact(ball),
            )

        proj = project_ball(ball, K)
        if proj is not None:
            center, radius = proj
            traj.append(center)
            if len(traj) >= 2:
                cv2.polylines(frame, [np.asarray(traj, dtype=np.int32)], False, (93, 126, 188), 2, cv2.LINE_AA)
            draw_object_sprite(frame, ball, K, object_kind, object_obs.get(int(ball["frame"])), object_proxy)

        contact_part, active_side = infer_contact_part(ball, str(ball.get("default_human_part", "hand")))
        left_proxy, right_proxy, _ = choose_active_contact_proxy(joints[idx], ball_xyz, contact_part, active_side)
        if left_proxy is not None and right_proxy is not None:
            left_dist = float(np.linalg.norm(left_proxy - ball_xyz))
            right_dist = float(np.linalg.norm(right_proxy - ball_xyz))
            dist_txt = f"  L={left_dist:.3f}m  R={right_dist:.3f}m"
        else:
            dist_txt = ""
        floor_v = float(ball.get("floor_v", 0.0) or 0.0)
        floor_gap = (feet_support_v - floor_v) if (feet_support_v is not None and floor_v > 1e-6) else None
        floor_gap_txt = f"  feet-floor={floor_gap:+.1f}px" if floor_gap is not None else ""
        cv2.putText(
            frame,
            f"frame {ball['frame']:03d}  t={ball['time']:.2f}s  x={ball['x']:.3f}  y={ball['y']:.3f}  z={ball['z']:.3f}",
            (24, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            (22, 22, 22),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"frame {ball['frame']:03d}  t={ball['time']:.2f}s  part={contact_part}{dist_txt}  active={active_proxy_name or 'n/a'}{floor_gap_txt}",
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
    mp4_path = finalize_video_file(writer_path, mp4_path, video_codec, h264_encoder)
    return png_path, mp4_path


def render_camera3d(
    ball_rows: list[dict[str, float]],
    joints: np.ndarray | None,
    sampled_vertices: np.ndarray | None,
    out_dir: Path,
    object_kind: str,
    object_obs: dict[int, dict[str, str]],
    fps: float,
    width: int,
    height: int,
    with_human: bool,
    video_codec: str,
    h264_encoder: str,
    object_proxy: dict[str, object] | None = None,
) -> tuple[Path, Path]:
    mp4_path = out_dir / "camera3d.mp4"
    png_path = out_dir / "camera3d_preview.png"
    writer, writer_path = make_video_writer(mp4_path, fps, (width, height), video_codec)

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
        if object_kind == "mug":
            draw_mug_3d(ax, ball_xyz_cam[idx], object_obs.get(int(ball["frame"])), object_proxy=object_proxy)
        else:
            ax.scatter([ball_now[0]], [ball_now[2]], [ball_now[1]], s=180, color="#db7a20", edgecolors="#1f1f1f", linewidths=1.0, depthshade=False)

        if with_human and sampled_vertices is not None and joints is not None:
            verts = cam_to_worldlike(sampled_vertices[idx])
            ax.scatter(verts[:, 0], verts[:, 2], verts[:, 1], s=5.0, color="#9c89f2", alpha=0.22, depthshade=False, linewidths=0)
            body_j = cam_to_worldlike(joints[idx, :22, :])
            for a, b in BODY_EDGES:
                seg = body_j[[a, b]]
                ax.plot(seg[:, 0], seg[:, 2], seg[:, 1], color="#6b54d2", linewidth=2.6, alpha=0.95)
            ax.scatter(body_j[:, 0], body_j[:, 2], body_j[:, 1], s=18, color="#e4dbff", edgecolors="#6b54d2", linewidths=0.4, depthshade=False)
            contact_part, active_side = infer_contact_part(ball, str(ball.get("default_human_part", "hand")))
            left_proxy_cam, right_proxy_cam, active_name = choose_active_contact_proxy(joints[idx], ball_xyz_cam[idx], contact_part, active_side)
            if should_draw_human_contact_proxy(ball, contact_part) and left_proxy_cam is not None and right_proxy_cam is not None:
                left_proxy = cam_to_worldlike(left_proxy_cam[None, :])[0]
                right_proxy = cam_to_worldlike(right_proxy_cam[None, :])[0]
                ax.scatter([left_proxy[0]], [left_proxy[2]], [left_proxy[1]], s=80, color="#46b446", edgecolors="#1f1f1f", linewidths=0.8, depthshade=False)
                ax.scatter([right_proxy[0]], [right_proxy[2]], [right_proxy[1]], s=80, color="#46c8e6", edgecolors="#1f1f1f", linewidths=0.8, depthshade=False)
                if active_name in {"left", "right"} and should_highlight_anchor_contact(ball):
                    active_proxy = left_proxy if active_name == "left" else right_proxy
                    ax.scatter([active_proxy[0]], [active_proxy[2]], [active_proxy[1]], s=180, facecolors="none", edgecolors="#ee2b2b", linewidths=2.0, depthshade=False)
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
    mp4_path = finalize_video_file(writer_path, mp4_path, video_codec, h264_encoder)
    return png_path, mp4_path


def render_side_yz(
    ball_rows: list[dict[str, float]],
    joints: np.ndarray | None,
    sampled_vertices: np.ndarray | None,
    out_dir: Path,
    object_kind: str,
    object_obs: dict[int, dict[str, str]],
    fps: float,
    width: int,
    height: int,
    with_human: bool,
    video_codec: str,
    h264_encoder: str,
    object_proxy: dict[str, object] | None = None,
) -> tuple[Path, Path]:
    mp4_path = out_dir / "side_yz.mp4"
    png_path = out_dir / "side_yz_preview.png"
    writer, writer_path = make_video_writer(mp4_path, fps, (width, height), video_codec)

    ball_xyz_cam = np.asarray([[r["x"], r["y"], r["z"]] for r in ball_rows], dtype=np.float32)
    ball_xyz = cam_to_worldlike(ball_xyz_cam)
    all_pts = [ball_xyz]
    if with_human and sampled_vertices is not None:
        all_pts.append(cam_to_worldlike(sampled_vertices.reshape(-1, 3)))
    all_pts = np.concatenate(all_pts, axis=0)

    y_vals = all_pts[:, 1]
    z_vals = all_pts[:, 2]
    margin_y = max(0.10, 0.20 * float(y_vals.max() - y_vals.min()))
    margin_z = max(0.10, 0.20 * float(z_vals.max() - z_vals.min()))

    preview_written = False
    for idx, ball in enumerate(ball_rows):
        fig = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100, facecolor="#f7f8fb")
        ax = fig.add_subplot(111)
        ax.set_facecolor("#f7f8fb")
        ax.set_xlim(float(z_vals.min() - margin_z), float(z_vals.max() + margin_z))
        ax.set_ylim(float(y_vals.min() - margin_y), float(y_vals.max() + margin_y))
        ax.set_xlabel("Z_cam (depth, m)")
        ax.set_ylabel("Y_worldlike (up, m)")
        ax.set_title("Shared-camera Y-Z side view")
        ax.grid(True, alpha=0.18)

        ax.plot(ball_xyz[:, 2], ball_xyz[:, 1], color="#d1d7e1", linewidth=1.2, alpha=0.55)
        ax.plot(ball_xyz[: idx + 1, 2], ball_xyz[: idx + 1, 1], color="#4c72b0", linewidth=2.4, alpha=0.95)
        ball_now = ball_xyz[idx]
        if object_kind == "mug":
            ax.scatter([ball_now[2]], [ball_now[1]], s=170, color="#d6e8ee", edgecolors="#25353a", linewidths=1.0, zorder=5)
            obs = object_obs.get(int(ball["frame"]))
            side = str((obs or {}).get("handle_side", "") or "").strip().lower()
            visible = str((obs or {}).get("handle_visible", "") or "") == "1"
            if side in {"left", "right"}:
                ax.scatter([ball_now[2]], [ball_now[1]], s=300, facecolors="none", edgecolors="#1f77b4" if visible else "#9aa7b2", linewidths=2.0, zorder=6)
        else:
            ax.scatter([ball_now[2]], [ball_now[1]], s=180, color="#db7a20", edgecolors="#1f1f1f", linewidths=1.0, zorder=5)

        if with_human and sampled_vertices is not None and joints is not None:
            verts = cam_to_worldlike(sampled_vertices[idx])
            ax.scatter(verts[:, 2], verts[:, 1], s=5.0, color="#9c89f2", alpha=0.22, linewidths=0)
            body_j = cam_to_worldlike(joints[idx, :22, :])
            for a, b in BODY_EDGES:
                seg = body_j[[a, b]]
                ax.plot(seg[:, 2], seg[:, 1], color="#6b54d2", linewidth=2.6, alpha=0.95)
            ax.scatter(body_j[:, 2], body_j[:, 1], s=18, color="#e4dbff", edgecolors="#6b54d2", linewidths=0.4, zorder=4)
            contact_part, active_side = infer_contact_part(ball, str(ball.get("default_human_part", "hand")))
            left_proxy_cam, right_proxy_cam, active_name = choose_active_contact_proxy(joints[idx], ball_xyz_cam[idx], contact_part, active_side)
            if should_draw_human_contact_proxy(ball, contact_part) and left_proxy_cam is not None and right_proxy_cam is not None:
                left_proxy = cam_to_worldlike(left_proxy_cam[None, :])[0]
                right_proxy = cam_to_worldlike(right_proxy_cam[None, :])[0]
                ax.scatter([left_proxy[2]], [left_proxy[1]], s=80, color="#46b446", edgecolors="#1f1f1f", linewidths=0.8, zorder=6)
                ax.scatter([right_proxy[2]], [right_proxy[1]], s=80, color="#46c8e6", edgecolors="#1f1f1f", linewidths=0.8, zorder=6)
                if active_name in {"left", "right"} and should_highlight_anchor_contact(ball):
                    active_proxy = left_proxy if active_name == "left" else right_proxy
                    ax.scatter([active_proxy[2]], [active_proxy[1]], s=180, facecolors="none", edgecolors="#ee2b2b", linewidths=2.0, zorder=7)

        y_worldlike = -ball["y"]
        fig.text(
            0.05,
            0.05,
            f"frame {ball['frame']:03d}   t={ball['time']:.2f}s   Y={y_worldlike:.3f}m   Z={ball['z']:.3f}m",
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
    mp4_path = finalize_video_file(writer_path, mp4_path, video_codec, h264_encoder)
    return png_path, mp4_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-dir", type=Path, default=Path("samples_known_object/02_mug"))
    parser.add_argument("--body-model-root", type=Path, default=Path("third-party/GVHMR/inputs/checkpoints/body_models"))
    parser.add_argument(
        "--ball-csv",
        type=Path,
        default=None,
        help="Path to the object pose trajectory CSV to render. Defaults to radius-free stage2 output.",
    )
    parser.add_argument(
        "--render-tag",
        type=str,
        default="pose6d_object_proxy_anchor_refined",
        help="Subdirectory name under results/renders for this render batch.",
    )
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--vertex-stride", type=int, default=12)
    parser.add_argument("--with-human", action="store_true")
    parser.add_argument(
        "--video-codec",
        choices=["h264", "mp4v"],
        default="h264",
        help="Final output codec. h264 writes mp4v first, then transcodes with ffmpeg.",
    )
    parser.add_argument(
        "--h264-encoder",
        default="auto",
        help="ffmpeg encoder for h264 output: auto, libx264, or libopenh264.",
    )
    args = parser.parse_args()
    h264_encoder = pick_h264_encoder(args.h264_encoder) if args.video_codec == "h264" else ""

    sample_dir = args.sample_dir
    results_dir = sample_dir / "results"
    out_root = results_dir / "renders" / args.render_tag
    ball_out = out_root / "ball"
    human_out = out_root / "with_human"
    ball_out.mkdir(parents=True, exist_ok=True)
    human_out.mkdir(parents=True, exist_ok=True)

    ball_csv = args.ball_csv or (results_dir / "pose6d_object_proxy_anchor_refined" / "object_pose6d_sharedcam_contactphase_trajectory.csv")
    ball_rows = read_ball_pose(ball_csv)
    object_obs = read_object_observations(results_dir / "object_observations" / "object_observations.csv")
    object_kind = infer_object_kind(sample_dir, object_obs)
    object_proxy = read_object_proxy(sample_dir)
    human = read_human_result(results_dir / "gvhmr" / "result.pkl")
    if len(ball_rows) != human["transl"].shape[0]:
        raise RuntimeError("Ball/human frame count mismatch")

    # Ball-only outputs always get written.
    ball_overlay_png, ball_overlay_mp4 = render_overlay_ball_only(
        sample_dir, ball_rows, human["K_fullimg"][0], ball_out, object_kind, object_obs, args.fps, args.video_codec, h264_encoder, object_proxy
    )
    ball_cam3d_png, ball_cam3d_mp4 = render_camera3d(
        ball_rows, None, None, ball_out, object_kind, object_obs, args.fps, args.width, args.height, with_human=False, video_codec=args.video_codec, h264_encoder=h264_encoder, object_proxy=object_proxy
    )
    ball_side_png, ball_side_mp4 = render_side_yz(
        ball_rows, None, None, ball_out, object_kind, object_obs, args.fps, args.width, args.height, with_human=False, video_codec=args.video_codec, h264_encoder=h264_encoder, object_proxy=object_proxy
    )
    print(f"ball_csv: {ball_csv}")
    print(f"ball_overlay_preview: {ball_overlay_png}")
    print(f"ball_overlay_mp4: {ball_overlay_mp4}")
    print(f"ball_camera3d_preview: {ball_cam3d_png}")
    print(f"ball_camera3d_mp4: {ball_cam3d_mp4}")
    print(f"ball_side_yz_preview: {ball_side_png}")
    print(f"ball_side_yz_mp4: {ball_side_mp4}")

    if args.with_human:
        joints, sampled_vertices = build_body_outputs(args.body_model_root, human, args.vertex_stride)
        human_overlay_png, human_overlay_mp4 = render_overlay_with_human(
            sample_dir,
            ball_rows,
            sampled_vertices,
            joints,
            human["K_fullimg"],
            human_out,
            object_kind,
            object_obs,
            args.fps,
            args.video_codec,
            h264_encoder,
            object_proxy,
        )
        human_cam3d_png, human_cam3d_mp4 = render_camera3d(
            ball_rows,
            joints,
            sampled_vertices,
            human_out,
            object_kind,
            object_obs,
            args.fps,
            args.width,
            args.height,
            with_human=True,
            video_codec=args.video_codec,
            h264_encoder=h264_encoder,
            object_proxy=object_proxy,
        )
        human_side_png, human_side_mp4 = render_side_yz(
            ball_rows,
            joints,
            sampled_vertices,
            human_out,
            object_kind,
            object_obs,
            args.fps,
            args.width,
            args.height,
            with_human=True,
            video_codec=args.video_codec,
            h264_encoder=h264_encoder,
            object_proxy=object_proxy,
        )
        print(f"human_overlay_preview: {human_overlay_png}")
        print(f"human_overlay_mp4: {human_overlay_mp4}")
        print(f"human_camera3d_preview: {human_cam3d_png}")
        print(f"human_camera3d_mp4: {human_cam3d_mp4}")
        print(f"human_side_yz_preview: {human_side_png}")
        print(f"human_side_yz_mp4: {human_side_mp4}")


if __name__ == "__main__":
    main()
