#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import smplx
import torch
from scipy.signal import savgol_filter


REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.shared.generic_contact_pipeline.components.contact.contact_part_utils import build_contact_part_centers


def parse_scalar(value: object, default: float = math.nan) -> float:
    if value in {"", None}:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows found in {path}")
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})
    return path


def read_object_mesh_tracks(path: Path) -> dict[int, dict[str, np.ndarray]]:
    by_frame: dict[int, list[dict[str, str]]] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            by_frame.setdefault(int(float(row["frame"])), []).append(row)
    out: dict[int, dict[str, np.ndarray]] = {}
    for frame, rows in by_frame.items():
        xy = []
        visible = []
        is_boundary = []
        point_ids = []
        for row in rows:
            x = row.get("x", "")
            y = row.get("y", "")
            if x == "" or y == "":
                continue
            xy.append([float(x), float(y)])
            visible.append(float(row.get("visible", 1.0) or 1.0) > 0.5)
            is_boundary.append(str(row.get("point_type", "")) == "boundary")
            point_ids.append(str(row.get("point_id", "")))
        if xy:
            out[frame] = {
                "xy": np.asarray(xy, dtype=np.float64),
                "visible": np.asarray(visible, dtype=bool),
                "is_boundary": np.asarray(is_boundary, dtype=bool),
                "point_ids": np.asarray(point_ids, dtype=object),
            }
    return out


def read_depth_index(path: Path) -> dict[int, Path]:
    depth_dir = path.parent
    out: dict[int, Path] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out[int(float(row["frame"]))] = depth_dir / row["file"]
    if not out:
        raise RuntimeError(f"No DA3 scene-depth entries in {path}")
    return out


def load_depth(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
    elif path.suffix.lower() == ".npz":
        data = np.load(path)
        arr = data[next(iter(data.files))]
    else:
        import cv2

        arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise RuntimeError(f"Could not load depth map: {path}")
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


def get_video_hw(video_path: Path) -> tuple[int, int]:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video size from {video_path}: {(width, height)}")
    return width, height


def map_to_depth_uv(u: float, v: float, src_hw: tuple[int, int], depth_hw: tuple[int, int]) -> tuple[float, float]:
    src_w, src_h = src_hw
    depth_h, depth_w = depth_hw
    return u * depth_w / float(src_w), v * depth_h / float(src_h)


def sample_depth_at_uv(depth: np.ndarray, u: float, v: float, window: int = 5) -> tuple[float, float]:
    if not np.isfinite(u) or not np.isfinite(v):
        return math.nan, 0.0
    h, w = depth.shape[:2]
    x = int(round(u))
    y = int(round(v))
    if x < 0 or x >= w or y < 0 or y >= h:
        return math.nan, 0.0
    r = max(0, int(window))
    x1 = max(0, x - r)
    x2 = min(w, x + r + 1)
    y1 = max(0, y - r)
    y2 = min(h, y + r + 1)
    patch = np.asarray(depth[y1:y2, x1:x2], dtype=np.float64)
    vals = patch[np.isfinite(patch) & (patch > 0)]
    if vals.size == 0:
        return math.nan, 0.0
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    conf = float(1.0 / (1.0 + mad / max(abs(med), 1e-6)))
    return med, conf


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


def build_body_joints(body_models_root: Path, human_params: dict[str, np.ndarray]) -> np.ndarray:
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
            return_verts=False,
        )
    return output.joints.detach().cpu().numpy().astype(np.float64)


def project_points(points_cam: np.ndarray, k_mats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = np.clip(points_cam[:, 2], 1e-6, None)
    u = k_mats[:, 0, 0] * (points_cam[:, 0] / z) + k_mats[:, 0, 2]
    v = k_mats[:, 1, 1] * (points_cam[:, 1] / z) + k_mats[:, 1, 2]
    valid = points_cam[:, 2] > 1e-6
    return np.stack([u, v], axis=1), valid


def visible_mesh_points(mesh: dict[str, np.ndarray], boundary_preferred: bool = False) -> tuple[np.ndarray, np.ndarray]:
    valid = mesh["visible"].copy()
    if boundary_preferred:
        boundary_valid = valid & mesh["is_boundary"]
        if np.any(boundary_valid):
            valid = boundary_valid
    if not np.any(valid):
        return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=object)
    return mesh["xy"][valid], mesh["point_ids"][valid]


def select_active_body_proxy_from_mesh(
    mesh: dict[str, np.ndarray],
    part_centers: dict[str, np.ndarray],
    idx: int,
    k_mats: np.ndarray,
) -> tuple[str, np.ndarray | None, float, float]:
    pts, _ = visible_mesh_points(mesh, boundary_preferred=True)
    if len(pts) == 0:
        return "", None, math.nan, 0.0
    best_label = ""
    best_uv = None
    best_z = math.nan
    best_dist = math.inf
    for label, centers in part_centers.items():
        point = centers[idx : idx + 1]
        uv, valid = project_points(point, k_mats[idx : idx + 1])
        if not bool(valid[0]):
            continue
        cand_uv = np.asarray(uv[0], dtype=np.float64)
        if not np.all(np.isfinite(cand_uv)):
            continue
        dist = float(np.min(np.linalg.norm(pts - cand_uv[None, :], axis=1)))
        if dist < best_dist:
            best_dist = dist
            best_label = label
            best_uv = cand_uv
            best_z = float(point[0, 2])
    if best_uv is None:
        return "", None, math.nan, 0.0
    conf = float(np.clip(np.exp(-best_dist / 40.0), 0.0, 1.0))
    return best_label, best_uv, best_z, conf


def select_contact_proxy_from_human_point(
    mesh: dict[str, np.ndarray],
    human_uv: np.ndarray | None,
    distance_sigma_px: float = 40.0,
    top_k: int = 8,
) -> tuple[float, float, float, str, str]:
    if human_uv is None or not np.all(np.isfinite(human_uv)):
        return math.nan, math.nan, 0.0, "missing_human_proxy", ""
    pts, ids = visible_mesh_points(mesh, boundary_preferred=False)
    if len(pts) == 0:
        return math.nan, math.nan, 0.0, "missing_object_mesh", ""
    valid = mesh["visible"].copy()
    kept_boundary = mesh["is_boundary"][valid]
    dist = np.linalg.norm(pts - np.asarray(human_uv, dtype=np.float64)[None, :], axis=1)
    scores = np.exp(-dist / max(distance_sigma_px, 1e-6)) * np.where(kept_boundary, 1.25, 1.0)
    order = np.argsort(-scores)
    selected = order[: min(top_k, len(order))]
    selected_scores = scores[selected]
    weight_sum = float(np.sum(selected_scores))
    if weight_sum <= 1e-8:
        return math.nan, math.nan, 0.0, "missing_weighted_contact_proxy", ""
    uv = np.sum(pts[selected] * selected_scores[:, None], axis=0) / weight_sum
    min_dist = float(np.min(dist))
    conf = float(np.clip(np.exp(-min_dist / distance_sigma_px), 0.0, 1.0))
    nearest_id = str(ids[int(np.argmin(dist))]) if len(ids) else ""
    return float(uv[0]), float(uv[1]), conf, "human_nearest_mesh_topk", nearest_id


def select_support_proxy_bottom_percentile(
    mesh: dict[str, np.ndarray], q: float = 95.0
) -> tuple[float, float, float, str]:
    pts, _ = visible_mesh_points(mesh, boundary_preferred=True)
    if len(pts) == 0:
        return math.nan, math.nan, 0.0, "missing_mesh"
    support_v = float(np.percentile(pts[:, 1], q))
    support_u = float(np.median(pts[:, 0][pts[:, 1] >= support_v - 1e-6]))
    conf = min(1.0, len(pts) / 8.0)
    return support_u, support_v, conf, "bottom_percentile_95"


def select_stable_ref_proxy(row: dict[str, str]) -> tuple[float, float, float, str]:
    candidates = [
        ("center", "center_x", "center_y", "observation_conf"),
        ("mask_center", "mask_center_x", "mask_center_y", "mask_conf"),
        ("track_center", "track_center_x", "track_center_y", "track_geometry_conf"),
        ("ref", "ref_u", "ref_v", "observation_conf"),
    ]
    for name, key_u, key_v, key_conf in candidates:
        u = parse_scalar(row.get(key_u, ""))
        v = parse_scalar(row.get(key_v, ""))
        if np.isfinite(u) and np.isfinite(v):
            conf = parse_scalar(row.get(key_conf, ""), 1.0)
            if not np.isfinite(conf):
                conf = 1.0
            return float(u), float(v), float(np.clip(conf, 0.0, 1.0)), name
    return math.nan, math.nan, 0.0, "missing_stable_ref"


def rolling_stat(values: np.ndarray, radius: int, mode: str) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype=np.float64)
    for i in range(len(values)):
        lo = max(0, i - radius)
        hi = min(len(values), i + radius + 1)
        win = values[lo:hi]
        win = win[np.isfinite(win)]
        if len(win) == 0:
            continue
        out[i] = float(np.median(win) if mode == "median" else np.std(win))
    return out


def rolling_abs_deviation(values: np.ndarray, window: int = 7) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    out = np.full(len(values), np.nan, dtype=np.float64)
    half = window // 2
    for i, value in enumerate(values):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        win = values[lo:hi]
        win = win[np.isfinite(win)]
        if len(win) == 0 or not np.isfinite(value):
            continue
        out[i] = abs(float(value) - float(np.median(win)))
    return out


def ema_filter(values: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype=np.float64)
    prev = math.nan
    for i, value in enumerate(values):
        if np.isfinite(value):
            prev = float(value) if not np.isfinite(prev) else alpha * prev + (1.0 - alpha) * float(value)
        out[i] = prev
    return out


def low_value_score(x: np.ndarray, p_good: float, p_bad: float) -> np.ndarray:
    return np.clip((p_bad - x) / max(p_bad - p_good, 1e-6), 0.0, 1.0)


def robust_smooth_signal(x: np.ndarray, window: int = 5, polyorder: int = 2) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    if not np.any(finite):
        return x.copy()
    filled = x.copy()
    idx = np.arange(len(x))
    filled[~finite] = np.interp(idx[~finite], idx[finite], x[finite])
    med = rolling_stat(filled, radius=1, mode="median")
    if len(med) < 5:
        return med
    win = min(window, len(med) if len(med) % 2 == 1 else len(med) - 1)
    if win < 3:
        return med
    return savgol_filter(med, window_length=win, polyorder=min(polyorder, win - 1), mode="interp")


def high_value_score(x: np.ndarray, p_bad: float, p_good: float) -> np.ndarray:
    return np.clip((x - p_bad) / max(p_good - p_bad, 1e-6), 0.0, 1.0)


def read_audio_events(path: Path) -> list[dict[str, float | int]]:
    if not path.exists():
        return []
    rows: list[dict[str, float | int]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("audio_frame"):
                continue
            rows.append(
                {
                    "audio_frame": int(float(row["audio_frame"])),
                    "audio_score": float(row.get("audio_score", 0.0) or 0.0),
                }
            )
    return rows


def build_audio_support(frames: np.ndarray, audio_rows: list[dict[str, float | int]], radius: int = 2) -> np.ndarray:
    support = np.zeros(len(frames), dtype=np.float64)
    frame_to_idx = {int(frame): i for i, frame in enumerate(frames.tolist())}
    for row in audio_rows:
        center = int(row["audio_frame"])
        score = float(row["audio_score"])
        for frame in range(center - radius, center + radius + 1):
            idx = frame_to_idx.get(frame)
            if idx is None:
                continue
            dist = abs(frame - center)
            support[idx] = max(support[idx], score * max(0.0, 1.0 - 0.25 * dist))
    return support


def build_proxy_depth(
    *,
    sample_dir: Path,
    object_observation_csv: Path,
    output_csv: Path,
    body_model_root: Path,
) -> Path:
    results_dir = sample_dir / "results"
    object_rows = read_rows(object_observation_csv)
    mesh_tracks = read_object_mesh_tracks(results_dir / "tracking" / "object_mesh_tracks_test.csv")
    frame_to_depth = read_depth_index(results_dir / "da3" / "scene_depth" / "index.csv")
    video_hw = get_video_hw(sample_dir / "video.mp4")
    audio_rows = read_audio_events(results_dir / "events" / "audio_events.csv")

    human = read_human_result(results_dir / "gvhmr" / "result.pkl")
    joints = build_body_joints(body_model_root, human)
    k_mats = np.asarray(human["K_fullimg"], dtype=np.float64)
    joints = joints[: len(object_rows)]
    k_mats = k_mats[: len(object_rows)]
    part_centers = build_contact_part_centers(joints)

    rows: list[dict[str, object]] = []
    for idx, row in enumerate(object_rows):
        frame = int(float(row["frame"]))
        mesh = mesh_tracks.get(frame)
        if mesh is None:
            continue
        label, active_uv, active_z, active_conf = select_active_body_proxy_from_mesh(mesh, part_centers, idx, k_mats)
        ref_u, ref_v, ref_conf, ref_source = select_stable_ref_proxy(row)
        support_u, support_v_raw, support_conf_raw, support_source = select_support_proxy_bottom_percentile(mesh)
        contact_u, contact_v, contact_conf, contact_source, contact_name = select_contact_proxy_from_human_point(mesh, active_uv)
        if not all(np.isfinite(v) for v in [ref_u, ref_v, support_u, support_v_raw, contact_u, contact_v]):
            continue
        depth_map = load_depth(frame_to_depth[frame])
        ref_ud, ref_vd = map_to_depth_uv(ref_u, ref_v, video_hw, depth_map.shape[:2])
        contact_ud, contact_vd = map_to_depth_uv(contact_u, contact_v, video_hw, depth_map.shape[:2])
        object_depth_raw, ref_depth_conf = sample_depth_at_uv(depth_map, ref_ud, ref_vd)
        contact_depth, contact_depth_conf = sample_depth_at_uv(depth_map, contact_ud, contact_vd)
        if not np.isfinite(object_depth_raw) or not np.isfinite(contact_depth):
            continue
        rows.append(
            {
                "frame": frame,
                "time": parse_scalar(row.get("time", "")),
                "ref_u": ref_u,
                "ref_v": ref_v,
                "ref_source": ref_source,
                "contact_u": contact_u,
                "contact_v": contact_v,
                "contact_source": contact_source,
                "support_u": support_u,
                "support_v_raw": support_v_raw,
                "support_source": support_source,
                "object_depth_raw": object_depth_raw,
                "contact_proxy_depth_m": contact_depth,
                "ref_conf": ref_conf,
                "support_conf_raw": support_conf_raw,
                "contact_conf": contact_conf,
                "depth_conf_raw": min(ref_depth_conf, contact_depth_conf),
                "active_label": label,
                "active_label_conf": active_conf,
                "active_part_u": active_uv[0] if active_uv is not None else math.nan,
                "active_part_v": active_uv[1] if active_uv is not None else math.nan,
                "active_part_z": active_z,
                "contact_proxy_name": contact_name,
            }
        )
    if not rows:
        raise RuntimeError("No valid ball proxy-depth rows built")

    depth_raw = np.asarray([float(r["object_depth_raw"]) for r in rows], dtype=np.float64)
    ref_u = np.asarray([float(r["ref_u"]) for r in rows], dtype=np.float64)
    ref_v = np.asarray([float(r["ref_v"]) for r in rows], dtype=np.float64)
    support_raw = np.asarray([float(r["support_v_raw"]) for r in rows], dtype=np.float64)
    frames = np.asarray([int(r["frame"]) for r in rows], dtype=np.int32)

    depth_local_med = rolling_stat(depth_raw, radius=6, mode="median")
    depth_local_std = rolling_stat(depth_raw, radius=6, mode="std")
    depth_outlier = np.abs(depth_raw - depth_local_med)
    gate_base = np.nanpercentile(depth_outlier, 75)
    gate_dynamic = np.where(np.isfinite(depth_local_std), np.maximum(0.35, 2.0 * depth_local_std), 0.35)
    gate = np.minimum(np.maximum(gate_dynamic, 0.20), max(float(gate_base), 0.35))
    depth_input = np.where(depth_outlier <= gate, np.clip(depth_raw, depth_local_med - gate, depth_local_med + gate), depth_local_med)
    depth_smooth = ema_filter(depth_input, alpha=0.45)
    depth_conf = low_value_score(depth_outlier, np.percentile(depth_outlier, 20), np.percentile(depth_outlier, 70))

    ref_u_smooth = robust_smooth_signal(ref_u, window=5, polyorder=2)
    ref_v_smooth = robust_smooth_signal(ref_v, window=5, polyorder=2)
    support_dv_raw = support_raw - ref_v
    support_dv_smooth = robust_smooth_signal(support_dv_raw, window=7, polyorder=2)
    support_smooth = ref_v_smooth + support_dv_smooth
    support_gap_px = np.abs(support_raw - support_smooth)
    support_gap_std = rolling_stat(support_gap_px, radius=3, mode="std")
    support_std = rolling_stat(support_smooth, radius=3, mode="std")
    support_conf = low_value_score(
        support_gap_px, np.percentile(support_gap_px, 10), np.percentile(support_gap_px, 60)
    ) * low_value_score(
        support_gap_std, np.percentile(support_gap_std, 20), np.percentile(support_gap_std, 80)
    ) * low_value_score(
        support_std, np.percentile(support_std, 20), np.percentile(support_std, 80)
    )
    ref_jitter = np.sqrt(
        np.square(rolling_abs_deviation(ref_u, window=7)) + np.square(rolling_abs_deviation(ref_v, window=7))
    )
    finite_jitter = ref_jitter[np.isfinite(ref_jitter)]
    if finite_jitter.size:
        proxy_conf = low_value_score(ref_jitter, np.percentile(finite_jitter, 30), np.percentile(finite_jitter, 85))
    else:
        proxy_conf = np.ones(len(rows), dtype=np.float64)
    support_jitter = rolling_abs_deviation(support_dv_raw, window=7)
    finite_support_jitter = support_jitter[np.isfinite(support_jitter)]
    if finite_support_jitter.size:
        support_conf *= low_value_score(
            support_jitter, np.percentile(finite_support_jitter, 30), np.percentile(finite_support_jitter, 85)
        )

    vx = np.gradient(ref_u)
    vy = np.gradient(ref_v)
    acc = np.sqrt(np.gradient(vx) ** 2 + np.gradient(vy) ** 2)
    motion_score = high_value_score(acc, np.percentile(acc, 50), np.percentile(acc, 90))
    audio_score = build_audio_support(frames, audio_rows, radius=2)

    out_rows: list[dict[str, object]] = []
    for i, row in enumerate(rows):
        offset = float(row["contact_proxy_depth_m"]) - float(depth_smooth[i])
        out_rows.append(
            {
                "frame": row["frame"],
                "time": f"{float(row['time']):.6f}",
                "ref_u": f"{float(row['ref_u']):.3f}",
                "ref_v": f"{float(row['ref_v']):.3f}",
                "ref_u_smooth": f"{ref_u_smooth[i]:.3f}",
                "ref_v_smooth": f"{ref_v_smooth[i]:.3f}",
                "ref_source": row["ref_source"],
                "contact_u": f"{float(row['contact_u']):.3f}",
                "contact_v": f"{float(row['contact_v']):.3f}",
                "contact_source": row["contact_source"],
                "support_u": f"{float(row['support_u']):.3f}",
                "support_v": f"{support_smooth[i]:.3f}",
                "support_v_raw": f"{support_raw[i]:.3f}",
                "support_dv": f"{(support_smooth[i] - float(row['ref_v'])):.3f}",
                "support_dv_smooth": f"{support_dv_smooth[i]:.3f}",
                "support_source": row["support_source"],
                "object_ref_depth_m": f"{depth_smooth[i]:.6f}",
                "contact_proxy_depth_m": f"{float(row['contact_proxy_depth_m']):.6f}",
                "contact_depth_offset_m": f"{offset:.6f}",
                "ref_conf": f"{float(row['ref_conf']):.6f}",
                "support_conf": f"{support_conf[i]:.6f}",
                "contact_conf": f"{float(row['contact_conf']):.6f}",
                "depth_conf": f"{depth_conf[i]:.6f}",
                "observation_conf": f"{min(float(row['ref_conf']), max(depth_conf[i], 0.1)):.6f}",
                "proxy_conf": f"{proxy_conf[i]:.6f}",
                "proxy_jitter_px": f"{ref_jitter[i]:.6f}",
                "support_jitter_px": f"{support_jitter[i]:.6f}",
                "proxy_sigma_px": "5.000000",
                "support_sigma_px": "8.000000",
                "active_label": row["active_label"],
                "active_label_conf": f"{float(row['active_label_conf']):.6f}",
                "active_part_u": f"{float(row['active_part_u']):.3f}",
                "active_part_v": f"{float(row['active_part_v']):.3f}",
                "active_part_z": f"{float(row['active_part_z']):.6f}",
                "contact_proxy_name": row["contact_proxy_name"],
                "object_depth_raw": f"{depth_raw[i]:.6f}",
                "object_depth_smooth": f"{depth_smooth[i]:.6f}",
                "object_depth_confidence": f"{depth_conf[i]:.6f}",
                "support_gap_px": f"{support_gap_px[i]:.6f}",
                "object_motion_score": f"{motion_score[i]:.6f}",
                "audio_score": f"{audio_score[i]:.6f}",
            }
        )
    return write_csv(output_csv, out_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build generic ball contact proxy-depth offsets.")
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--object-observation-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument(
        "--body-model-root",
        type=Path,
        default=Path("third-party/GVHMR/inputs/checkpoints/body_models"),
    )
    args = parser.parse_args()
    out = build_proxy_depth(
        sample_dir=args.sample_dir,
        object_observation_csv=args.object_observation_csv,
        output_csv=args.output_csv,
        body_model_root=args.body_model_root,
    )
    print(f"ball_proxy_depth_csv: {out}")


if __name__ == "__main__":
    main()
