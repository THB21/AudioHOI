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

THIS_DIR = Path(__file__).resolve().parent
CONTACT_DIR = THIS_DIR.parent / "human_ball" / "contact"
if str(CONTACT_DIR) not in sys.path:
    sys.path.insert(0, str(CONTACT_DIR))

from contact_part_utils import (
    build_contact_part_centers,
    infer_default_part,
    normalize_contact_label,
)


def parse_float(row: dict[str, str], key: str, default: float = math.nan) -> float:
    value = row.get(key, "")
    if value is None or value == "":
        return default
    return float(value)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows found in {path}")
    return rows


def read_da3_priors(path: Path) -> dict[int, tuple[float, float]]:
    out: dict[int, tuple[float, float]] = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(row["frame"])
            raw = parse_float(row, "da3_depth_raw")
            smooth = parse_float(row, "da3_depth_smooth")
            out[frame] = (raw, smooth)
    return out


def read_object_mesh_tracks(path: Path) -> dict[int, dict[str, np.ndarray]]:
    by_frame: dict[int, list[dict[str, object]]] = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(row["frame"])
            by_frame.setdefault(frame, []).append(row)
    out: dict[int, dict[str, np.ndarray]] = {}
    for frame, rows in by_frame.items():
        xy = []
        visible = []
        is_boundary = []
        point_ids = []
        for r in rows:
            x = r.get("x", "")
            y = r.get("y", "")
            if x == "" or y == "":
                continue
            xy.append([float(x), float(y)])
            visible.append(float(r.get("visible", 1.0) or 1.0) > 0.5)
            is_boundary.append(str(r.get("point_type", "")) == "boundary")
            point_ids.append(str(r.get("point_id", "")))
        if xy:
            out[frame] = {
                "xy": np.asarray(xy, dtype=np.float64),
                "visible": np.asarray(visible, dtype=bool),
                "is_boundary": np.asarray(is_boundary, dtype=bool),
                "point_ids": np.asarray(point_ids, dtype=object),
            }
    return out

def read_index(path: Path) -> dict[int, Path]:
    depth_dir = path.parent
    frame_to_path: dict[int, Path] = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_to_path[int(row["frame"])] = depth_dir / row["file"]
    if not frame_to_path:
        raise RuntimeError(f"No DA3 entries in {path}")
    return frame_to_path


def load_depth(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
    elif path.suffix.lower() == ".npz":
        data = np.load(path)
        key = next(iter(data.files))
        arr = data[key]
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
        raise RuntimeError(f"Could not open video for size query: {video_path}")
    width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video size from {video_path}: {(width, height)}")
    return width, height


def map_to_depth_uv(u: float, v: float, src_hw: tuple[int, int], depth_hw: tuple[int, int]) -> tuple[float, float]:
    src_w, src_h = src_hw
    depth_h, depth_w = depth_hw
    su = depth_w / float(src_w)
    sv = depth_h / float(src_h)
    return u * su, v * sv


def robust_patch_median(depth: np.ndarray, u: float, v: float, patch_r: int = 3) -> float:
    h, w = depth.shape[:2]
    cx = int(round(u))
    cy = int(round(v))
    x0 = max(0, cx - patch_r)
    x1 = min(w, cx + patch_r + 1)
    y0 = max(0, cy - patch_r)
    y1 = min(h, cy + patch_r + 1)
    patch = depth[y0:y1, x0:x1]
    vals = patch[np.isfinite(patch)]
    vals = vals[vals > 0]
    if vals.size == 0:
        return float("nan")
    return float(np.median(vals))


def bilinear_sample_depth(depth: np.ndarray, u: float, v: float) -> float:
    h, w = depth.shape[:2]
    if not (np.isfinite(u) and np.isfinite(v)):
        return float("nan")
    if u < 0 or v < 0 or u > (w - 1) or v > (h - 1):
        return float("nan")
    x0 = int(math.floor(u))
    y0 = int(math.floor(v))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    dx = float(u - x0)
    dy = float(v - y0)
    q00 = float(depth[y0, x0])
    q10 = float(depth[y0, x1])
    q01 = float(depth[y1, x0])
    q11 = float(depth[y1, x1])
    vals = np.asarray([q00, q10, q01, q11], dtype=np.float32)
    if not np.all(np.isfinite(vals)) or np.any(vals <= 0):
        return float("nan")
    top = (1.0 - dx) * q00 + dx * q10
    bottom = (1.0 - dx) * q01 + dx * q11
    return float((1.0 - dy) * top + dy * bottom)


def sample_depth_at_uv(depth: np.ndarray, u: float, v: float) -> tuple[float, float]:
    exact = bilinear_sample_depth(depth, u, v)
    patch = robust_patch_median(depth, u, v, patch_r=1)
    if np.isfinite(exact):
        if np.isfinite(patch):
            conf = 1.0 / (1.0 + abs(exact - patch))
        else:
            conf = 0.75
        return exact, float(conf)
    if np.isfinite(patch):
        return patch, 0.5
    return float("nan"), 0.0


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


def project_points(points_cam: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = np.clip(points_cam[:, 2], 1e-6, None)
    u = K[:, 0, 0] * (points_cam[:, 0] / z) + K[:, 0, 2]
    v = K[:, 1, 1] * (points_cam[:, 1] / z) + K[:, 1, 2]
    valid = points_cam[:, 2] > 1e-6
    return np.stack([u, v], axis=1), valid


def choose_contact_proxy(
    obs: dict[str, str],
    active_uv: np.ndarray | None,
) -> tuple[float, float, float, str]:
    candidates: list[tuple[str, float, float, float]] = []

    def add(name: str, x_key: str, y_key: str, visible_key: str | None = None) -> None:
        x = parse_float(obs, x_key)
        y = parse_float(obs, y_key)
        visible = 1.0 if visible_key is None else parse_float(obs, visible_key, 0.0)
        if np.isfinite(x) and np.isfinite(y) and visible > 0.0:
            candidates.append((name, x, y, visible))

    add("center", "center_x", "center_y", "center_visible")
    add("left", "left_x", "left_y", "left_visible")
    add("right", "right_x", "right_y", "right_visible")
    add("top", "top_x", "top_y", "top_visible")
    add("bottom", "bottom_x", "bottom_y", "bottom_visible")
    low_x = parse_float(obs, "lowest_visible_x")
    low_y = parse_float(obs, "lowest_visible_y")
    if np.isfinite(low_x) and np.isfinite(low_y):
        candidates.append(("lowest_visible", low_x, low_y, 1.0))

    if not candidates:
        cx = parse_float(obs, "center_x", 0.0)
        cy = parse_float(obs, "center_y", 0.0)
        return cx, cy, 0.0, "fallback_center"

    if active_uv is None or not np.all(np.isfinite(active_uv)):
        name, u, v, conf = candidates[0]
        return u, v, conf, name

    best = min(
        candidates,
        key=lambda item: float(np.hypot(item[1] - active_uv[0], item[2] - active_uv[1])),
    )
    name, u, v, conf = best
    return u, v, conf, name


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Experimental builder for radius-free object proxy observations.")
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--out-subdir", type=str, default="object_proxy_observations_test")
    parser.add_argument("--object-observation-csv", type=Path, default=None)
    parser.add_argument("--da3-prior-csv", type=Path, default=None)
    parser.add_argument("--contact-state-csv", type=Path, default=None)
    parser.add_argument("--contact-event-csv", type=Path, default=None)
    parser.add_argument("--object-mesh-csv", type=Path, default=None)
    parser.add_argument("--body-model-root", type=Path, default=Path("third-party/GVHMR/inputs/checkpoints/body_models"))
    args = parser.parse_args()

    sample_dir = args.sample_dir
    results_dir = sample_dir / "results"
    object_csv = args.object_observation_csv or (results_dir / "object_observations" / "object_observations.csv")
    da3_csv = args.da3_prior_csv or (results_dir / "da3" / "priors" / "ball_depth_prior.csv")
    mesh_csv = args.object_mesh_csv or (results_dir / "tracking" / "object_mesh_tracks_test.csv")
    depth_index_csv = results_dir / "da3" / "scene_depth" / "index.csv"
    state_csv = args.contact_state_csv or (results_dir / "contact_candidates" / "contact_state_frames.csv")
    event_csv = args.contact_event_csv or (results_dir / "contact_candidates" / "contact_candidates_labeled.csv")
    out_dir = results_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    object_rows = read_rows(object_csv)
    state_rows = read_rows(state_csv)
    event_rows = read_rows(event_csv)
    da3 = read_da3_priors(da3_csv)
    mesh_tracks = read_object_mesh_tracks(mesh_csv) if mesh_csv.exists() else {}
    frame_to_depth = read_index(depth_index_csv) if depth_index_csv.exists() else {}
    video_hw = get_video_hw(sample_dir / "video.mp4") if frame_to_depth else None
    state_by_frame = {int(r["frame"]): r for r in state_rows}

    human = read_human_result(results_dir / "gvhmr" / "result.pkl")
    joints = build_body_joints(args.body_model_root, human)
    K = np.asarray(human["K_fullimg"], dtype=np.float64)
    if len(joints) < len(object_rows):
        raise RuntimeError(f"GVHMR has fewer frames than object observations: {len(joints)} < {len(object_rows)}")
    joints = joints[: len(object_rows)]
    K = K[: len(object_rows)]

    default_part = infer_default_part([*state_rows, *event_rows], fallback="hand")
    part_centers = build_contact_part_centers(joints)

    out_rows: list[dict[str, object]] = []
    for idx, row in enumerate(object_rows):
        frame = int(row["frame"])
        if frame not in state_by_frame:
            continue
        state_row = state_by_frame[frame]
        label = normalize_contact_label(state_row, default_part=default_part, fallback_side="right")
        active_uv = None
        active_z = math.nan
        if label in part_centers:
            pts = part_centers[label][idx : idx + 1]
            uv, valid = project_points(pts, K[idx : idx + 1])
            if bool(valid[0]):
                active_uv = uv[0]
                active_z = float(pts[0, 2])

        ref_u = parse_float(row, "center_x")
        ref_v = parse_float(row, "center_y")
        support_u = parse_float(row, "lowest_visible_x", default=parse_float(row, "bottom_x", default=ref_u))
        support_v = parse_float(row, "lowest_visible_y", default=parse_float(row, "bottom_y", default=ref_v))
        mesh = mesh_tracks.get(frame)
        if mesh is not None:
            valid = mesh["visible"].copy()
            boundary_valid = valid & mesh["is_boundary"]
            if np.any(boundary_valid):
                valid = boundary_valid
            if np.any(valid):
                pts = mesh["xy"][valid]
                lowest_idx = int(np.argmax(pts[:, 1]))
                support_u = float(pts[lowest_idx, 0])
                support_v = float(pts[lowest_idx, 1])
        support_dv = support_v - ref_v if np.isfinite(support_v) and np.isfinite(ref_v) else 0.0

        contact_u, contact_v, contact_proxy_conf, contact_proxy_name = math.nan, math.nan, 0.0, "no_mesh"
        if active_uv is not None and np.all(np.isfinite(active_uv)) and mesh is not None:
            valid = mesh["visible"].copy()
            boundary_valid = valid & mesh["is_boundary"]
            if np.any(boundary_valid):
                valid = boundary_valid
            if np.any(valid):
                candidates = mesh["xy"][valid]
                candidate_ids = mesh["point_ids"][valid]
                dists = np.linalg.norm(candidates - active_uv[None, :], axis=1)
                j = int(np.argmin(dists))
                contact_u = float(candidates[j, 0])
                contact_v = float(candidates[j, 1])
                contact_proxy_conf = 1.0
                contact_proxy_name = str(candidate_ids[j])
        if not np.isfinite(contact_u) or not np.isfinite(contact_v):
            contact_u, contact_v, contact_proxy_conf, contact_proxy_name = choose_contact_proxy(row, active_uv)
        raw_depth, smooth_depth = da3.get(frame, (math.nan, math.nan))
        object_ref_depth_m = math.nan
        contact_proxy_depth_m = math.nan
        ref_depth_conf = 0.0
        contact_depth_conf = 0.0
        if frame in frame_to_depth and video_hw is not None:
            depth_map = load_depth(frame_to_depth[frame])
            ref_ud, ref_vd = map_to_depth_uv(ref_u, ref_v, video_hw, depth_map.shape[:2])
            object_ref_depth_m, ref_depth_conf = sample_depth_at_uv(depth_map, ref_ud, ref_vd)
            if np.isfinite(contact_u) and np.isfinite(contact_v):
                contact_ud, contact_vd = map_to_depth_uv(contact_u, contact_v, video_hw, depth_map.shape[:2])
                contact_proxy_depth_m, contact_depth_conf = sample_depth_at_uv(depth_map, contact_ud, contact_vd)
        depth_conf = min(ref_depth_conf, contact_depth_conf)
        if not np.isfinite(object_ref_depth_m):
            continue
        if not np.isfinite(contact_proxy_depth_m):
            continue
        contact_depth_offset_m = float(contact_proxy_depth_m - object_ref_depth_m)

        observation_conf = parse_float(row, "observation_conf", 1.0)
        support_conf = 1.0 if np.isfinite(support_v) else 0.0
        ref_conf = parse_float(row, "mask_conf", 1.0)

        out_rows.append(
            {
                "frame": frame,
                "time": f"{float(row['time']):.6f}",
                "ref_u": f"{ref_u:.3f}",
                "ref_v": f"{ref_v:.3f}",
                "support_u": f"{support_u:.3f}" if np.isfinite(support_u) else "",
                "support_v": f"{support_v:.3f}" if np.isfinite(support_v) else "",
                "support_dv": f"{support_dv:.3f}",
                "contact_u": f"{contact_u:.3f}" if np.isfinite(contact_u) else "",
                "contact_v": f"{contact_v:.3f}" if np.isfinite(contact_v) else "",
                "object_ref_depth_m": f"{object_ref_depth_m:.6f}",
                "contact_proxy_depth_m": f"{contact_proxy_depth_m:.6f}",
                "contact_depth_offset_m": f"{contact_depth_offset_m:.6f}",
                "ref_conf": f"{ref_conf:.6f}",
                "support_conf": f"{support_conf:.6f}",
                "contact_conf": f"{contact_proxy_conf:.6f}",
                "depth_conf": f"{depth_conf:.6f}",
                "observation_conf": f"{observation_conf:.6f}",
                "active_label": label,
                "active_part_u": f"{active_uv[0]:.3f}" if active_uv is not None else "",
                "active_part_v": f"{active_uv[1]:.3f}" if active_uv is not None else "",
                "active_part_z": f"{active_z:.6f}" if np.isfinite(active_z) else "",
                "contact_proxy_name": contact_proxy_name,
                "da3_depth_raw": f"{raw_depth:.6f}" if np.isfinite(raw_depth) else "",
                "da3_depth_smooth": f"{smooth_depth:.6f}" if np.isfinite(smooth_depth) else "",
            }
        )

    out_csv = out_dir / "object_proxy_observations.csv"
    write_csv(
        out_csv,
        out_rows,
        [
            "frame",
            "time",
            "ref_u",
            "ref_v",
            "support_u",
            "support_v",
            "support_dv",
            "contact_u",
            "contact_v",
            "object_ref_depth_m",
            "contact_proxy_depth_m",
            "contact_depth_offset_m",
            "ref_conf",
            "support_conf",
            "contact_conf",
            "depth_conf",
            "observation_conf",
            "active_label",
            "active_part_u",
            "active_part_v",
            "active_part_z",
            "contact_proxy_name",
            "da3_depth_raw",
            "da3_depth_smooth",
        ],
    )
    print(f"object_proxy_observations_csv: {out_csv}")


if __name__ == "__main__":
    main()
