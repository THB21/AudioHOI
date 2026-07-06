#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from pathlib import Path

import numpy as np
import smplx
import torch


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


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
        writer.writerows(rows)
    return path


def read_human(path: Path) -> dict[str, np.ndarray]:
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


def build_joints(root: Path, human: dict[str, np.ndarray]) -> np.ndarray:
    model = smplx.create(
        str(root),
        model_type="smplx",
        gender="neutral",
        ext="npz",
        use_pca=False,
        flat_hand_mean=True,
        num_betas=10,
        batch_size=human["transl"].shape[0],
    )
    with torch.inference_mode():
        out = model(
            body_pose=torch.from_numpy(human["body_pose"]),
            betas=torch.from_numpy(human["betas"]),
            global_orient=torch.from_numpy(human["global_orient"]),
            transl=torch.from_numpy(human["transl"]),
            return_verts=False,
        )
    return out.joints.detach().cpu().numpy().astype(np.float64)


def project(points: np.ndarray, k: np.ndarray) -> np.ndarray:
    z = np.clip(points[:, 2], 1e-6, None)
    return np.column_stack([k[:, 0, 0] * points[:, 0] / z + k[:, 0, 2], k[:, 1, 1] * points[:, 1] / z + k[:, 1, 2]])


def palm_centers(joints: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "left": joints[:, [20, 25, 28, 31, 34, 37], :].mean(axis=1),
        "right": joints[:, [21, 40, 43, 46, 49, 52], :].mean(axis=1),
    }


HAND_JOINT_IDS = {
    "left": [20, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
    "right": [21, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
}
FINGERTIP_IDS = {
    "left": [27, 30, 33, 36, 39],
    "right": [42, 45, 48, 51, 54],
}


def line_tracks(path: Path) -> dict[int, dict[str, tuple[float, float]]]:
    out: dict[int, dict[str, tuple[float, float]]] = {}
    for row in read_csv(path):
        pid = row.get("point_id", "")
        if pid not in {"tip_a", "tip_b"}:
            continue
        fr = int(float(row["frame"]))
        out.setdefault(fr, {})[pid] = (float(row["x"]), float(row["y"]))
    return out


def line_tracks_from_correspondence(path: Path) -> dict[int, dict[str, tuple[float, float]]]:
    out: dict[int, dict[str, tuple[float, float]]] = {}
    for row in read_csv(path):
        try:
            fr = int(float(row["frame"]))
            out[fr] = {
                "tip_a": (float(row["visible_x1"]), float(row["visible_y1"])),
                "tip_b": (float(row["visible_x2"]), float(row["visible_y2"])),
            }
        except Exception:
            continue
    return out


def nearest_on_segment(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float, float]:
    axis = b - a
    denom = float(np.dot(axis, axis))
    if denom <= 1e-6:
        return a.copy(), 0.0, float(np.linalg.norm(point - a))
    t = float(np.clip(np.dot(point - a, axis) / denom, 0.0, 1.0))
    nearest = a + t * axis
    return nearest, t, float(np.linalg.norm(point - nearest))


def point_distances_to_segment(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    axis = b - a
    denom = float(np.dot(axis, axis))
    if denom <= 1e-6:
        return np.linalg.norm(points - a[None, :], axis=1)
    t = np.clip(((points - a[None, :]) @ axis) / denom, 0.0, 1.0)
    nearest = a[None, :] + t[:, None] * axis[None, :]
    return np.linalg.norm(points - nearest, axis=1)


def contact_is_active(dist_px: float, local_s: float) -> bool:
    if not math.isfinite(dist_px):
        return False
    if dist_px <= 75.0:
        return True
    # Keep endpoint grasps a little more tolerant because visible line tracks
    # can truncate the currently occluded side of a long object.
    return dist_px <= 105.0 and (local_s <= 0.08 or local_s >= 0.92)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build persistent two-palm line-object contact candidates.")
    ap.add_argument("--sample-dir", type=Path, required=True)
    ap.add_argument("--object-observations", type=Path, required=True)
    ap.add_argument("--contact-candidates", type=Path, required=True)
    ap.add_argument("--contact-state", type=Path, required=True)
    ap.add_argument("--metrics", type=Path, required=True)
    ap.add_argument("--line-correspondence", type=Path, default=None)
    ap.add_argument("--body-model-root", type=Path, default=Path("third-party/GVHMR/inputs/checkpoints/body_models"))
    args = ap.parse_args()

    if args.line_correspondence and args.line_correspondence.exists():
        tracks = line_tracks_from_correspondence(args.line_correspondence)
        line_source = str(args.line_correspondence)
    else:
        fallback_line_path = args.sample_dir / "results/tracking/object_mesh_tracks_test.csv"
        tracks = line_tracks(fallback_line_path)
        line_source = str(fallback_line_path)
    obs_by_frame = {int(float(r["frame"])): r for r in read_csv(args.object_observations)}
    human = read_human(args.sample_dir / "results/gvhmr/result.pkl")
    joints = build_joints(args.body_model_root, human)
    palm_xyz_by_side = palm_centers(joints)
    palm_uv = {side: project(points, human["K_fullimg"]) for side, points in palm_xyz_by_side.items()}
    hand_uv = {
        side: np.stack([project(joints[:, ids, :][fr], human["K_fullimg"][fr : fr + 1]) for fr in range(joints.shape[0])])
        for side, ids in HAND_JOINT_IDS.items()
    }
    fingertip_uv = {
        side: np.stack([project(joints[:, ids, :][fr], human["K_fullimg"][fr : fr + 1]) for fr in range(joints.shape[0])])
        for side, ids in FINGERTIP_IDS.items()
    }

    rows: list[dict[str, object]] = []
    for fr in sorted(tracks):
        line = tracks[fr]
        if "tip_a" not in line or "tip_b" not in line:
            continue
        a = np.asarray(line["tip_a"], dtype=np.float64)
        b = np.asarray(line["tip_b"], dtype=np.float64)
        time = obs_by_frame.get(fr, {}).get("time", f"{(fr - 1) / 24.0:.6f}")
        for side in ("left", "right"):
            uv = palm_uv[side][min(fr - 1, len(palm_uv[side]) - 1)]
            palm_xyz = palm_xyz_by_side[side][min(fr - 1, len(palm_uv[side]) - 1)]
            hand_points = hand_uv[side][min(fr - 1, len(hand_uv[side]) - 1)]
            fingertip_points = fingertip_uv[side][min(fr - 1, len(fingertip_uv[side]) - 1)]
            hand_dists = point_distances_to_segment(hand_points, a, b)
            fingertip_dists = point_distances_to_segment(fingertip_points, a, b)
            nearest, local_s, dist = nearest_on_segment(uv, a, b)
            is_active = contact_is_active(dist, local_s)
            conf = 1.0 / (1.0 + dist / 80.0)
            rows.append(
                {
                    "frame": fr,
                    "time": time,
                    "contact_active": 1 if is_active else 0,
                    "human_part": "palm",
                    "human_side": side,
                    "object_part": "main_body",
                    "object_local_id": f"{side}_palm_line_anchor",
                    "object_local_s": f"{local_s:.6f}",
                    "contact_u": f"{nearest[0]:.3f}",
                    "contact_v": f"{nearest[1]:.3f}",
                    "palm_u": f"{uv[0]:.3f}",
                    "palm_v": f"{uv[1]:.3f}",
                    "palm_x": f"{palm_xyz[0]:.6f}",
                    "palm_y": f"{palm_xyz[1]:.6f}",
                    "palm_z": f"{palm_xyz[2]:.6f}",
                    "anchor_z_m": f"{palm_xyz[2]:.6f}",
                    "palm_to_line_px": f"{dist:.3f}",
                    "hand_joint_min_line_px": f"{float(np.min(hand_dists)):.3f}",
                    "hand_joint_median_line_px": f"{float(np.median(hand_dists)):.3f}",
                    "hand_joint_near_count_25px": int(np.count_nonzero(hand_dists <= 25.0)),
                    "fingertip_min_line_px": f"{float(np.min(fingertip_dists)):.3f}",
                    "fingertip_median_line_px": f"{float(np.median(fingertip_dists)):.3f}",
                    "contact_depth_offset_m": "0.000000",
                    "contact_conf": f"{conf:.6f}",
                    "anchor_score": f"{conf:.6f}",
                    "source": "persistent_two_palm_line_from_gvhmr_palm_projection",
                    "line_source": line_source,
                }
            )

    stable = {}
    for side in ("left", "right"):
        side_rows = [r for r in rows if r["human_side"] == side]
        dists = [float(r["palm_to_line_px"]) for r in side_rows]
        if dists:
            # Mug-style stable grasp anchors should be updated only from
            # trustworthy visible-contact frames. Frames with bad line tracks
            # keep the previous/global stable local anchor instead of moving it.
            gate = min(45.0, float(np.percentile(dists, 45)))
        else:
            gate = 45.0
        vals = [float(r["object_local_s"]) for r in side_rows if float(r["palm_to_line_px"]) <= gate]
        if len(vals) < max(6, len(side_rows) // 12):
            vals = [float(r["object_local_s"]) for r in side_rows]
        stable[side] = float(np.median(vals)) if vals else math.nan
    state_rows = []
    for row in rows:
        side = str(row["human_side"])
        drift = abs(float(row["object_local_s"]) - stable[side]) if math.isfinite(stable[side]) else math.nan
        row["stable_object_local_s"] = f"{stable[side]:.6f}" if math.isfinite(stable[side]) else ""
        row["local_s_drift"] = f"{drift:.6f}" if math.isfinite(drift) else ""
        state_rows.append(
            {
                "frame": row["frame"],
                "time": row["time"],
                "contact_active": row["contact_active"],
                "human_contact_state": row["contact_active"],
                "contact_part": "palm",
                "contact_label": f"{side}_palm_on_line_object",
                "contact_side": side,
                "active_object_point_id": row["object_local_id"],
                "active_object_u": row["contact_u"],
                "active_object_v": row["contact_v"],
                "state_active_part_u": row["palm_u"],
                "state_active_part_v": row["palm_v"],
                "state_active_part_x": row["palm_x"],
                "state_active_part_y": row["palm_y"],
                "state_active_part_z": row["palm_z"],
                "anchor_z_m": row["anchor_z_m"],
                "active_part_distance_px": row["palm_to_line_px"],
                "contact_conf": row["contact_conf"],
                "object_local_s": row["object_local_s"],
                "stable_object_local_s": row["stable_object_local_s"],
                "local_s_drift": row["local_s_drift"],
            }
        )

    write_csv(args.contact_candidates, rows)
    write_csv(args.contact_state, state_rows)
    dists = [float(r["palm_to_line_px"]) for r in rows]
    drifts = [float(r["local_s_drift"]) for r in rows if r["local_s_drift"] != ""]
    metrics = {
        "component": "persistent_two_palm_line",
        "generic_pattern": "stable_line_grasp_anchor",
        "reused_pipeline_patterns": [
            "mug stable-grasp-anchor: keep persistent object-local hand anchors across frames",
            "chair line/segment observation: represent elongated rigid parts by semantic line coordinates",
        ],
        "rows": len(rows),
        "active_rows": sum(1 for r in rows if int(r["contact_active"]) == 1),
        "inactive_rows": sum(1 for r in rows if int(r["contact_active"]) == 0),
        "contact_candidates": str(args.contact_candidates),
        "contact_state": str(args.contact_state),
        "line_source": line_source,
        "stable_object_local_s": stable,
        "median_palm_to_line_px": float(np.median(dists)) if dists else None,
        "p95_palm_to_line_px": float(np.percentile(dists, 95)) if dists else None,
        "median_local_s_drift": float(np.median(drifts)) if drifts else None,
        "stable_anchor_update_rule": "mug-style: estimate persistent local palm anchors from low palm-to-line distance frames only; outlier visible line frames keep stable anchor and do not update it",
        "policy": "persistent two-palm contact candidates for elongated rigid objects; records fixed local line coordinate per palm for later contact residuals",
    }
    args.metrics.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
