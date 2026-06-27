#!/usr/bin/env python3
"""Build chair two-hand top-rail grasp state from generic observations.

This is a migrated, generic-owned version of the solved chair contact cue:
Qwen labels the semantic contact keyframes, GVHMR provides dense palm motion,
and the actual object target is snapped back onto the observed top rail so
contact cannot pull the chair away from the 2D video evidence.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import smplx
import torch

REPO = Path(__file__).resolve().parents[5]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.shared.human_ball.contact.contact_part_utils import build_contact_part_centers  # noqa: E402


FIELDS = [
    "frame",
    "time",
    "primary_active",
    "primary_hand_side",
    "primary_target_u",
    "primary_target_v",
    "primary_hand_u",
    "primary_hand_v",
    "primary_dist_px",
    "secondary_active",
    "secondary_hand_side",
    "secondary_target_u",
    "secondary_target_v",
    "secondary_hand_u",
    "secondary_hand_v",
    "secondary_dist_px",
    "qwen_contact_visible",
    "qwen_contact_part",
    "qwen_confidence",
    "frame_mode",
    "contact_active",
    "hand_side",
    "contact_object_part",
    "contact_fingers",
    "primary_contact_finger",
    "grasp_type",
    "stable_grasp_local_x",
    "stable_grasp_local_y",
    "stable_grasp_local_z",
    "stable_grasp_conf",
    "stable_grasp_source_frame",
    "contact_target_u",
    "contact_target_v",
    "direct_contact_u",
    "direct_contact_v",
    "secondary_contact_active",
    "secondary_hand_side",
    "secondary_contact_fingers",
    "secondary_primary_contact_finger",
    "secondary_grasp_type",
    "secondary_stable_grasp_local_x",
    "secondary_stable_grasp_local_y",
    "secondary_stable_grasp_local_z",
    "secondary_stable_grasp_conf",
    "secondary_stable_grasp_source_frame",
    "secondary_contact_target_u",
    "secondary_contact_target_v",
    "secondary_direct_contact_u",
    "secondary_direct_contact_v",
    "source",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def ff(row: dict[str, str] | None, key: str, default: float = math.nan) -> float:
    if not row:
        return default
    try:
        value = row.get(key, "")
        if value in {"", None}:
            return default
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return default


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


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


def project_xyz(xyz: np.ndarray, K: np.ndarray) -> np.ndarray:
    z = max(float(xyz[2]), 1e-6)
    return np.array([K[0, 2] + K[0, 0] * xyz[0] / z, K[1, 2] + K[1, 1] * xyz[1] / z], dtype=float)


def closest_on_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float, float]:
    ab = b - a
    denom = float(ab @ ab)
    t = 0.5 if denom <= 1e-6 else float(np.clip(((p - a) @ ab) / denom, 0.0, 1.0))
    q = a + t * ab
    return q, float(np.linalg.norm(p - q)), t


def qwen_by_frame(path: Path) -> dict[int, dict[str, str]]:
    if not path.exists():
        return {}
    out: dict[int, dict[str, str]] = {}
    for row in read_rows(path):
        try:
            out[int(float(row["frame"]))] = row
        except Exception:
            pass
    return out


def gvhmr_toprail_targets(
    sample_dir: Path,
    obs_rows: list[dict[str, str]],
    body_models_root: Path,
    dist_thresh_px: float,
) -> tuple[dict[str, dict[int, dict[str, float]]], dict[str, object]]:
    gvhmr_path = sample_dir / "results/gvhmr/result.pkl"
    if not gvhmr_path.exists():
        return {}, {"available": False, "reason": f"missing {gvhmr_path}"}
    human = read_human_result(gvhmr_path)
    joints = build_body_joints(body_models_root, human)
    centers = build_contact_part_centers(joints)
    K = np.asarray(human["K_fullimg"], dtype=float)
    out: dict[str, dict[int, dict[str, float]]] = {"left_hand": {}, "right_hand": {}}
    for obs in obs_rows:
        fr = int(float(obs["frame"]))
        idx = fr - 1
        if idx < 0 or idx >= len(K):
            continue
        a = np.array([ff(obs, "top_rail_left_u"), ff(obs, "top_rail_left_v")], dtype=float)
        b = np.array([ff(obs, "top_rail_right_u"), ff(obs, "top_rail_right_v")], dtype=float)
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            continue
        for side in ("left_hand", "right_hand"):
            xyz = np.asarray(centers[side][idx], dtype=float)
            if not np.all(np.isfinite(xyz)) or xyz[2] <= 0.2:
                continue
            uv = project_xyz(xyz, K[idx])
            snap, dist, t = closest_on_segment(uv, a, b)
            out[side][fr] = {
                "hand_u": float(uv[0]),
                "hand_v": float(uv[1]),
                "target_u": float(snap[0]),
                "target_v": float(snap[1]),
                "dist_px": float(dist),
                "rail_t": float(t),
                "active": float(dist <= dist_thresh_px),
            }
    summary = {
        "available": True,
        "dist_thresh_px": dist_thresh_px,
        "active_counts": {side: sum(1 for row in out[side].values() if row["active"] > 0.5) for side in out},
    }
    return out, summary


def build_state(
    sample_dir: Path,
    obs_csv: Path,
    qwen_csv: Path,
    body_models_root: Path,
    dist_thresh_px: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    obs_rows = read_rows(obs_csv)
    qwen = qwen_by_frame(qwen_csv)
    targets, gvhmr_summary = gvhmr_toprail_targets(sample_dir, obs_rows, body_models_root, dist_thresh_px)
    rows: list[dict[str, object]] = []
    for obs in obs_rows:
        fr = int(float(obs["frame"]))
        q = qwen.get(fr, {})
        primary = targets.get("right_hand", {}).get(fr, {})
        secondary = targets.get("left_hand", {}).get(fr, {})
        primary_active = bool(primary.get("active", 0.0) > 0.5)
        secondary_active = bool(secondary.get("active", 0.0) > 0.5)
        left_u, left_v = ff(obs, "top_rail_left_u"), ff(obs, "top_rail_left_v")
        right_u, right_v = ff(obs, "top_rail_right_u"), ff(obs, "top_rail_right_v")
        rows.append(
            {
                "frame": fr,
                "time": obs.get("time", f"{(fr - 1) / 24.0:.6f}"),
                "primary_active": "1" if primary_active else "0",
                "primary_hand_side": "right_hand",
                "primary_target_u": f"{primary.get('target_u', math.nan):.3f}" if primary_active else "",
                "primary_target_v": f"{primary.get('target_v', math.nan):.3f}" if primary_active else "",
                "primary_hand_u": f"{primary.get('hand_u', math.nan):.3f}" if primary else "",
                "primary_hand_v": f"{primary.get('hand_v', math.nan):.3f}" if primary else "",
                "primary_dist_px": f"{primary.get('dist_px', math.nan):.3f}" if primary else "",
                "secondary_active": "1" if secondary_active else "0",
                "secondary_hand_side": "left_hand",
                "secondary_target_u": f"{secondary.get('target_u', math.nan):.3f}" if secondary_active else "",
                "secondary_target_v": f"{secondary.get('target_v', math.nan):.3f}" if secondary_active else "",
                "secondary_hand_u": f"{secondary.get('hand_u', math.nan):.3f}" if secondary else "",
                "secondary_hand_v": f"{secondary.get('hand_v', math.nan):.3f}" if secondary else "",
                "secondary_dist_px": f"{secondary.get('dist_px', math.nan):.3f}" if secondary else "",
                "qwen_contact_visible": "1" if truthy(q.get("contact_visible")) else "0",
                "qwen_contact_part": q.get("contact_object_part", ""),
                "qwen_confidence": q.get("confidence", ""),
                "frame_mode": "gvhmr_direct_grasp_anchor" if primary_active or secondary_active else "no_attachment",
                "contact_active": "1" if primary_active else "0",
                "hand_side": "right_hand" if primary_active else "",
                "contact_object_part": "top_rail" if primary_active else "",
                "contact_fingers": q.get("contact_fingers", "[\"palm\"]") if primary_active else "",
                "primary_contact_finger": q.get("primary_contact_finger", "palm") if primary_active else "",
                "grasp_type": "two_hand_top_rail_endpoint_grasp" if primary_active else "",
                "stable_grasp_local_x": "-0.191000" if primary_active else "",
                "stable_grasp_local_y": "0.130000" if primary_active else "",
                "stable_grasp_local_z": "0.751000" if primary_active else "",
                "stable_grasp_conf": "0.850000" if primary_active else "",
                "stable_grasp_source_frame": str(fr) if primary_active else "",
                "contact_target_u": f"{left_u:.3f}" if primary_active and math.isfinite(left_u) else "",
                "contact_target_v": f"{left_v:.3f}" if primary_active and math.isfinite(left_v) else "",
                "direct_contact_u": f"{primary.get('target_u', math.nan):.3f}" if primary_active else "",
                "direct_contact_v": f"{primary.get('target_v', math.nan):.3f}" if primary_active else "",
                "secondary_contact_active": "1" if secondary_active else "0",
                "secondary_hand_side": "left_hand" if secondary_active else "",
                "secondary_contact_fingers": q.get("contact_fingers", "[\"palm\"]") if secondary_active else "",
                "secondary_primary_contact_finger": q.get("primary_contact_finger", "palm") if secondary_active else "",
                "secondary_grasp_type": "two_hand_top_rail_endpoint_grasp" if secondary_active else "",
                "secondary_stable_grasp_local_x": "0.191000" if secondary_active else "",
                "secondary_stable_grasp_local_y": "0.130000" if secondary_active else "",
                "secondary_stable_grasp_local_z": "0.751000" if secondary_active else "",
                "secondary_stable_grasp_conf": "0.850000" if secondary_active else "",
                "secondary_stable_grasp_source_frame": str(fr) if secondary_active else "",
                "secondary_contact_target_u": f"{right_u:.3f}" if secondary_active and math.isfinite(right_u) else "",
                "secondary_contact_target_v": f"{right_v:.3f}" if secondary_active and math.isfinite(right_v) else "",
                "secondary_direct_contact_u": f"{secondary.get('target_u', math.nan):.3f}" if secondary_active else "",
                "secondary_direct_contact_v": f"{secondary.get('target_v', math.nan):.3f}" if secondary_active else "",
                "source": "gvhmr_palm_projected_to_observed_toprail",
            }
        )
    summary = {
        "n_frames": len(rows),
        "qwen_keyframes": sorted(qwen),
        "gvhmr": gvhmr_summary,
        "primary_active_frames": [int(r["frame"]) for r in rows if r["primary_active"] == "1"],
        "secondary_active_frames": [int(r["frame"]) for r in rows if r["secondary_active"] == "1"],
        "note": "This state gates candidate contact frames only. Candidate object points are generated from observed top-rail endpoints by the stage2 policy.",
    }
    return rows, summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample-dir", type=Path, required=True)
    ap.add_argument("--obs-csv", type=Path, required=True)
    ap.add_argument("--qwen-csv", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--out-summary", type=Path, required=True)
    ap.add_argument("--body-models-root", type=Path, default=Path("third-party/GVHMR/inputs/checkpoints/body_models"))
    ap.add_argument("--dist-thresh-px", type=float, default=55.0)
    args = ap.parse_args()

    rows, summary = build_state(args.sample_dir, args.obs_csv, args.qwen_csv, args.body_models_root, args.dist_thresh_px)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"csv": str(args.out_csv), "summary": str(args.out_summary), **summary}, indent=2))


if __name__ == "__main__":
    main()
