#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pickle
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    REPO = Path(__file__).resolve().parents[4]
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
else:
    REPO = Path(__file__).resolve().parents[4]

from scripts.shared.generic_contact_pipeline.core.config import available_cases, load_case_profile  # noqa: E402
from scripts.shared.generic_contact_pipeline.core.io import read_csv, repo_path, write_csv, write_json  # noqa: E402


OUT_DIR = REPO / "docs" / "related_work_audit"
BODY_MODELS_ROOT = REPO / "third-party" / "GVHMR" / "inputs" / "checkpoints" / "body_models"
LEFT_PALM_IDS = [20, 25, 28, 31, 34]
RIGHT_PALM_IDS = [21, 40, 43, 46, 49]
LEFT_FOOT_ID = 10
RIGHT_FOOT_ID = 11


def ff(value: object) -> float | None:
    try:
        if value in ("", None):
            return None
        out = float(value)  # type: ignore[arg-type]
        return out if math.isfinite(out) else None
    except Exception:
        return None


def first(row: dict[str, str], keys: list[str]) -> float | None:
    for key in keys:
        val = ff(row.get(key))
        if val is not None:
            return val
    return None


def median(vals: list[float]) -> float | None:
    vals = [v for v in vals if math.isfinite(v)]
    return statistics.median(vals) if vals else None


def mean(vals: list[float]) -> float | None:
    vals = [v for v in vals if math.isfinite(v)]
    return statistics.fmean(vals) if vals else None


def p95(vals: list[float]) -> float | None:
    vals = sorted(v for v in vals if math.isfinite(v))
    if not vals:
        return None
    return vals[min(len(vals) - 1, int(round(0.95 * (len(vals) - 1))))]


def ratio(a: Any, b: Any) -> float | None:
    aa = ff(a)
    bb = ff(b)
    if aa is None or bb is None or abs(bb) < 1e-12:
        return None
    return aa / bb


def is_active(row: dict[str, str], case: str) -> bool:
    for key in ["contact_active", "anchor_contact_state", "is_candidate", "human_contact_state"]:
        val = str(row.get(key, "")).strip().lower()
        if val in {"1", "true", "yes"}:
            return True
    if case == "mug":
        return any(
            str(row.get(k, "")).strip() in {"1", "true", "True"}
            for k in ["use_this_point_for_hand_attachment", "use_previous_grasp_for_hand_attachment"]
        )
    return False


def choose_pose_path(case: str, variant: str) -> Path | None:
    profile = load_case_profile(case)
    if variant == "generic":
        path = profile.result_dir / "object_pose.csv"
        return path if path.exists() else None
    rel = profile.baseline.get("final_pose_csv")
    if not rel:
        return None
    path = repo_path(rel)
    return path if path.exists() else None


def choose_contact_path(case: str, variant: str) -> Path | None:
    profile = load_case_profile(case)
    if variant == "generic":
        for path in [profile.result_dir / "object_contact_points.csv", profile.result_dir / "contact_candidates.csv"]:
            if path.exists():
                return path
    for key in ["final_contact_csv", "grasp_anchor_csv", "contact_candidates_csv", "contact_state_csv"]:
        rel = profile.baseline.get(key)
        if rel:
            path = repo_path(rel)
            if path.exists():
                return path
    return None


def choose_local_points_path(case: str, variant: str) -> Path | None:
    profile = load_case_profile(case)
    if variant == "generic":
        path = profile.result_dir / "object_local_points.csv"
        if path.exists():
            return path
    for key in ["object_local_points_csv", "semantic_local_points_csv"]:
        rel = profile.baseline.get(key)
        if rel:
            path = repo_path(rel)
            if path.exists():
                return path
    # Some baselines did not preserve local points under final_result; the
    # generic snapshot uses the same object model, so it is a valid source for
    # object-space surface consistency checks.
    fallback = profile.result_dir / "object_local_points.csv"
    return fallback if fallback.exists() else None


def vec_from_row(row: dict[str, str], triples: list[tuple[str, str, str]]) -> np.ndarray | None:
    for xs, ys, zs in triples:
        x, y, z = ff(row.get(xs)), ff(row.get(ys)), ff(row.get(zs))
        if x is not None and y is not None and z is not None:
            return np.array([x, y, z], dtype=np.float64)
    return None


def load_local_points(path: Path | None) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if path is None or not path.exists():
        return np.zeros((0, 3), dtype=np.float64), {}
    rows = read_csv(path)
    pts: list[np.ndarray] = []
    by_id: dict[str, np.ndarray] = {}
    for row in rows:
        p = vec_from_row(
            row,
            [
                ("local_x", "local_y", "local_z"),
                ("object_local_x", "object_local_y", "object_local_z"),
                ("mug_local_x", "mug_local_y", "mug_local_z"),
                ("chair_local_x", "chair_local_y", "chair_local_z"),
                ("nearest_vertex_local_x", "nearest_vertex_local_y", "nearest_vertex_local_z"),
            ],
        )
        if p is None or not np.isfinite(p).all():
            continue
        pts.append(p)
        pid = row.get("point_id") or row.get("object_local_id") or row.get("chair_local_point_id") or ""
        if pid:
            by_id[str(pid)] = p
    return np.asarray(pts, dtype=np.float64), by_id


def contact_local_point(row: dict[str, str], by_id: dict[str, np.ndarray]) -> np.ndarray | None:
    direct = vec_from_row(
        row,
        [
            ("stable_grasp_local_x", "stable_grasp_local_y", "stable_grasp_local_z"),
            ("current_candidate_local_x", "current_candidate_local_y", "current_candidate_local_z"),
            ("chair_local_x", "chair_local_y", "chair_local_z"),
            ("object_local_x", "object_local_y", "object_local_z"),
            ("mug_local_x", "mug_local_y", "mug_local_z"),
        ],
    )
    if direct is not None:
        return direct
    pid = row.get("object_local_id") or row.get("active_object_point_id") or row.get("chair_local_point_id") or ""
    return by_id.get(str(pid)) if pid else None


def pose_center(row: dict[str, str]) -> np.ndarray | None:
    return vec_from_row(row, [("tx", "ty", "tz"), ("x", "y", "z")])


def euler_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return rz @ ry @ rx


def quat_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = qx / n, qy / n, qz / n, qw / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotation_from_pose(row: dict[str, str]) -> np.ndarray:
    q = [ff(row.get(k)) for k in ["qx", "qy", "qz", "qw"]]
    if all(v is not None for v in q):
        return quat_matrix(q[0], q[1], q[2], q[3])  # type: ignore[arg-type]
    yaw, pitch, roll = ff(row.get("yaw")) or 0.0, ff(row.get("pitch")) or 0.0, ff(row.get("roll")) or 0.0
    return euler_matrix(yaw, pitch, roll)


def local_to_cam(local: np.ndarray, pose: dict[str, str]) -> np.ndarray | None:
    center = pose_center(pose)
    if center is None:
        return None
    scale = ff(pose.get("scale")) or 1.0
    return center + rotation_from_pose(pose) @ (local * scale)


def load_human_joints(sample_dir: Path) -> np.ndarray | None:
    try:
        import smplx  # type: ignore
        import torch  # type: ignore

        pkl_path = sample_dir / "results" / "gvhmr" / "result.pkl"
        if not pkl_path.exists():
            return None
        with pkl_path.open("rb") as f:
            data = pickle.load(f)
        params = data["smpl_params_incam"]
        transl = np.asarray(params["transl"], dtype=np.float32)
        model = smplx.create(
            str(BODY_MODELS_ROOT),
            model_type="smplx",
            gender="neutral",
            ext="npz",
            use_pca=False,
            flat_hand_mean=True,
            num_betas=10,
            batch_size=transl.shape[0],
        )
        with torch.inference_mode():
            out = model(
                body_pose=torch.from_numpy(np.asarray(params["body_pose"], dtype=np.float32)),
                betas=torch.from_numpy(np.asarray(params["betas"], dtype=np.float32)),
                global_orient=torch.from_numpy(np.asarray(params["global_orient"], dtype=np.float32)),
                transl=torch.from_numpy(transl),
                return_verts=False,
            )
        return out.joints.detach().cpu().numpy().astype(np.float64)
    except Exception as exc:
        print(f"[geometry_contact_probe] warning: could not build SMPL-X joints for {sample_dir}: {exc}")
        return None


def human_point_from_row(row: dict[str, str], joints_frame: np.ndarray | None) -> tuple[np.ndarray | None, str]:
    if joints_frame is None:
        return None, "missing_joints"
    hand_side = (row.get("hand_side") or row.get("human_side") or row.get("stable_grasp_hand_side") or row.get("active_label") or row.get("active_side") or "").lower()
    human_part = (row.get("human_part") or row.get("contact_label") or row.get("active_part") or "").lower()
    label = f"{hand_side}:{human_part}"
    if "left" in hand_side and ("hand" in human_part or "palm" in human_part or "hand" in hand_side):
        return joints_frame[LEFT_PALM_IDS].mean(axis=0), label
    if "right" in hand_side and ("hand" in human_part or "palm" in human_part or "hand" in hand_side):
        return joints_frame[RIGHT_PALM_IDS].mean(axis=0), label
    if "left" in hand_side and "foot" in human_part:
        return joints_frame[LEFT_FOOT_ID], label
    if "right" in hand_side and "foot" in human_part:
        return joints_frame[RIGHT_FOOT_ID], label
    if "left_hand" in human_part or "left_palm" in human_part:
        return joints_frame[LEFT_PALM_IDS].mean(axis=0), label
    if "right_hand" in human_part or "right_palm" in human_part:
        return joints_frame[RIGHT_PALM_IDS].mean(axis=0), label
    if "left_foot" in human_part:
        return joints_frame[LEFT_FOOT_ID], label
    if "right_foot" in human_part:
        return joints_frame[RIGHT_FOOT_ID], label
    return None, label or "unknown"


def proxy_radius(case: str, pose_rows: list[dict[str, str]], local_pts: np.ndarray) -> float | None:
    vals = [ff(r.get("radius_m")) for r in pose_rows]
    vals = [v for v in vals if v is not None]
    if vals:
        return float(statistics.median(vals))
    if local_pts.size:
        center = np.mean(local_pts, axis=0)
        d = np.linalg.norm(local_pts - center[None, :], axis=1)
        if len(d):
            # Use a conservative proxy radius for root/object clearance. For
            # chair this is not a collision result; it only catches huge flips.
            return float(np.quantile(d, 0.75 if case == "chair" else 0.95))
    return None


def summarize(values: list[float]) -> dict[str, object]:
    return {
        "count": len([v for v in values if math.isfinite(v)]),
        "median": median(values),
        "mean": mean(values),
        "p95": p95(values),
    }


def compute_variant(case: str, variant: str) -> tuple[list[dict[str, object]], dict[str, object]]:
    profile = load_case_profile(case)
    pose_path = choose_pose_path(case, variant)
    contact_path = choose_contact_path(case, variant)
    local_path = choose_local_points_path(case, variant)
    pose_rows = read_csv(pose_path) if pose_path and pose_path.exists() else []
    contact_rows = read_csv(contact_path) if contact_path and contact_path.exists() else []
    local_pts, local_by_id = load_local_points(local_path)
    joints = load_human_joints(profile.sample_dir)
    poses_by_frame = {int(ff(r.get("frame")) or i + 1): r for i, r in enumerate(pose_rows)}
    radius = proxy_radius(case, pose_rows, local_pts)
    per_frame: list[dict[str, object]] = []
    surface_res: list[float] = []
    human_gap: list[float] = []
    root_clearance: list[float] = []
    root_violation: list[float] = []
    built_in_gap: list[float] = []
    # Ball baselines preserve the most complete contact/depth labels on the
    # final pose CSV, while mug/chair preserve semantic contact anchors in
    # their contact tables.
    rows_for_contact = pose_rows if case in {"basketball", "football"} else (contact_rows if contact_rows else pose_rows)

    for i, row in enumerate(rows_for_contact):
        frame = int(ff(row.get("frame")) or i + 1)
        pose = poses_by_frame.get(frame)
        if not pose:
            continue
        center = pose_center(pose)
        local = contact_local_point(row, local_by_id)
        active = is_active(row, case)
        nearest_surface = None
        if local is not None and local_pts.size:
            nearest_surface = float(np.min(np.linalg.norm(local_pts - local[None, :], axis=1)))
            surface_res.append(nearest_surface)
        contact_cam = None
        if local is not None:
            contact_cam = local_to_cam(local, pose)
        elif center is not None and case in {"basketball", "football"}:
            contact_cam = center
        hg = None
        hlabel = ""
        if active and joints is not None and frame - 1 < len(joints):
            hp, hlabel = human_point_from_row(row, joints[frame - 1])
            if hp is not None and center is not None:
                if case in {"basketball", "football"} and radius is not None:
                    hg = abs(float(np.linalg.norm(hp - center)) - radius)
                elif contact_cam is not None:
                    hg = float(np.linalg.norm(hp - contact_cam))
                if hg is not None:
                    human_gap.append(hg)
        built = first(row, ["hand_contact_3d_gap_m", "contact_gap_m", "gap_m", "hand_object_z_gap_m", "contact_depth_gap"])
        if active and built is not None:
            built_in_gap.append(abs(built))
        rc = rv = None
        if center is not None and joints is not None and frame - 1 < len(joints) and radius is not None:
            root = joints[frame - 1, 0]
            rc = float(np.linalg.norm(center - root) - radius)
            rv = max(0.0, 0.25 - rc)
            root_clearance.append(rc)
            root_violation.append(rv)
        per_frame.append(
            {
                "case": case,
                "variant": variant,
                "frame": frame,
                "contact_active": int(active),
                "human_label": hlabel,
                "contact_local_surface_nn_m": nearest_surface,
                "mover_style_human_contact_gap_m": hg,
                "reported_contact_gap_m": abs(built) if built is not None else None,
                "root_object_clearance_proxy_m": rc,
                "root_collision_violation_proxy_m": rv,
                "pose_path": str(pose_path) if pose_path else "",
                "contact_path": str(contact_path) if contact_path else "",
                "local_points_path": str(local_path) if local_path else "",
            }
        )

    metrics = {
        "case": case,
        "variant": variant,
        "pose_path": str(pose_path) if pose_path else "",
        "contact_path": str(contact_path) if contact_path else "",
        "local_points_path": str(local_path) if local_path else "",
        "local_point_count": int(len(local_pts)),
        "has_smplx_joints": bool(joints is not None),
        "object_proxy_radius_m": radius,
        "surface_contact_nn": summarize(surface_res),
        "mover_style_human_contact_gap": summarize(human_gap),
        "reported_contact_gap": summarize(built_in_gap),
        "root_object_clearance_proxy": summarize(root_clearance),
        "root_collision_violation_proxy": summarize(root_violation),
        "note": "Proxy only: no dense SDF volume; human_contact_gap uses SMPL-X palm/foot joints and object contact anchors.",
    }
    return per_frame, metrics


def flatten_metric(case: str, variant: str, metric_name: str, stats: dict[str, object]) -> dict[str, object]:
    return {
        "case": case,
        "variant": variant,
        "metric": metric_name,
        "count": stats.get("count"),
        "median": stats.get("median"),
        "mean": stats.get("mean"),
        "p95": stats.get("p95"),
        "lower_is_better": 1 if metric_name != "root_object_clearance_proxy" else 0,
    }


def run(cases: list[str]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_frame_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    raw: dict[str, Any] = {}
    for case in cases:
        raw[case] = {}
        for variant in ["baseline", "generic"]:
            pf, metrics = compute_variant(case, variant)
            per_frame_rows.extend(pf)
            raw[case][variant] = metrics
            for metric_name in [
                "surface_contact_nn",
                "mover_style_human_contact_gap",
                "reported_contact_gap",
                "root_object_clearance_proxy",
                "root_collision_violation_proxy",
            ]:
                metric_rows.append(flatten_metric(case, variant, metric_name, metrics[metric_name]))

    ratio_rows: list[dict[str, object]] = []
    by_key = {(r["case"], r["variant"], r["metric"]): r for r in metric_rows}
    for case in cases:
        for metric_name in [
            "surface_contact_nn",
            "mover_style_human_contact_gap",
            "reported_contact_gap",
            "root_object_clearance_proxy",
            "root_collision_violation_proxy",
        ]:
            g = by_key.get((case, "generic", metric_name), {})
            b = by_key.get((case, "baseline", metric_name), {})
            ratio_rows.append(
                {
                    "case": case,
                    "metric": metric_name,
                    "generic_median": g.get("median"),
                    "baseline_median": b.get("median"),
                    "generic_over_baseline_median": ratio(g.get("median"), b.get("median")),
                    "generic_count": g.get("count"),
                    "baseline_count": b.get("count"),
                    "interpretation": "lower better except root_object_clearance_proxy",
                }
            )

    write_csv(OUT_DIR / "related_work_geometry_contact_probe_metrics.csv", metric_rows)
    write_csv(OUT_DIR / "related_work_geometry_contact_probe_ratios.csv", ratio_rows)
    write_csv(OUT_DIR / "related_work_geometry_contact_probe_per_frame.csv", per_frame_rows)
    write_json(OUT_DIR / "related_work_geometry_contact_probe_metrics.json", raw)
    lines = [
        "# Related Work Geometry/Contact Probe",
        "",
        "This is an AudioHOI proxy test inspired by MOVER/Open3DHOI contact and collision losses.",
        "It does not claim to run MOVER SDF. It checks what is available now: object-local contact anchors, SMPL-X palm/foot joints, and coarse root/object clearance.",
        "",
        "## Generic / Baseline Ratios",
        "",
        "| Case | Metric | Generic median | Baseline median | Ratio | Counts g/b |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in ratio_rows:
        lines.append(
            f"| {row['case']} | {row['metric']} | {row.get('generic_median')} | {row.get('baseline_median')} | {row.get('generic_over_baseline_median')} | {row.get('generic_count')}/{row.get('baseline_count')} |"
        )
    lines.extend(
        [
            "",
            "## What The Metrics Mean",
            "",
            "- `surface_contact_nn`: nearest-neighbor distance from the chosen object-local contact anchor to sampled local object points. This is the closest available analogue to an object-side contact Chamfer.",
            "- `mover_style_human_contact_gap`: SMPL-X palm/foot to object contact point distance. For balls it is `abs(||human-center|| - radius)`; for mug/chair it is palm-to-local-anchor distance after the pose transform.",
            "- `reported_contact_gap`: existing pipeline contact/depth gap columns, kept as a sanity bridge to previous probes.",
            "- `root_object_clearance_proxy`: root joint to object-center distance minus object proxy radius. It is a coarse collision guard, not dense body-object SDF.",
            "- `root_collision_violation_proxy`: `max(0, 0.25m - clearance)`. This only flags gross center/root overlap.",
        ]
    )
    (OUT_DIR / "related_work_geometry_contact_probe_summary.md").write_text("\n".join(lines) + "\n")
    print(f"[geometry_contact_probe] wrote {len(metric_rows)} metric rows, {len(per_frame_rows)} per-frame rows")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="all", choices=["all"] + available_cases())
    args = ap.parse_args()
    cases = available_cases() if args.case == "all" else [args.case]
    run(cases)


if __name__ == "__main__":
    main()
