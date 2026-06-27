#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import pickle
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

if __package__ in {None, ""}:
    REPO = Path(__file__).resolve().parents[5]
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
else:
    REPO = Path(__file__).resolve().parents[5]

from scripts.shared.generic_contact_pipeline.core.base.config import available_cases, load_case_profile  # noqa: E402
from scripts.shared.generic_contact_pipeline.core.base.io import read_csv, repo_path, write_csv, write_json  # noqa: E402


OUT_DIR = REPO / "docs" / "related_work_audit"
BODY_MODELS_ROOT = REPO / "third-party" / "GVHMR" / "inputs" / "checkpoints" / "body_models"


def ff(value: object) -> float | None:
    try:
        if value in ("", None):
            return None
        out = float(value)  # type: ignore[arg-type]
        return out if math.isfinite(out) else None
    except Exception:
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
    aa, bb = ff(a), ff(b)
    if aa is None or bb is None or abs(bb) < 1e-12:
        return None
    return aa / bb


def choose_pose_path(case: str, variant: str) -> Path | None:
    profile = load_case_profile(case)
    if variant == "generic":
        p = profile.result_dir / "object_pose.csv"
        return p if p.exists() else None
    rel = profile.baseline.get("final_pose_csv")
    if not rel:
        return None
    p = repo_path(rel)
    return p if p.exists() else None


def choose_contact_path(case: str, variant: str) -> Path | None:
    profile = load_case_profile(case)
    if variant == "generic":
        for p in [profile.result_dir / "object_contact_points.csv", profile.result_dir / "contact_candidates.csv"]:
            if p.exists():
                return p
    for key in ["final_contact_csv", "grasp_anchor_csv", "contact_candidates_csv", "contact_state_csv"]:
        rel = profile.baseline.get(key)
        if rel:
            p = repo_path(rel)
            if p.exists():
                return p
    return None


def is_contact_active(row: dict[str, str]) -> bool:
    for key in ["contact_frame", "contact_active", "human_contact_state", "anchor_contact_state"]:
        val = str(row.get(key, "")).strip().lower()
        if val in {"1", "true", "yes"}:
            return True
    return False


def active_contact_frames(case: str, variant: str, pose_rows: list[dict[str, str]]) -> set[int]:
    frames: set[int] = set()
    for i, row in enumerate(pose_rows):
        if is_contact_active(row):
            frames.add(int(ff(row.get("frame")) or i + 1))
    contact_path = choose_contact_path(case, variant)
    if contact_path is None or not contact_path.exists():
        return frames
    for i, row in enumerate(read_csv(contact_path)):
        fr = int(ff(row.get("frame")) or i + 1)
        if is_contact_active(row):
            frames.add(fr)
            continue
        if case == "mug" and any(
            str(row.get(k, "")).strip().lower() in {"1", "true", "yes"}
            for k in ["use_this_point_for_hand_attachment", "use_previous_grasp_for_hand_attachment"]
        ):
            frames.add(fr)
        if case == "chair" and str(row.get("contact_active", "")).strip().lower() in {"1", "true", "yes"}:
            frames.add(fr)
    return frames


def pose_center(row: dict[str, str]) -> np.ndarray | None:
    vals = [ff(row.get(k)) for k in ("tx", "ty", "tz")]
    if all(v is not None for v in vals):
        return np.asarray(vals, dtype=float)
    vals = [ff(row.get(k)) for k in ("x", "y", "z")]
    if all(v is not None for v in vals):
        return np.asarray(vals, dtype=float)
    return None


def read_obj_vertices(path: Path) -> np.ndarray:
    pts: list[list[float]] = []
    with path.open(errors="ignore") as f:
        for line in f:
            if not line.startswith("v "):
                continue
            parts = line.strip().split()
            if len(parts) >= 4:
                pts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.asarray(pts, dtype=np.float64)


def sphere_vertices(radius: float, n_lat: int = 12, n_lon: int = 32) -> np.ndarray:
    pts: list[list[float]] = []
    for i in range(1, n_lat):
        theta = math.pi * i / n_lat
        for j in range(n_lon):
            phi = 2.0 * math.pi * j / n_lon
            pts.append([
                radius * math.sin(theta) * math.cos(phi),
                radius * math.cos(theta),
                radius * math.sin(theta) * math.sin(phi),
            ])
    pts.append([0.0, radius, 0.0])
    pts.append([0.0, -radius, 0.0])
    return np.asarray(pts, dtype=np.float64)


def load_human_vertices(sample_dir: Path, stride: int) -> np.ndarray:
    import smplx  # type: ignore
    import torch  # type: ignore

    pkl_path = sample_dir / "results" / "gvhmr" / "result.pkl"
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
            return_verts=True,
        )
    verts = out.vertices.detach().cpu().numpy().astype(np.float64)
    return verts[:, :: max(1, stride), :]


def mug_pose_vec(row: dict[str, str]) -> np.ndarray | None:
    vals = [ff(row.get(k)) for k in ["x", "y", "z", "yaw", "pitch", "roll", "scale"]]
    if not all(v is not None for v in vals):
        return None
    return np.asarray(vals, dtype=np.float64)


def mug_object_vertices(rows: list[dict[str, str]]) -> dict[int, np.ndarray]:
    render_dir = REPO / "scripts/shared/generic_contact_pipeline/components/render"
    if str(render_dir) not in sys.path:
        sys.path.insert(0, str(render_dir))
    import render_mug_articraft_camera3d as m16  # type: ignore

    mesh_root = REPO / "samples_known_object/02_mug/articraft/materialized_mug_mesh/assets/meshes"
    local_parts = []
    for obj in sorted(mesh_root.glob("*.obj")):
        verts = read_obj_vertices(obj)
        if len(verts):
            local_parts.append(verts[:: max(1, len(verts) // 450)])
    local = np.concatenate(local_parts, axis=0)
    out: dict[int, np.ndarray] = {}
    for i, row in enumerate(rows):
        fr = int(ff(row.get("frame")) or i + 1)
        pose = mug_pose_vec(row)
        if pose is None:
            continue
        phase = ff(row.get("handle_phase_rad")) or 0.0
        out[fr] = m16.transform_mesh(pose, local, phase)
    return out


def ball_object_vertices(rows: list[dict[str, str]]) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for i, row in enumerate(rows):
        fr = int(ff(row.get("frame")) or i + 1)
        center = pose_center(row)
        radius = ff(row.get("radius_m")) or 0.12
        if center is None:
            continue
        out[fr] = sphere_vertices(radius) + center[None, :]
    return out


def chair_object_vertices(rows: list[dict[str, str]]) -> dict[int, np.ndarray]:
    render_dir = REPO / "scripts/shared/generic_contact_pipeline/components/render"
    if str(render_dir) not in sys.path:
        sys.path.insert(0, str(render_dir))
    import render_chair_articraft_visual_model as vismod  # type: ignore
    import render_chair_articraft_3d_scene as scene3d  # type: ignore

    urdf = REPO / "samples_known_object/05_chair/articraft/generated_record/rec_fork-6911e934-oldtopology-rearup-stretcherlevel-pass_20260619_1056/model.urdf"
    visuals = vismod.load_articraft_visuals(urdf)
    out: dict[int, np.ndarray] = {}
    for i, row in enumerate(rows):
        fr = int(ff(row.get("frame")) or i + 1)
        parts = []
        for mesh in vismod.visual_cam_meshes(scene3d.pose_params(row), visuals):
            cam = np.asarray(mesh["cam"], dtype=np.float64)
            if len(cam):
                parts.append(cam[:: max(1, len(cam) // 220)])
        if parts:
            out[fr] = np.concatenate(parts, axis=0)
    return out


def object_vertices_for_case(case: str, rows: list[dict[str, str]]) -> dict[int, np.ndarray]:
    if case in {"basketball", "football"}:
        return ball_object_vertices(rows)
    if case == "mug":
        return mug_object_vertices(rows)
    if case == "chair":
        return chair_object_vertices(rows)
    raise ValueError(case)


def summarize(vals: list[float]) -> dict[str, object]:
    return {"count": len([v for v in vals if math.isfinite(v)]), "median": median(vals), "mean": mean(vals), "p95": p95(vals)}


def compute_variant(case: str, variant: str, body_stride: int) -> tuple[list[dict[str, object]], dict[str, object]]:
    profile = load_case_profile(case)
    pose_path = choose_pose_path(case, variant)
    if pose_path is None or not pose_path.exists():
        return [], {"case": case, "variant": variant, "missing": "pose_path"}
    rows = read_csv(pose_path)
    contact_frames = active_contact_frames(case, variant, rows)
    body = load_human_vertices(profile.sample_dir, stride=body_stride)
    obj_by_frame = object_vertices_for_case(case, rows)
    per_frame: list[dict[str, object]] = []
    min_gaps: list[float] = []
    contact_min_gaps: list[float] = []
    near_ratios_2cm: list[float] = []
    near_ratios_5cm: list[float] = []
    object_counts: list[int] = []
    body_counts: list[int] = []
    for i, row in enumerate(rows):
        fr = int(ff(row.get("frame")) or i + 1)
        if fr - 1 < 0 or fr - 1 >= len(body):
            continue
        obj = obj_by_frame.get(fr)
        if obj is None or len(obj) == 0:
            continue
        body_pts = body[fr - 1]
        tree = cKDTree(body_pts)
        dist, _ = tree.query(obj, k=1, workers=-1)
        dist = np.asarray(dist, dtype=np.float64)
        min_gap = float(np.min(dist))
        med_gap = float(np.median(dist))
        near2 = float(np.mean(dist < 0.02))
        near5 = float(np.mean(dist < 0.05))
        active = is_contact_active(row) or fr in contact_frames
        min_gaps.append(min_gap)
        if active:
            contact_min_gaps.append(min_gap)
        near_ratios_2cm.append(near2)
        near_ratios_5cm.append(near5)
        object_counts.append(len(obj))
        body_counts.append(len(body_pts))
        per_frame.append(
            {
                "case": case,
                "variant": variant,
                "frame": fr,
                "contact_active": int(active),
                "dense_mesh_min_gap_m": min_gap,
                "dense_mesh_median_gap_m": med_gap,
                "object_vertex_ratio_within_2cm": near2,
                "object_vertex_ratio_within_5cm": near5,
                "object_vertex_count": len(obj),
                "body_vertex_count": len(body_pts),
                "pose_path": str(pose_path),
            }
        )
    metrics = {
        "case": case,
        "variant": variant,
        "pose_path": str(pose_path),
        "contact_frame_count": len(contact_frames),
        "body_stride": body_stride,
        "frame_count": len(rows),
        "object_vertex_count_median": median([float(v) for v in object_counts]),
        "body_vertex_count_median": median([float(v) for v in body_counts]),
        "dense_mesh_min_gap": summarize(min_gaps),
        "dense_mesh_contact_min_gap": summarize(contact_min_gaps),
        "object_vertex_ratio_within_2cm": summarize(near_ratios_2cm),
        "object_vertex_ratio_within_5cm": summarize(near_ratios_5cm),
        "note": "Sampled object vertices to sampled SMPL-X vertices. This is a dense proximity proxy, not signed SDF penetration.",
    }
    return per_frame, metrics


def metric_rows_from(metrics: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name in ["dense_mesh_min_gap", "dense_mesh_contact_min_gap", "object_vertex_ratio_within_2cm", "object_vertex_ratio_within_5cm"]:
        stats = metrics.get(name)
        if isinstance(stats, dict):
            rows.append(
                {
                    "case": metrics["case"],
                    "variant": metrics["variant"],
                    "metric": name,
                    "count": stats.get("count"),
                    "median": stats.get("median"),
                    "mean": stats.get("mean"),
                    "p95": stats.get("p95"),
                    "lower_is_better": 1 if "gap" in name else 0,
                    "pose_path": metrics.get("pose_path", ""),
                }
            )
    return rows


def run(cases: list[str], body_stride: int) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_pf: list[dict[str, object]] = []
    all_metrics: list[dict[str, object]] = []
    raw: dict[str, Any] = {}
    for case in cases:
        raw[case] = {}
        for variant in ["baseline", "generic"]:
            pf, metrics = compute_variant(case, variant, body_stride)
            all_pf.extend(pf)
            raw[case][variant] = metrics
            all_metrics.extend(metric_rows_from(metrics))
    by_key = {(r["case"], r["variant"], r["metric"]): r for r in all_metrics}
    ratio_rows: list[dict[str, object]] = []
    for case in cases:
        for name in ["dense_mesh_min_gap", "dense_mesh_contact_min_gap", "object_vertex_ratio_within_2cm", "object_vertex_ratio_within_5cm"]:
            g = by_key.get((case, "generic", name), {})
            b = by_key.get((case, "baseline", name), {})
            ratio_rows.append(
                {
                    "case": case,
                    "metric": name,
                    "generic_median": g.get("median"),
                    "baseline_median": b.get("median"),
                    "generic_over_baseline_median": ratio(g.get("median"), b.get("median")),
                    "generic_count": g.get("count"),
                    "baseline_count": b.get("count"),
                    "interpretation": "gaps lower better; contact vertex ratios higher better",
                }
            )
    write_csv(OUT_DIR / "related_work_dense_mesh_probe_metrics.csv", all_metrics)
    write_csv(OUT_DIR / "related_work_dense_mesh_probe_ratios.csv", ratio_rows)
    write_csv(OUT_DIR / "related_work_dense_mesh_probe_per_frame.csv", all_pf)
    write_json(OUT_DIR / "related_work_dense_mesh_probe_metrics.json", raw)
    lines = [
        "# Related Work Dense Mesh Probe",
        "",
        "This probe approximates InterCap/MOVER/Open3DHOI mesh-level contact/collision checks using sampled SMPL-X vertices and sampled object vertices.",
        "It is not signed SDF penetration. It measures nearest human-object surface proximity in the same GVHMR camera frame.",
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
            "## Metrics",
            "",
            "- `dense_mesh_min_gap`: per-frame minimum nearest-neighbor distance from sampled object vertices to sampled SMPL-X vertices.",
            "- `dense_mesh_contact_min_gap`: same gap but only on frames with contact flags in the pose CSV.",
            "- `object_vertex_ratio_within_2cm/5cm`: fraction of object vertices close to the human surface; higher means broader near-contact or possible overlap.",
            "- Because this uses unsigned nearest-neighbor distance, penetration requires a future SDF/inside-outside test.",
        ]
    )
    (OUT_DIR / "related_work_dense_mesh_probe_summary.md").write_text("\n".join(lines) + "\n")
    print(f"[dense_mesh_probe] wrote {len(all_metrics)} metrics, {len(all_pf)} per-frame rows")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="all", choices=["all"] + available_cases())
    ap.add_argument("--body-stride", type=int, default=24)
    args = ap.parse_args()
    cases = available_cases() if args.case == "all" else [args.case]
    run(cases, body_stride=args.body_stride)


if __name__ == "__main__":
    main()
