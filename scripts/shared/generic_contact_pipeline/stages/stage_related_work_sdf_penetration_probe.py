#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import pickle
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

if __package__ in {None, ""}:
    REPO = Path(__file__).resolve().parents[4]
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
else:
    REPO = Path(__file__).resolve().parents[4]

from scripts.shared.generic_contact_pipeline.core.config import available_cases, load_case_profile  # noqa: E402
from scripts.shared.generic_contact_pipeline.core.io import read_csv, write_csv, write_json  # noqa: E402
from scripts.shared.generic_contact_pipeline.stages.stage_related_work_dense_mesh_probe import (  # noqa: E402
    active_contact_frames,
    choose_pose_path,
    ff,
    object_vertices_for_case,
)


OUT_DIR = REPO / "docs" / "related_work_audit"
BODY_MODELS_ROOT = REPO / "third-party" / "GVHMR" / "inputs" / "checkpoints" / "body_models"


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


def summarize(vals: list[float]) -> dict[str, object]:
    return {"count": len([v for v in vals if math.isfinite(v)]), "median": median(vals), "mean": mean(vals), "p95": p95(vals)}


def select_frames(rows: list[dict[str, str]], contact_frames: set[int], step: int) -> list[int]:
    frames = [int(ff(r.get("frame")) or i + 1) for i, r in enumerate(rows)]
    selected = {fr for fr in frames if (fr - 1) % max(1, step) == 0}
    selected.update(contact_frames)
    return sorted(fr for fr in selected if fr in set(frames))


def cap_frames(frames: list[int], max_frames: int) -> list[int]:
    if max_frames <= 0 or len(frames) <= max_frames:
        return frames
    idx = np.linspace(0, len(frames) - 1, max_frames)
    chosen = sorted({frames[int(round(i))] for i in idx})
    return chosen


def load_human_mesh_vertices(sample_dir: Path, frame_numbers: list[int]) -> tuple[dict[int, np.ndarray], np.ndarray]:
    import smplx  # type: ignore
    import torch  # type: ignore

    pkl_path = sample_dir / "results" / "gvhmr" / "result.pkl"
    with pkl_path.open("rb") as f:
        data = pickle.load(f)
    params = data["smpl_params_incam"]
    idx = np.asarray([fr - 1 for fr in frame_numbers], dtype=np.int64)
    model = smplx.create(
        str(BODY_MODELS_ROOT),
        model_type="smplx",
        gender="neutral",
        ext="npz",
        use_pca=False,
        flat_hand_mean=True,
        num_betas=10,
        batch_size=len(idx),
    )
    with torch.inference_mode():
        out = model(
            body_pose=torch.from_numpy(np.asarray(params["body_pose"][idx], dtype=np.float32)),
            betas=torch.from_numpy(np.asarray(params["betas"][idx], dtype=np.float32)),
            global_orient=torch.from_numpy(np.asarray(params["global_orient"][idx], dtype=np.float32)),
            transl=torch.from_numpy(np.asarray(params["transl"][idx], dtype=np.float32)),
            return_verts=True,
        )
    verts = out.vertices.detach().cpu().numpy().astype(np.float64)
    return {fr: verts[i] for i, fr in enumerate(frame_numbers)}, np.asarray(model.faces, dtype=np.int64)


def signed_distance_to_body(body_vertices: np.ndarray, faces: np.ndarray, object_vertices: np.ndarray) -> np.ndarray:
    mesh = trimesh.Trimesh(vertices=body_vertices, faces=faces, process=False)
    # trimesh signed_distance is positive inside the mesh and negative outside.
    return np.asarray(trimesh.proximity.signed_distance(mesh, object_vertices), dtype=np.float64)


def compute_variant(case: str, variant: str, frame_step: int, max_object_vertices: int, max_selected_frames: int) -> tuple[list[dict[str, object]], dict[str, object]]:
    profile = load_case_profile(case)
    pose_path = choose_pose_path(case, variant)
    if pose_path is None or not pose_path.exists():
        return [], {"case": case, "variant": variant, "missing": "pose_path"}
    rows = read_csv(pose_path)
    contact_frames = active_contact_frames(case, variant, rows)
    selected_uncapped = select_frames(rows, contact_frames, frame_step)
    selected = cap_frames(selected_uncapped, max_selected_frames)
    body_by_frame, faces = load_human_mesh_vertices(profile.sample_dir, selected)
    object_by_frame = object_vertices_for_case(case, rows)

    per_frame: list[dict[str, object]] = []
    max_pen: list[float] = []
    mean_pen: list[float] = []
    ratio_pen: list[float] = []
    contact_max_pen: list[float] = []
    contact_ratio_pen: list[float] = []
    for fr in selected:
        body = body_by_frame.get(fr)
        obj = object_by_frame.get(fr)
        if body is None or obj is None or len(obj) == 0:
            continue
        if len(obj) > max_object_vertices:
            obj = obj[:: max(1, int(math.ceil(len(obj) / max_object_vertices)))]
        try:
            sd = signed_distance_to_body(body, faces, obj)
            status = "signed_distance"
        except Exception as exc:
            per_frame.append(
                {
                    "case": case,
                    "variant": variant,
                    "frame": fr,
                    "contact_active": int(fr in contact_frames),
                    "status": f"failed:{type(exc).__name__}",
                    "error": str(exc)[:200],
                    "pose_path": str(pose_path),
                }
            )
            continue
        positive = sd[sd > 0.0]
        max_v = float(np.max(positive)) if len(positive) else 0.0
        mean_v = float(np.mean(positive)) if len(positive) else 0.0
        ratio_v = float(len(positive) / max(1, len(sd)))
        max_pen.append(max_v)
        mean_pen.append(mean_v)
        ratio_pen.append(ratio_v)
        if fr in contact_frames:
            contact_max_pen.append(max_v)
            contact_ratio_pen.append(ratio_v)
        per_frame.append(
            {
                "case": case,
                "variant": variant,
                "frame": fr,
                "contact_active": int(fr in contact_frames),
                "status": status,
                "object_vertex_count": len(obj),
                "body_vertex_count": len(body),
                "max_positive_signed_distance_m": max_v,
                "mean_positive_signed_distance_m": mean_v,
                "object_vertex_inside_body_ratio": ratio_v,
                "signed_distance_median_m": float(np.median(sd)),
                "signed_distance_min_m": float(np.min(sd)),
                "signed_distance_max_m": float(np.max(sd)),
                "pose_path": str(pose_path),
            }
        )

    metrics = {
        "case": case,
        "variant": variant,
        "pose_path": str(pose_path),
        "frame_step": frame_step,
        "selected_frame_count_uncapped": len(selected_uncapped),
        "selected_frame_count": len(selected),
        "contact_frame_count": len(contact_frames),
        "body_faces": int(len(faces)),
        "sdf_max_penetration": summarize(max_pen),
        "sdf_mean_positive_penetration": summarize(mean_pen),
        "sdf_inside_vertex_ratio": summarize(ratio_pen),
        "sdf_contact_max_penetration": summarize(contact_max_pen),
        "sdf_contact_inside_vertex_ratio": summarize(contact_ratio_pen),
        "note": "trimesh signed_distance against SMPL-X body mesh. Positive values mean object vertices are inside the body mesh.",
    }
    return per_frame, metrics


def metric_rows_from(metrics: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name in [
        "sdf_max_penetration",
        "sdf_mean_positive_penetration",
        "sdf_inside_vertex_ratio",
        "sdf_contact_max_penetration",
        "sdf_contact_inside_vertex_ratio",
    ]:
        stats = metrics.get(name)
        if not isinstance(stats, dict):
            continue
        rows.append(
            {
                "case": metrics["case"],
                "variant": metrics["variant"],
                "metric": name,
                "count": stats.get("count"),
                "median": stats.get("median"),
                "mean": stats.get("mean"),
                "p95": stats.get("p95"),
                "lower_is_better": 1,
                "pose_path": metrics.get("pose_path", ""),
            }
        )
    return rows


def run(cases: list[str], frame_step: int, max_object_vertices: int, max_selected_frames: int) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_frame_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    raw: dict[str, Any] = {}
    for case in cases:
        raw[case] = {}
        for variant in ["baseline", "generic"]:
            pf, metrics = compute_variant(case, variant, frame_step, max_object_vertices, max_selected_frames)
            per_frame_rows.extend(pf)
            raw[case][variant] = metrics
            metric_rows.extend(metric_rows_from(metrics))
    by_key = {(r["case"], r["variant"], r["metric"]): r for r in metric_rows}
    ratio_rows: list[dict[str, object]] = []
    for case in cases:
        for name in [
            "sdf_max_penetration",
            "sdf_mean_positive_penetration",
            "sdf_inside_vertex_ratio",
            "sdf_contact_max_penetration",
            "sdf_contact_inside_vertex_ratio",
        ]:
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
                    "interpretation": "lower is better; positive signed distance means object vertices inside body mesh",
                }
            )
    write_csv(OUT_DIR / "related_work_sdf_penetration_probe_metrics.csv", metric_rows)
    write_csv(OUT_DIR / "related_work_sdf_penetration_probe_ratios.csv", ratio_rows)
    write_csv(OUT_DIR / "related_work_sdf_penetration_probe_per_frame.csv", per_frame_rows)
    write_json(OUT_DIR / "related_work_sdf_penetration_probe_metrics.json", raw)
    lines = [
        "# Related Work SDF Penetration Probe",
        "",
        "This probe is the closest current AudioHOI-side approximation of MOVER/InterDiff/Open3DHOI SDF penetration losses.",
        "It queries trimesh signed distance from sampled object vertices to the per-frame SMPL-X body mesh. In trimesh, positive signed distance means the point is inside the mesh.",
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
            "- `sdf_max_penetration`: max positive signed distance among sampled object vertices.",
            "- `sdf_mean_positive_penetration`: mean of positive signed distances; 0 if no sampled object vertex is inside the body.",
            "- `sdf_inside_vertex_ratio`: fraction of sampled object vertices inside the body mesh.",
            "- Contact variants restrict to frames selected by the existing contact tables.",
            "- This is still not a complete MOVER scene/body SDF pipeline, but it is a true signed-distance query against SMPL-X body meshes.",
        ]
    )
    (OUT_DIR / "related_work_sdf_penetration_probe_summary.md").write_text("\n".join(lines) + "\n")
    print(f"[sdf_penetration_probe] wrote {len(metric_rows)} metrics, {len(per_frame_rows)} per-frame rows")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="all", choices=["all"] + available_cases())
    ap.add_argument("--frame-step", type=int, default=8)
    ap.add_argument("--max-object-vertices", type=int, default=600)
    ap.add_argument("--max-selected-frames", type=int, default=140)
    args = ap.parse_args()
    cases = available_cases() if args.case == "all" else [args.case]
    run(cases, frame_step=args.frame_step, max_object_vertices=args.max_object_vertices, max_selected_frames=args.max_selected_frames)


if __name__ == "__main__":
    main()
