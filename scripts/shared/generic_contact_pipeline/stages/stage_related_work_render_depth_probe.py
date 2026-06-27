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
    load_human_vertices,
    object_vertices_for_case,
)


OUT_DIR = REPO / "docs" / "related_work_audit"


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


def load_K(sample_dir: Path) -> np.ndarray:
    p = sample_dir / "results" / "gvhmr" / "result.pkl"
    with p.open("rb") as f:
        data = pickle.load(f)
    K = np.asarray(data["K_fullimg"], dtype=np.float64)
    return K[0] if K.ndim == 3 else K


def zbuffer_points(points: np.ndarray, K: np.ndarray, bin_size: int, image_w: int, image_h: int) -> np.ndarray:
    grid_w = max(1, int(math.ceil(image_w / bin_size)))
    grid_h = max(1, int(math.ceil(image_h / bin_size)))
    zbuf = np.full((grid_h, grid_w), np.inf, dtype=np.float64)
    if len(points) == 0:
        return zbuf
    z = points[:, 2]
    valid = np.isfinite(z) & (z > 1e-5)
    if not np.any(valid):
        return zbuf
    pts = points[valid]
    z = pts[:, 2]
    u = K[0, 0] * pts[:, 0] / z + K[0, 2]
    v = K[1, 1] * pts[:, 1] / z + K[1, 2]
    gx = np.floor(u / bin_size).astype(np.int64)
    gy = np.floor(v / bin_size).astype(np.int64)
    inside = (gx >= 0) & (gx < grid_w) & (gy >= 0) & (gy < grid_h)
    if not np.any(inside):
        return zbuf
    flat_idx = gy[inside] * grid_w + gx[inside]
    flat = zbuf.reshape(-1)
    np.minimum.at(flat, flat_idx, z[inside])
    return zbuf


def summarize(vals: list[float]) -> dict[str, object]:
    return {"count": len([v for v in vals if math.isfinite(v)]), "median": median(vals), "mean": mean(vals), "p95": p95(vals)}


def compute_variant(case: str, variant: str, body_stride: int, bin_size: int) -> tuple[list[dict[str, object]], dict[str, object]]:
    profile = load_case_profile(case)
    pose_path = choose_pose_path(case, variant)
    if pose_path is None or not pose_path.exists():
        return [], {"case": case, "variant": variant, "missing": "pose_path"}
    rows = read_csv(pose_path)
    K = load_K(profile.sample_dir)
    image_w = int(round(K[0, 2] * 2.0))
    image_h = int(round(K[1, 2] * 2.0))
    body = load_human_vertices(profile.sample_dir, stride=body_stride)
    obj_by_frame = object_vertices_for_case(case, rows)
    contact_frames = active_contact_frames(case, variant, rows)

    per_frame: list[dict[str, object]] = []
    overlap_abs: list[float] = []
    overlap_abs_contact: list[float] = []
    signed_median: list[float] = []
    object_front_ratio: list[float] = []
    near_depth_ratio_5cm: list[float] = []
    overlap_counts: list[float] = []

    for i, row in enumerate(rows):
        fr = int(ff(row.get("frame")) or i + 1)
        if fr - 1 < 0 or fr - 1 >= len(body):
            continue
        obj = obj_by_frame.get(fr)
        if obj is None or len(obj) == 0:
            continue
        human_z = zbuffer_points(body[fr - 1], K, bin_size, image_w, image_h)
        object_z = zbuffer_points(obj, K, bin_size, image_w, image_h)
        overlap = np.isfinite(human_z) & np.isfinite(object_z)
        active = fr in contact_frames
        if not np.any(overlap):
            per_frame.append(
                {
                    "case": case,
                    "variant": variant,
                    "frame": fr,
                    "contact_active": int(active),
                    "overlap_bin_count": 0,
                    "render_depth_abs_gap_median_m": "",
                    "render_depth_signed_gap_median_m": "",
                    "object_in_front_ratio": "",
                    "render_depth_near_ratio_5cm": "",
                    "pose_path": str(pose_path),
                }
            )
            continue
        signed = object_z[overlap] - human_z[overlap]
        abs_gap = np.abs(signed)
        med_abs = float(np.median(abs_gap))
        med_signed = float(np.median(signed))
        front = float(np.mean(signed < 0.0))
        near5 = float(np.mean(abs_gap < 0.05))
        count = int(np.sum(overlap))
        overlap_abs.append(med_abs)
        if active:
            overlap_abs_contact.append(med_abs)
        signed_median.append(med_signed)
        object_front_ratio.append(front)
        near_depth_ratio_5cm.append(near5)
        overlap_counts.append(float(count))
        per_frame.append(
            {
                "case": case,
                "variant": variant,
                "frame": fr,
                "contact_active": int(active),
                "overlap_bin_count": count,
                "render_depth_abs_gap_median_m": med_abs,
                "render_depth_signed_gap_median_m": med_signed,
                "object_in_front_ratio": front,
                "render_depth_near_ratio_5cm": near5,
                "pose_path": str(pose_path),
            }
        )

    metrics = {
        "case": case,
        "variant": variant,
        "pose_path": str(pose_path),
        "body_stride": body_stride,
        "bin_size_px": bin_size,
        "image_w": image_w,
        "image_h": image_h,
        "frame_count": len(rows),
        "contact_frame_count": len(contact_frames),
        "overlap_bin_count": summarize(overlap_counts),
        "render_depth_abs_gap": summarize(overlap_abs),
        "render_depth_contact_abs_gap": summarize(overlap_abs_contact),
        "render_depth_signed_gap": summarize(signed_median),
        "object_in_front_ratio": summarize(object_front_ratio),
        "render_depth_near_ratio_5cm": summarize(near_depth_ratio_5cm),
        "note": "Point-splat z-buffer proxy. This approximates rendered ordinal/depth overlap, not a differentiable rasterizer.",
    }
    return per_frame, metrics


def metric_rows_from(metrics: dict[str, object]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for name in [
        "overlap_bin_count",
        "render_depth_abs_gap",
        "render_depth_contact_abs_gap",
        "render_depth_signed_gap",
        "object_in_front_ratio",
        "render_depth_near_ratio_5cm",
    ]:
        stats = metrics.get(name)
        if not isinstance(stats, dict):
            continue
        out.append(
            {
                "case": metrics["case"],
                "variant": metrics["variant"],
                "metric": name,
                "count": stats.get("count"),
                "median": stats.get("median"),
                "mean": stats.get("mean"),
                "p95": stats.get("p95"),
                "lower_is_better": 1 if "abs_gap" in name else 0,
                "pose_path": metrics.get("pose_path", ""),
            }
        )
    return out


def run(cases: list[str], body_stride: int, bin_size: int) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_frame_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    raw: dict[str, Any] = {}
    for case in cases:
        raw[case] = {}
        for variant in ["baseline", "generic"]:
            pf, metrics = compute_variant(case, variant, body_stride, bin_size)
            per_frame_rows.extend(pf)
            raw[case][variant] = metrics
            metric_rows.extend(metric_rows_from(metrics))
    by_key = {(r["case"], r["variant"], r["metric"]): r for r in metric_rows}
    ratio_rows: list[dict[str, object]] = []
    for case in cases:
        for name in [
            "overlap_bin_count",
            "render_depth_abs_gap",
            "render_depth_contact_abs_gap",
            "render_depth_signed_gap",
            "object_in_front_ratio",
            "render_depth_near_ratio_5cm",
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
                    "interpretation": "abs depth gaps lower better; overlap/front/near ratios are diagnostic",
                }
            )

    write_csv(OUT_DIR / "related_work_render_depth_probe_metrics.csv", metric_rows)
    write_csv(OUT_DIR / "related_work_render_depth_probe_ratios.csv", ratio_rows)
    write_csv(OUT_DIR / "related_work_render_depth_probe_per_frame.csv", per_frame_rows)
    write_json(OUT_DIR / "related_work_render_depth_probe_metrics.json", raw)
    lines = [
        "# Related Work Render-Depth Probe",
        "",
        "This probe approximates Open3DHOI/ZeroHSI/MOVER rendered depth and ordinal-depth checks.",
        "It splats sampled SMPL-X and object vertices into a coarse z-buffer under GVHMR K_fullimg, then compares depth only in overlapping bins.",
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
            "- `render_depth_abs_gap`: median absolute object-vs-human z-buffer depth gap on overlapping bins.",
            "- `render_depth_contact_abs_gap`: same metric restricted to contact-active frames.",
            "- `render_depth_signed_gap`: object depth minus human depth. Negative means object is in front in overlapping bins.",
            "- `object_in_front_ratio`: fraction of overlapping bins where object depth is closer than human depth.",
            "- `render_depth_near_ratio_5cm`: fraction of overlapping bins whose absolute depth gap is below 5cm.",
            "- This is a point-splat proxy, not differentiable Gaussian/PyTorch3D rasterization.",
        ]
    )
    (OUT_DIR / "related_work_render_depth_probe_summary.md").write_text("\n".join(lines) + "\n")
    print(f"[render_depth_probe] wrote {len(metric_rows)} metrics, {len(per_frame_rows)} per-frame rows")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="all", choices=["all"] + available_cases())
    ap.add_argument("--body-stride", type=int, default=40)
    ap.add_argument("--bin-size", type=int, default=8)
    args = ap.parse_args()
    cases = available_cases() if args.case == "all" else [args.case]
    run(cases, body_stride=args.body_stride, bin_size=args.bin_size)


if __name__ == "__main__":
    main()
