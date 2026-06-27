#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    REPO = Path(__file__).resolve().parents[5]
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))

from scripts.shared.generic_contact_pipeline.core.base.config import available_cases, load_case_profile  # noqa: E402
from scripts.shared.generic_contact_pipeline.core.base.io import read_csv, repo_path, write_csv, write_json  # noqa: E402


OUT_DIR = REPO / "docs" / "related_work_audit"


def f(row: dict[str, str], *keys: str) -> float | None:
    for key in keys:
        val = row.get(key)
        if val in ("", None):
            continue
        try:
            out = float(val)
        except Exception:
            continue
        if math.isfinite(out):
            return out
    return None


def mean(vals: list[float]) -> float | None:
    vals = [v for v in vals if math.isfinite(v)]
    return statistics.fmean(vals) if vals else None


def median(vals: list[float]) -> float | None:
    vals = [v for v in vals if math.isfinite(v)]
    return statistics.median(vals) if vals else None


def p95(vals: list[float]) -> float | None:
    vals = sorted(v for v in vals if math.isfinite(v))
    if not vals:
        return None
    idx = min(len(vals) - 1, int(round(0.95 * (len(vals) - 1))))
    return vals[idx]


def vec_translation(row: dict[str, str]) -> tuple[float, float, float] | None:
    vals = [f(row, "tx", "x"), f(row, "ty", "y"), f(row, "tz", "z")]
    if any(v is None for v in vals):
        return None
    return vals[0], vals[1], vals[2]  # type: ignore[return-value]


def vec_rotation(row: dict[str, str]) -> tuple[float, ...] | None:
    q = [f(row, "qx"), f(row, "qy"), f(row, "qz"), f(row, "qw")]
    if all(v is not None for v in q):
        return tuple(v for v in q if v is not None)
    euler = [f(row, "yaw"), f(row, "pitch"), f(row, "roll")]
    if all(v is not None for v in euler):
        return tuple(v for v in euler if v is not None)
    deltas = [f(row, "rx_delta"), f(row, "ry_delta"), f(row, "rz_delta")]
    if all(v is not None for v in deltas):
        return tuple(v for v in deltas if v is not None)
    return None


def l2(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def second_diff_norm(seq: list[tuple[float, ...] | None]) -> list[float]:
    out: list[float] = []
    for a, b, c in zip(seq[:-2], seq[1:-1], seq[2:]):
        if a is None or b is None or c is None:
            continue
        out.append(math.sqrt(sum((c[i] - 2 * b[i] + a[i]) ** 2 for i in range(min(len(a), len(b), len(c))))))
    return out


def velocity_norm(seq: list[tuple[float, ...] | None]) -> list[float]:
    out: list[float] = []
    for a, b in zip(seq[:-1], seq[1:]):
        if a is None or b is None:
            continue
        out.append(l2(a, b))
    return out


def visual_proxy(rows: list[dict[str, str]]) -> list[float]:
    vals: list[float] = []
    for r in rows:
        v = f(r, "residual_px", "endpoint_rmse_px", "line_rmse_px", "rail_rmse_px")
        if v is not None:
            vals.append(abs(v))
            continue
        uo, vo, up, vp = f(r, "u_obs"), f(r, "v_obs"), f(r, "u_proj"), f(r, "v_proj")
        if None not in (uo, vo, up, vp):
            vals.append(math.hypot(up - uo, vp - vo))  # type: ignore[operator]
    return vals


def contact_proxy(rows: list[dict[str, str]], extra_rows: list[dict[str, str]]) -> list[float]:
    vals: list[float] = []
    for r in rows + extra_rows:
        candidates = [
            f(r, "contact_depth_gap"),
            f(r, "hand_contact_3d_gap_m"),
            f(r, "hand_object_z_gap_m"),
            f(r, "gap_m"),
            f(r, "contact_gap_m"),
        ]
        for c in candidates:
            if c is not None:
                vals.append(abs(c))
                break
    return vals


def depth_order_proxy(rows: list[dict[str, str]], extra_rows: list[dict[str, str]]) -> list[float]:
    vals: list[float] = []
    for r in rows + extra_rows:
        v = f(r, "contact_depth_gap", "hand_object_z_gap_m", "z_ref_global_shift")
        if v is not None:
            vals.append(abs(v))
    return vals


def support_static_proxy(rows: list[dict[str, str]]) -> list[float]:
    trans = [vec_translation(r) for r in rows]
    valid = [t for t in trans if t is not None]
    if len(valid) < 5:
        return []
    tail = valid[max(0, int(0.85 * len(valid))):]
    if len(tail) < 2:
        return []
    drift = [l2(a, b) for a, b in zip(tail[:-1], tail[1:])]
    return drift


def is_active(row: dict[str, str]) -> bool:
    return str(row.get("contact_active", row.get("contact_frame", "0"))).strip().lower() in {"1", "true", "yes"}


def contact_anchor_series(extra_rows: list[dict[str, str]]) -> list[list[tuple[float, ...] | None]]:
    groups: dict[str, list[tuple[int, tuple[float, ...] | None]]] = {}
    for idx, row in enumerate(extra_rows):
        if not is_active(row):
            continue
        side = row.get("hand_side") or row.get("human_side") or row.get("active_part") or "contact"
        frame = int(f(row, "frame") or idx)
        coords = [
            f(row, "stable_local_x", "chair_local_x", "object_local_x"),
            f(row, "stable_local_y", "chair_local_y", "object_local_y"),
            f(row, "stable_local_z", "chair_local_z", "object_local_z"),
        ]
        if all(v is not None for v in coords):
            item: tuple[float, ...] | None = tuple(v for v in coords if v is not None)
        else:
            offset = f(row, "contact_depth_offset_m", "hand_object_z_gap_m", "contact_depth_gap")
            item = (offset,) if offset is not None else None
        groups.setdefault(side, []).append((frame, item))
    series: list[list[tuple[float, ...] | None]] = []
    for vals in groups.values():
        seq = [v for _, v in sorted(vals, key=lambda item: item[0])]
        if any(v is not None for v in seq):
            series.append(seq)
    return series


def contact_anchor_velocity(extra_rows: list[dict[str, str]]) -> list[float]:
    vals: list[float] = []
    for seq in contact_anchor_series(extra_rows):
        vals.extend(velocity_norm(seq))
    return vals


def contact_anchor_accel(extra_rows: list[dict[str, str]]) -> list[float]:
    vals: list[float] = []
    for seq in contact_anchor_series(extra_rows):
        vals.extend(second_diff_norm(seq))
    return vals


def direct_chair_contact_metric(case: str, variant: str) -> tuple[float | None, float | None]:
    if case != "chair":
        return None, None
    profile = load_case_profile(case)
    if variant == "generic":
        path = profile.result_dir / "stage4_pairprop_contact_refine" / "pairprop_quality_metrics.json"
        if path.exists():
            stats = json.loads(path.read_text()).get("candidate", {}).get("contact_gap", {}).get("all", {})
            return stats.get("median"), stats.get("p90")
    else:
        path = profile.sample_dir / "results" / "mainline_0425" / "smooth_metrics.json"
        if path.exists():
            stats = json.loads(path.read_text()).get("contact_point_stats", {})
            return stats.get("median"), stats.get("p90")
    return None, None


def opengaus_mask_proxy(case: str, variant: str) -> tuple[float | None, float | None, float | None]:
    profile = load_case_profile(case)
    if variant == "generic":
        obs_path: Path | None = profile.result_dir / "object_observations.csv"
    else:
        rel = profile.baseline.get("object_observations_csv") or profile.baseline.get("semantic_observations_csv") or ""
        obs_path = repo_path(rel) if rel else None
    if obs_path is None or not obs_path.exists():
        return None, None, None
    obs = read_csv(obs_path)
    centers: list[tuple[float, float] | None] = []
    areas: list[float] = []
    widths: list[float] = []
    for row in obs:
        cx = f(row, "mask_center_x", "center_x", "top_rail_center_u", "seat_center_u")
        cy = f(row, "mask_center_y", "center_y", "top_rail_center_v", "seat_center_v")
        centers.append((cx, cy) if cx is not None and cy is not None else None)
        area = f(row, "mask_area_px")
        if area is not None:
            areas.append(area)
        width = f(row, "bbox_w_px")
        if width is None:
            x1, x2 = f(row, "mask_bbox_x1", "bbox_x1"), f(row, "mask_bbox_x2", "bbox_x2")
            if x1 is not None and x2 is not None:
                width = abs(x2 - x1)
        if width is not None:
            widths.append(width)
    center_vel = velocity_norm(centers)
    area_mean = mean(areas)
    width_mean = mean(widths)
    area_cv = statistics.pstdev(areas) / area_mean if len(areas) > 2 and area_mean else None
    width_cv = statistics.pstdev(widths) / width_mean if len(widths) > 2 and width_mean else None
    return median(center_vel), area_cv, width_cv


def load_variant(case: str, variant: str) -> tuple[Path | None, list[dict[str, str]], list[dict[str, str]]]:
    profile = load_case_profile(case)
    if variant == "generic":
        pose_path = profile.result_dir / "object_pose.csv"
        contact_candidates = [
            profile.result_dir / "object_contact_points.csv",
            profile.result_dir / "contact_candidates.csv",
        ]
    else:
        pose_rel = profile.baseline.get("final_pose_csv", "")
        pose_path = repo_path(pose_rel) if pose_rel else None
        contact_candidates = [
            repo_path(profile.baseline.get("final_contact_csv", "")) if profile.baseline.get("final_contact_csv") else None,
            repo_path(profile.baseline.get("contact_state_csv", "")) if profile.baseline.get("contact_state_csv") else None,
            repo_path(profile.baseline.get("grasp_anchor_csv", "")) if profile.baseline.get("grasp_anchor_csv") else None,
        ]
    if pose_path is None or not pose_path.exists():
        return pose_path, [], []
    rows = read_csv(pose_path)
    extra: list[dict[str, str]] = []
    for cpath in contact_candidates:
        if cpath and cpath.exists():
            extra.extend(read_csv(cpath))
    return pose_path, rows, extra


def analyze_variant(case: str, variant: str) -> dict[str, Any]:
    path, rows, extra = load_variant(case, variant)
    trans = [vec_translation(r) for r in rows]
    rot = [vec_rotation(r) for r in rows]
    vel = velocity_norm(trans)
    acc = second_diff_norm(trans)
    rvel = velocity_norm(rot)
    racc = second_diff_norm(rot)
    visual = visual_proxy(rows)
    contact = contact_proxy(rows, extra)
    direct_contact_median, direct_contact_p90 = direct_chair_contact_metric(case, variant)
    if direct_contact_median is not None:
        contact.append(float(direct_contact_median))
    depth = depth_order_proxy(rows, extra)
    support = support_static_proxy(rows)
    anchor_vel = contact_anchor_velocity(extra)
    anchor_acc = contact_anchor_accel(extra)
    mask_center_vel, mask_area_cv, mask_width_cv = opengaus_mask_proxy(case, variant)
    return {
        "case": case,
        "variant": variant,
        "pose_csv": str(path) if path else "",
        "frame_count": len(rows),
        "method_interdiff_mean_translation_velocity": mean(vel),
        "method_interdiff_p95_translation_accel": p95(acc),
        "method_interdiff_mean_rotation_velocity": mean(rvel),
        "method_interdiff_p95_rotation_accel": p95(racc),
        "method_interdiff_contact_frame_anchor_velocity_p95": p95(anchor_vel),
        "method_interdiff_contact_frame_anchor_accel_p95": p95(anchor_acc),
        "method_mover_median_contact_gap_proxy_m": median(contact),
        "method_mover_p95_contact_gap_proxy_m": p95(contact),
        "method_mover_direct_contact_gap_median_m": direct_contact_median,
        "method_mover_direct_contact_gap_p90_m": direct_contact_p90,
        "method_mover_median_depth_order_proxy_m": median(depth),
        "method_mover_tail_static_drift_mean_m": mean(support),
        "method_opengaus_median_visual_proxy_px": median(visual),
        "method_opengaus_p95_visual_proxy_px": p95(visual),
        "method_opengaus_mask_center_velocity_median_px": mask_center_vel,
        "method_opengaus_mask_area_cv": mask_area_cv,
        "method_opengaus_mask_width_cv": mask_width_cv,
        "available_contact_proxy_samples": len(contact),
        "available_visual_proxy_samples": len(visual),
        "source": "related_work_proxy_loss_probe_from_existing_audiohoi_csv",
    }


def ratio(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or not math.isfinite(a) or not math.isfinite(b) or abs(b) < 1e-12:
        return None
    return a / b


def add_comparison_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_case: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_case.setdefault(row["case"], {})[row["variant"]] = row
    comp: list[dict[str, Any]] = []
    keys = [k for k in rows[0].keys() if k.startswith("method_")] if rows else []
    for case, variants in by_case.items():
        g = variants.get("generic")
        b = variants.get("baseline")
        if not g or not b:
            continue
        out: dict[str, Any] = {"case": case, "variant": "generic_over_baseline_ratio"}
        for key in keys:
            out[key] = ratio(g.get(key), b.get(key))  # type: ignore[arg-type]
        comp.append(out)
    return comp


def fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if not math.isfinite(v):
            return ""
        return f"{v:.6g}"
    return str(v)


def write_markdown(rows: list[dict[str, Any]], comp: list[dict[str, Any]]) -> Path:
    md = OUT_DIR / "method_loss_probe_summary.md"
    lines = [
        "# Related Work Proxy Loss Probe",
        "",
        "This is a lightweight comparison test inspired by inspected related-work code. It does not claim to reproduce the full methods.",
        "",
        "Mapped tests:",
        "",
        "- InterDiff-style: translation/rotation velocity and acceleration as motion smoothness proxies.",
        "- InterDiff-style: contact-frame anchor velocity/acceleration when local contact anchors are available.",
        "- MOVER-style: contact gap, direct chair pairprop contact gap, depth-order gap, and tail static drift proxies.",
        "- Open3DHOI/HOI-Gaussian-style: 2D visual, mask-center, and mask-size stability proxies where available.",
        "",
        "## Variant Metrics",
        "",
        "| case | variant | frames | visual med px | contact med m | direct contact med m | contact-frame vel p95 | depth proxy med m | transl acc p95 | static drift mean m |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            "| {case} | {variant} | {frame_count} | {visual} | {contact} | {direct_contact} | {anchor_vel} | {depth} | {acc} | {static} |".format(
                case=r["case"],
                variant=r["variant"],
                frame_count=r["frame_count"],
                visual=fmt(r.get("method_opengaus_median_visual_proxy_px")),
                contact=fmt(r.get("method_mover_median_contact_gap_proxy_m")),
                direct_contact=fmt(r.get("method_mover_direct_contact_gap_median_m")),
                anchor_vel=fmt(r.get("method_interdiff_contact_frame_anchor_velocity_p95")),
                depth=fmt(r.get("method_mover_median_depth_order_proxy_m")),
                acc=fmt(r.get("method_interdiff_p95_translation_accel")),
                static=fmt(r.get("method_mover_tail_static_drift_mean_m")),
            )
        )
    lines.extend(["", "## Generic / Baseline Ratios", ""])
    lines.append("| case | visual ratio | contact ratio | direct-contact ratio | contact-frame vel ratio | depth ratio | transl-acc ratio | static-drift ratio |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in comp:
        lines.append(
            "| {case} | {visual} | {contact} | {direct_contact} | {anchor_vel} | {depth} | {acc} | {static} |".format(
                case=r["case"],
                visual=fmt(r.get("method_opengaus_median_visual_proxy_px")),
                contact=fmt(r.get("method_mover_median_contact_gap_proxy_m")),
                direct_contact=fmt(r.get("method_mover_direct_contact_gap_median_m")),
                anchor_vel=fmt(r.get("method_interdiff_contact_frame_anchor_velocity_p95")),
                depth=fmt(r.get("method_mover_median_depth_order_proxy_m")),
                acc=fmt(r.get("method_interdiff_p95_translation_accel")),
                static=fmt(r.get("method_mover_tail_static_drift_mean_m")),
            )
        )
    lines.extend(
        [
            "",
            "Interpretation rules:",
            "",
            "- A ratio near 1 means the generic pipeline matches the old solved baseline on that proxy.",
            "- A ratio below 1 is better for residual/gap/acceleration/drift terms.",
            "- Empty cells mean the related-work-style proxy is not observable from the available CSV schema.",
            "- These probes are deliberately conservative: collision/SDF and true differentiable ordinal depth require mesh/render buffers and are recorded as future full tests.",
        ]
    )
    md.write_text("\n".join(lines) + "\n")
    return md


def run(cases: list[str]) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for case in cases:
        rows.append(analyze_variant(case, "generic"))
        rows.append(analyze_variant(case, "baseline"))
    comp = add_comparison_rows(rows)
    fields = [
        "case",
        "variant",
        "pose_csv",
        "frame_count",
        "method_interdiff_mean_translation_velocity",
        "method_interdiff_p95_translation_accel",
        "method_interdiff_mean_rotation_velocity",
        "method_interdiff_p95_rotation_accel",
        "method_interdiff_contact_frame_anchor_velocity_p95",
        "method_interdiff_contact_frame_anchor_accel_p95",
        "method_mover_median_contact_gap_proxy_m",
        "method_mover_p95_contact_gap_proxy_m",
        "method_mover_direct_contact_gap_median_m",
        "method_mover_direct_contact_gap_p90_m",
        "method_mover_median_depth_order_proxy_m",
        "method_mover_tail_static_drift_mean_m",
        "method_opengaus_median_visual_proxy_px",
        "method_opengaus_p95_visual_proxy_px",
        "method_opengaus_mask_center_velocity_median_px",
        "method_opengaus_mask_area_cv",
        "method_opengaus_mask_width_cv",
        "available_contact_proxy_samples",
        "available_visual_proxy_samples",
        "source",
    ]
    write_csv(OUT_DIR / "method_loss_probe_metrics.csv", rows, fields)
    write_csv(OUT_DIR / "method_loss_probe_ratios.csv", comp)
    write_json(OUT_DIR / "method_loss_probe_metrics.json", {"variants": rows, "ratios": comp})
    md = write_markdown(rows, comp)
    return {
        "cases": cases,
        "variant_rows": len(rows),
        "ratio_rows": len(comp),
        "metrics_csv": str(OUT_DIR / "method_loss_probe_metrics.csv"),
        "ratios_csv": str(OUT_DIR / "method_loss_probe_ratios.csv"),
        "summary_md": str(md),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run related-work-inspired proxy loss probes on AudioHOI solved outputs.")
    ap.add_argument("--case", default="all", help="case name or all")
    args = ap.parse_args()
    cases = available_cases() if args.case == "all" else [args.case]
    print(json.dumps(run(cases), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
