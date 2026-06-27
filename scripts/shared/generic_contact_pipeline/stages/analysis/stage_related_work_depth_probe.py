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
else:
    REPO = Path(__file__).resolve().parents[5]

from scripts.shared.generic_contact_pipeline.core.base.config import available_cases, load_case_profile  # noqa: E402
from scripts.shared.generic_contact_pipeline.core.base.io import read_csv, repo_path, write_csv, write_json  # noqa: E402
from scripts.shared.generic_contact_pipeline.stages.analysis.stage_related_work_observation_probe import (  # noqa: E402
    read_png_nonzero_points_fast,
)


OUT_DIR = REPO / "docs" / "related_work_audit"


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


def load_depth(path: Path):
    import numpy as np  # type: ignore

    return np.load(path)


def sample_depth(depth, u: float | None, v: float | None, image_w: int, image_h: int, radius: int = 1) -> float | None:
    import numpy as np  # type: ignore

    if u is None or v is None:
        return None
    h, w = depth.shape[:2]
    x = int(round(u * (w - 1) / max(image_w - 1, 1)))
    y = int(round(v * (h - 1) / max(image_h - 1, 1)))
    x1, x2 = max(0, x - radius), min(w, x + radius + 1)
    y1, y2 = max(0, y - radius), min(h, y + radius + 1)
    patch = depth[y1:y2, x1:x2]
    vals = patch[np.isfinite(patch)]
    vals = vals[vals > 0]
    if vals.size == 0:
        return None
    return float(np.median(vals))


def sample_mask_depth(depth, mask_path: Path, sample_stride: int = 12) -> float | None:
    import numpy as np  # type: ignore

    points = read_png_nonzero_points_fast(mask_path)
    if not points:
        return None
    h, w = depth.shape[:2]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    image_w = max(xs) + 1
    image_h = max(ys) + 1
    vals: list[float] = []
    for x_img, y_img in points[::max(1, sample_stride)]:
        x = int(round(x_img * (w - 1) / max(image_w - 1, 1)))
        y = int(round(y_img * (h - 1) / max(image_h - 1, 1)))
        val = float(depth[y, x])
        if math.isfinite(val) and val > 0:
            vals.append(val)
    return float(np.median(vals)) if vals else None


def image_size_from_mask(mask_path: Path) -> tuple[int, int] | None:
    pts = read_png_nonzero_points_fast(mask_path)
    if not pts:
        return None
    return max(p[0] for p in pts) + 1, max(p[1] for p in pts) + 1


def frame_file(root: Path, frame: int, suffix: str) -> Path | None:
    for idx in [frame, frame - 1]:
        for digits in [5, 6]:
            p = root / f"{idx:0{digits}d}{suffix}"
            if p.exists():
                return p
    return None


def choose_obs_path(case: str, variant: str) -> Path | None:
    profile = load_case_profile(case)
    if variant == "generic":
        p = profile.result_dir / "object_observations.csv"
        return p if p.exists() else None
    for key in ["object_observations_csv", "semantic_observations_csv", "object_proxy_observations_csv"]:
        rel = profile.baseline.get(key)
        if rel:
            p = repo_path(rel)
            if p.exists():
                return p
    return None


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
    for key in ["final_contact_csv", "contact_state_csv", "grasp_anchor_csv", "contact_candidates_csv"]:
        rel = profile.baseline.get(key)
        if rel:
            p = repo_path(rel)
            if p.exists():
                return p
    return None


def merge_by_frame(*tables: list[dict[str, str]]) -> list[dict[str, str]]:
    by_frame: dict[int, dict[str, str]] = {}
    for table in tables:
        for idx, row in enumerate(table):
            frame = int(ff(row.get("frame")) or idx + 1)
            dst = by_frame.setdefault(frame, {"frame": str(frame)})
            for key, val in row.items():
                if val not in ("", None):
                    dst.setdefault(key, val)
    return [by_frame[k] for k in sorted(by_frame)]


def pose_depth(row: dict[str, str]) -> float | None:
    return first(row, ["tz", "z", "object_ref_depth_m", "da3_depth_smooth", "object_depth_smooth"])


def center_uv(row: dict[str, str]) -> tuple[float | None, float | None]:
    return (
        first(row, ["u_proj", "u_obs", "ref_u_fit", "ref_u_smooth", "ref_u", "center_x", "mask_center_x", "top_rail_center_u", "seat_center_u", "contact_point_2d_u"]),
        first(row, ["v_proj", "v_obs", "ref_v_fit", "ref_v_smooth", "ref_v", "center_y", "mask_center_y", "top_rail_center_v", "seat_center_v", "contact_point_2d_v"]),
    )


def support_uv(row: dict[str, str]) -> tuple[float | None, float | None]:
    return (
        first(row, ["support_u", "floor_support_proxy_center_u", "front_right_foot_u", "front_left_foot_u"]),
        first(row, ["support_v", "floor_support_proxy_center_v", "front_right_foot_v", "front_left_foot_v"]),
    )


def contact_uv(row: dict[str, str]) -> tuple[float | None, float | None]:
    return (
        first(row, ["contact_point_2d_u", "contact_u", "active_part_u"]),
        first(row, ["contact_point_2d_v", "contact_v", "active_part_v"]),
    )


def active_contact(row: dict[str, str]) -> bool:
    raw = str(row.get("contact_active", row.get("contact_frame", row.get("human_contact_state", "0")))).strip().lower()
    return raw in {"1", "true", "yes", "active"}


def analyze_variant(case: str, variant: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    profile = load_case_profile(case)
    obs_path = choose_obs_path(case, variant)
    pose_path = choose_pose_path(case, variant)
    contact_path = choose_contact_path(case, variant)
    rows = merge_by_frame(
        read_csv(obs_path) if obs_path else [],
        read_csv(pose_path) if pose_path else [],
        read_csv(contact_path) if contact_path else [],
    )
    depth_dir = profile.sample_dir / "results" / "da3" / "scene_depth"
    mask_dir = profile.sample_dir / "results" / "segmentation" / "masks"
    floor_prior_path = profile.sample_dir / "results" / "da3" / "priors" / "floor_prior.csv"
    floor_by_frame = {int(ff(r.get("frame")) or i + 1): r for i, r in enumerate(read_csv(floor_prior_path))} if floor_prior_path.exists() else {}

    per_frame: list[dict[str, Any]] = []
    vals: dict[str, list[float]] = {}
    for idx, row in enumerate(rows):
        frame = int(ff(row.get("frame")) or idx + 1)
        depth_path = frame_file(depth_dir, frame, ".npy")
        mask_path = frame_file(mask_dir, frame, "_mask.png")
        if not depth_path or not mask_path:
            continue
        depth = load_depth(depth_path)
        image_size = image_size_from_mask(mask_path)
        if image_size is None:
            continue
        image_w, image_h = image_size
        cu, cv = center_uv(row)
        su, sv = support_uv(row)
        tu, tv = contact_uv(row)
        center_d = sample_depth(depth, cu, cv, image_w, image_h)
        support_d = sample_depth(depth, su, sv, image_w, image_h)
        contact_d = sample_depth(depth, tu, tv, image_w, image_h) if active_contact(row) else None
        mask_d = sample_mask_depth(depth, mask_path)
        pred_z = pose_depth(row)
        floor_d = ff(floor_by_frame.get(frame, {}).get("floor_depth_proxy"))

        metrics: dict[str, float] = {}
        if pred_z is not None and center_d is not None:
            metrics["opengaus_pose_center_da3_gap_m"] = abs(pred_z - center_d)
        if pred_z is not None and mask_d is not None:
            metrics["opengaus_pose_mask_da3_gap_m"] = abs(pred_z - mask_d)
        if center_d is not None and mask_d is not None:
            metrics["opengaus_center_mask_depth_gap_m"] = abs(center_d - mask_d)
        if contact_d is not None and center_d is not None:
            metrics["mover_contact_center_depth_gap_m"] = abs(contact_d - center_d)
        if support_d is not None and floor_d is not None:
            metrics["mover_support_floor_depth_gap_m"] = abs(support_d - floor_d)
        if support_d is not None and center_d is not None:
            # Positive means support/floor sample is behind the visible object center.
            metrics["mover_center_in_front_of_support_margin_m"] = support_d - center_d
            metrics["mover_center_support_order_violation_m"] = max(0.0, center_d - support_d)

        out = {
            "case": case,
            "variant": variant,
            "frame": frame,
            "depth_path": str(depth_path),
            "center_depth_da3": center_d,
            "mask_median_depth_da3": mask_d,
            "support_depth_da3": support_d,
            "contact_depth_da3": contact_d,
            "pose_depth_proxy": pred_z,
            "floor_depth_proxy": floor_d,
            "contact_active": int(active_contact(row)),
        }
        out.update(metrics)
        per_frame.append(out)
        for key, value in metrics.items():
            vals.setdefault(key, []).append(value)

    summary: dict[str, Any] = {
        "case": case,
        "variant": variant,
        "frames": len(rows),
        "depth_frames_used": len(per_frame),
        "observation_csv": str(obs_path) if obs_path else "",
        "pose_csv": str(pose_path) if pose_path else "",
        "contact_csv": str(contact_path) if contact_path else "",
    }
    for key, arr in vals.items():
        summary[f"{key}_median"] = median(arr)
        summary[f"{key}_p95"] = p95(arr)
        summary[f"{key}_mean"] = mean(arr)
        summary[f"{key}_samples"] = len(arr)
    return summary, per_frame


def write_markdown(summaries: list[dict[str, Any]], ratios: list[dict[str, Any]]) -> Path:
    path = OUT_DIR / "related_work_depth_probe_summary.md"
    lines = [
        "# Related Work DA3 Depth Probe",
        "",
        "Generated by `scripts/shared/generic_contact_pipeline/stages/stage_related_work_depth_probe.py`.",
        "",
        "This is a DA3 scene-depth proxy for MOVER/Open3DHOI-style depth terms.",
        "It samples the saved DA3 depth maps at object center, object mask, support point, and contact point where available.",
        "",
        "Important caveat: AudioHOI final pose depth may live in a shifted GVHMR/camera frame, so pose-vs-DA3 absolute gaps are comparison proxies, not calibrated metric error claims.",
        "",
        "## Variant Metrics",
        "",
        "| case | variant | depth frames | pose-center gap med m | pose-mask gap med m | center-mask gap med m | contact-center gap med m | support-floor gap med m | support order violation med m |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        lines.append(
            "| {case} | {variant} | {frames} | {pose_center} | {pose_mask} | {center_mask} | {contact_center} | {support_floor} | {order} |".format(
                case=s["case"],
                variant=s["variant"],
                frames=s["depth_frames_used"],
                pose_center=fmt(s.get("opengaus_pose_center_da3_gap_m_median")),
                pose_mask=fmt(s.get("opengaus_pose_mask_da3_gap_m_median")),
                center_mask=fmt(s.get("opengaus_center_mask_depth_gap_m_median")),
                contact_center=fmt(s.get("mover_contact_center_depth_gap_m_median")),
                support_floor=fmt(s.get("mover_support_floor_depth_gap_m_median")),
                order=fmt(s.get("mover_center_support_order_violation_m_median")),
            )
        )
    lines.extend(["", "## Generic / Baseline Ratios", ""])
    lines.append("| case | pose-center | pose-mask | center-mask | contact-center | support-floor | order violation |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in ratios:
        lines.append(
            "| {case} | {pose_center} | {pose_mask} | {center_mask} | {contact_center} | {support_floor} | {order} |".format(
                case=r["case"],
                pose_center=fmt(r.get("opengaus_pose_center_da3_gap_m_median")),
                pose_mask=fmt(r.get("opengaus_pose_mask_da3_gap_m_median")),
                center_mask=fmt(r.get("opengaus_center_mask_depth_gap_m_median")),
                contact_center=fmt(r.get("mover_contact_center_depth_gap_m_median")),
                support_floor=fmt(r.get("mover_support_floor_depth_gap_m_median")),
                order=fmt(r.get("mover_center_support_order_violation_m_median")),
            )
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- Lower is better for absolute gap/violation terms.",
            "- `center-mask gap` is a local DA3 consistency check inside the object mask.",
            "- `contact-center gap` is a MOVER-style contact depth compatibility proxy, not a contact classifier.",
            "- `support order violation` is zero when the support sample is not in front of the object center.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")
    return path


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}" if math.isfinite(value) else ""
    return str(value)


def run(cases: list[str]) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = []
    for case in cases:
        for variant in ["generic", "baseline"]:
            s, f = analyze_variant(case, variant)
            summaries.append(s)
            frames.extend(f)
    by_case: dict[str, dict[str, dict[str, Any]]] = {}
    for s in summaries:
        by_case.setdefault(s["case"], {})[s["variant"]] = s
    keys = sorted(k for s in summaries for k in s if k.endswith("_median"))
    ratios: list[dict[str, Any]] = []
    for case, variants in by_case.items():
        g = variants.get("generic")
        b = variants.get("baseline")
        if not g or not b:
            continue
        row: dict[str, Any] = {"case": case, "variant": "generic_over_baseline_ratio"}
        for key in keys:
            row[key] = ratio(g.get(key), b.get(key))
        ratios.append(row)

    write_csv(OUT_DIR / "related_work_depth_probe_metrics.csv", summaries)
    write_csv(OUT_DIR / "related_work_depth_probe_ratios.csv", ratios)
    write_csv(OUT_DIR / "related_work_depth_probe_per_frame.csv", frames)
    write_json(OUT_DIR / "related_work_depth_probe_metrics.json", {"summaries": summaries, "ratios": ratios})
    md = write_markdown(summaries, ratios)
    return {
        "cases": cases,
        "summary_rows": len(summaries),
        "per_frame_rows": len(frames),
        "metrics_csv": str(OUT_DIR / "related_work_depth_probe_metrics.csv"),
        "ratios_csv": str(OUT_DIR / "related_work_depth_probe_ratios.csv"),
        "summary_md": str(md),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run DA3 scene-depth probes inspired by MOVER/Open3DHOI ordinal-depth losses.")
    ap.add_argument("--case", default="all", help="case name or all")
    args = ap.parse_args()
    cases = available_cases() if args.case == "all" else [args.case]
    print(json.dumps(run(cases), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
