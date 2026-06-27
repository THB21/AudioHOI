from __future__ import annotations

import math
from pathlib import Path

from .config import CaseProfile
from .io import read_csv, write_csv, write_json
from .schema import stage_paths


FIELDS = [
    "frame",
    "time",
    "stage_label",
    "r_center_u",
    "r_center_v",
    "r_depth",
    "r_support",
    "r_contact",
    "r_smooth_xy",
    "r_smooth_z",
    "r_reg",
    "E_visual",
    "E_depth",
    "E_support",
    "E_contact",
    "E_smooth",
    "E_reg",
    "E_total",
    "contact_active",
    "audio_contact_frame",
]


DEFAULT_WEIGHTS = {
    "center": 18.0,
    "depth": 0.25,
    "support": 10.0,
    "contact": 10.0,
    "smooth_xy": 0.08,
    "smooth_z": 0.20,
    "reg": 0.05,
}


def f(row: dict[str, str] | None, key: str, default: float | None = None) -> float | None:
    if row is None:
        return default
    try:
        value = row.get(key)
        if value in {"", None}:
            return default
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return default


def frame(row: dict[str, str]) -> int:
    try:
        return int(float(row.get("frame", "0") or 0))
    except Exception:
        return 0


def by_frame(rows: list[dict[str, str]]) -> dict[int, dict[str, str]]:
    return {frame(row): row for row in rows}


def sq(value: float | None) -> float:
    if value is None or not math.isfinite(value):
        return 0.0
    return value * value


def residual_rows(
    profile: CaseProfile,
    pose_csv: Path,
    *,
    stage_label: str,
    init_csv: Path | None = None,
    weights: dict[str, float] | None = None,
) -> list[dict[str, object]]:
    if profile.case_name not in {"basketball", "football"}:
        return []
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)
    paths = stage_paths(profile)
    pose_rows = read_csv(pose_csv)
    obs_by_frame = by_frame(read_csv(paths["object_observations"]))
    contact_by_frame = by_frame(read_csv(paths["contact_candidates"])) if paths["contact_candidates"].exists() else {}
    init_by_frame = by_frame(read_csv(init_csv)) if init_csv and init_csv.exists() else {}
    out: list[dict[str, object]] = []
    prev: dict[str, str] | None = None
    for row in pose_rows:
        fr = frame(row)
        obs = obs_by_frame.get(fr, {})
        contact = contact_by_frame.get(fr, {})
        init = init_by_frame.get(fr, {})
        u_proj = f(row, "u_proj", f(row, "u_obs", f(obs, "ref_u", 0.0))) or 0.0
        v_proj = f(row, "v_proj", f(row, "v_obs", f(obs, "ref_v", 0.0))) or 0.0
        u_obs = f(row, "u_obs", f(obs, "ref_u", u_proj)) or u_proj
        v_obs = f(row, "v_obs", f(obs, "ref_v", v_proj)) or v_proj
        sigma_uv = max(f(obs, "proxy_sigma_px", 5.0) or 5.0, 1.0)
        proxy_conf = max(f(obs, "proxy_conf", 1.0) or 1.0, 0.05)
        cscale = math.sqrt(w["center"] * proxy_conf)
        r_center_u = cscale * ((u_proj - u_obs) / sigma_uv)
        r_center_v = cscale * ((v_proj - v_obs) / sigma_uv)

        tz = f(row, "tz", 0.0) or 0.0
        depth_prior = f(row, "depth_prior_m", f(obs, "object_depth_smooth", f(obs, "object_ref_depth_m")))
        r_depth = None if depth_prior is None else math.sqrt(w["depth"]) * (tz - depth_prior) / 0.30

        pred_support_v = f(row, "bottom_proj_v")
        support_v = f(row, "floor_v", f(obs, "support_v"))
        support_conf = f(obs, "support_conf", f(row, "support_confidence", 0.0)) or 0.0
        support_gap = abs(f(obs, "support_gap_px", 0.0) or 0.0)
        sigma_support = max(f(obs, "support_sigma_px", 8.0) or 8.0, 1.0)
        support_valid = pred_support_v is not None and support_v is not None and support_conf > 0.15 and support_gap < 18.0
        r_support = None
        if support_valid:
            r_support = math.sqrt(w["support"] * min(1.0, support_conf)) * ((pred_support_v - support_v) / sigma_support)  # type: ignore[operator]

        contact_active = str(contact.get("contact_active", row.get("contact_frame", "0")))
        anchor_score = f(contact, "anchor_score", f(obs, "contact_conf", 0.0)) or 0.0
        offset = f(contact, "contact_depth_offset_m", f(row, "contact_depth_offset_m"))
        part_z = f(obs, "active_part_z", f(row, "active_part_z"))
        r_contact = None
        if contact_active in {"1", "1.0"} and offset is not None and part_z is not None and anchor_score > 0.05:
            # Mirrors the original radius-free contact z residual:
            # contact_w * anchor_conf^2 * ((z + offset) - active_part_z).
            r_contact = w["contact"] * (min(1.0, anchor_score) ** 2) * ((tz + max(offset, 0.0)) - part_z)

        r_smooth_xy = None
        r_smooth_z = None
        if prev is not None:
            dx = (f(row, "tx", 0.0) or 0.0) - (f(prev, "tx", 0.0) or 0.0)
            dy = (f(row, "ty", 0.0) or 0.0) - (f(prev, "ty", 0.0) or 0.0)
            dz = (f(row, "tz", 0.0) or 0.0) - (f(prev, "tz", 0.0) or 0.0)
            r_smooth_xy = math.sqrt(w["smooth_xy"]) * math.sqrt(dx * dx + dy * dy)
            r_smooth_z = math.sqrt(w["smooth_z"]) * dz

        r_reg = None
        if init:
            dx = (f(row, "tx", 0.0) or 0.0) - (f(init, "tx", 0.0) or 0.0)
            dy = (f(row, "ty", 0.0) or 0.0) - (f(init, "ty", 0.0) or 0.0)
            dz = (f(row, "tz", 0.0) or 0.0) - (f(init, "tz", 0.0) or 0.0)
            r_reg = math.sqrt(w["reg"]) * math.sqrt(dx * dx + dy * dy + dz * dz)

        e_visual = sq(r_center_u) + sq(r_center_v)
        e_depth = sq(r_depth)
        e_support = sq(r_support)
        e_contact = sq(r_contact)
        e_smooth = sq(r_smooth_xy) + sq(r_smooth_z)
        e_reg = sq(r_reg)
        total = e_visual + e_depth + e_support + e_contact + e_smooth + e_reg
        out.append(
            {
                "frame": fr,
                "time": row.get("time", ""),
                "stage_label": stage_label,
                "r_center_u": "" if r_center_u is None else f"{r_center_u:.9f}",
                "r_center_v": "" if r_center_v is None else f"{r_center_v:.9f}",
                "r_depth": "" if r_depth is None else f"{r_depth:.9f}",
                "r_support": "" if r_support is None else f"{r_support:.9f}",
                "r_contact": "" if r_contact is None else f"{r_contact:.9f}",
                "r_smooth_xy": "" if r_smooth_xy is None else f"{r_smooth_xy:.9f}",
                "r_smooth_z": "" if r_smooth_z is None else f"{r_smooth_z:.9f}",
                "r_reg": "" if r_reg is None else f"{r_reg:.9f}",
                "E_visual": f"{e_visual:.9f}",
                "E_depth": f"{e_depth:.9f}",
                "E_support": f"{e_support:.9f}",
                "E_contact": f"{e_contact:.9f}",
                "E_smooth": f"{e_smooth:.9f}",
                "E_reg": f"{e_reg:.9f}",
                "E_total": f"{total:.9f}",
                "contact_active": contact_active,
                "audio_contact_frame": row.get("audio_contact_frame", ""),
            }
        )
        prev = row
    return out


def write_ball_residual_report(
    profile: CaseProfile,
    pose_csv: Path,
    *,
    stage_label: str,
    out_name: str,
    init_csv: Path | None = None,
) -> dict[str, object]:
    rows = residual_rows(profile, pose_csv, stage_label=stage_label, init_csv=init_csv)
    if not rows:
        return {"rows": 0, "path": "", "stage_label": stage_label}
    out_dir = profile.result_dir / "loss_analysis"
    out_path = write_csv(out_dir / out_name, rows, FIELDS)
    totals = {key: sum(float(r[key]) for r in rows if r.get(key) not in {"", None}) for key in ["E_total", "E_visual", "E_depth", "E_support", "E_contact", "E_smooth", "E_reg"]}
    summary = {
        "stage_label": stage_label,
        "path": str(out_path),
        "rows": len(rows),
        "terms": totals,
        "note": "Optimizer-style normalized ball residuals mirroring the radius-free center/support/contact/smooth/reg construction; report only, does not alter pose.",
    }
    write_json(out_dir / f"{Path(out_name).stem}_summary.json", summary)
    return summary
