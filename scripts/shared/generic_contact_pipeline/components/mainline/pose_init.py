from __future__ import annotations

import importlib
import math
from collections import defaultdict

from scipy.spatial.transform import Rotation

from ...core.base.config import CaseProfile
from ...core.base.camera import backproject_uvz
from ...core.base.io import read_csv, write_csv, write_json
from ...core.base.schema import stage_paths


SE3_FIELDS = ["frame", "time", "tx", "ty", "tz", "qw", "qx", "qy", "qz"]


def _legacy_pose_adapter(profile: CaseProfile) -> dict[str, object]:
    name = profile.component("pose_model")
    mod = importlib.import_module(f"scripts.shared.generic_contact_pipeline.components.pose.models.{name}")
    result = mod.build(profile)
    result = dict(result)
    result["adapter_component"] = name
    return result


def _number(value: object, default: float) -> float:
    try:
        if value in {"", None}:
            return default
        x = float(value)  # type: ignore[arg-type]
        return x if math.isfinite(x) else default
    except Exception:
        return default


def _normalize_quat(row: dict[str, str]) -> tuple[float, float, float, float]:
    has_quat = any(row.get(key) not in {"", None} for key in ("qw", "qx", "qy", "qz"))
    if not has_quat and any(row.get(key) not in {"", None} for key in ("yaw", "pitch", "roll")):
        yaw = _number(row.get("yaw"), 0.0)
        pitch = _number(row.get("pitch"), 0.0)
        roll = _number(row.get("roll"), 0.0)
        qx, qy, qz, qw = Rotation.from_euler("zyx", [yaw, pitch, roll]).as_quat()
        return float(qw), float(qx), float(qy), float(qz)
    qw = _number(row.get("qw"), 1.0)
    qx = _number(row.get("qx"), 0.0)
    qy = _number(row.get("qy"), 0.0)
    qz = _number(row.get("qz"), 0.0)
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm < 1e-8:
        return 1.0, 0.0, 0.0, 0.0
    return qw / norm, qx / norm, qy / norm, qz / norm


def _ensure_se3_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in rows:
        qw, qx, qy, qz = _normalize_quat(row)
        new = dict(row)
        new.update(
            {
                "frame": row.get("frame", ""),
                "time": row.get("time", ""),
                "tx": f"{_number(row.get('tx'), _number(row.get('x'), 0.0)):.6f}",
                "ty": f"{_number(row.get('ty'), _number(row.get('y'), 0.0)):.6f}",
                "tz": f"{max(0.2, _number(row.get('tz'), _number(row.get('z'), 3.0))):.6f}",
                "qw": f"{qw:.6f}",
                "qx": f"{qx:.6f}",
                "qy": f"{qy:.6f}",
                "qz": f"{qz:.6f}",
                "se3_schema": "1",
                "source": (row.get("source", "") + "+generic_se3_pose_init").strip("+"),
            }
        )
        out.append(new)
    return out


def _anchor_depth_by_frame(rows: list[dict[str, str]]) -> dict[int, float]:
    grouped: defaultdict[int, list[float]] = defaultdict(list)
    for row in rows:
        try:
            if row.get("pose_anchor_allowed") not in {"1", "1.0", "true", "True"}:
                continue
            if row.get("contact_observed", "1") in {"0", "0.0", "false", "False"}:
                continue
            fr = int(float(row.get("frame", "0") or 0))
            z = _number(row.get("anchor_z_m", row.get("palm_z", "")), math.nan)
            if math.isfinite(z) and z > 0.0:
                grouped[fr].append(z)
        except Exception:
            continue
    return {fr: sorted(vals)[len(vals) // 2] for fr, vals in grouped.items() if vals}


def _line_solution(row: dict[str, str]) -> tuple[float, float, float, float] | None:
    try:
        x1 = _number(row.get("visible_x1", row.get("physical_x1")), math.nan)
        y1 = _number(row.get("visible_y1", row.get("physical_y1")), math.nan)
        x2 = _number(row.get("visible_x2", row.get("physical_x2")), math.nan)
        y2 = _number(row.get("visible_y2", row.get("physical_y2")), math.nan)
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
            return None
        length = _number(row.get("visible_len_px"), math.hypot(x2 - x1, y2 - y1))
        if not math.isfinite(length) or length < 8.0:
            return None
        angle = _number(row.get("visible_angle_rad"), math.atan2(y2 - y1, x2 - x1))
        return 0.5 * (x1 + x2), 0.5 * (y1 + y2), length, angle
    except Exception:
        return None


def _apply_geometry_pose_prior(profile: CaseProfile, rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], dict[str, object]]:
    paths = stage_paths(profile)
    if not profile.data.get("line_object") or not paths["line_correspondence"].exists() or not rows:
        return rows, {"enabled": False, "reason": "no line geometry prior inputs"}
    length_m = float(profile.data.get("line_object", {}).get("length_m", 1.0))
    if length_m <= 0.0:
        return rows, {"enabled": False, "reason": "invalid line_object.length_m"}
    corr_by_frame = {}
    for row in read_csv(paths["line_correspondence"]):
        try:
            corr_by_frame[int(float(row.get("frame", "0") or 0))] = row
        except Exception:
            continue
    anchor_depth = _anchor_depth_by_frame(read_csv(paths["anchor_state"]) if paths["anchor_state"].exists() else [])
    out: list[dict[str, object]] = []
    updated = 0
    skipped_untrusted = 0
    for row in rows:
        rec = dict(row)
        try:
            fr = int(float(str(row.get("frame", "0") or 0)))
        except Exception:
            out.append(rec)
            continue
        corr = corr_by_frame.get(fr)
        if not corr:
            out.append(rec)
            continue
        trusted = corr.get("line_observation_trusted", "1") == "1" and corr.get("occlusion_state", "visible") == "visible"
        solution = _line_solution(corr)
        if solution is None:
            out.append(rec)
            continue
        if not trusted:
            skipped_untrusted += 1
            out.append(rec)
            continue
        cx, cy, pixel_len, angle = solution
        z_from_length = profile.camera["fx"] * length_m / max(pixel_len, 1.0)
        z_anchor = anchor_depth.get(fr)
        if z_anchor is not None:
            z_anchor = max(z_from_length - 0.35, min(z_from_length + 0.35, z_anchor))
            z = 0.86 * z_from_length + 0.14 * z_anchor
        else:
            z = z_from_length
        z = max(0.5, min(8.0, z))
        tx, ty, tz = backproject_uvz(cx, cy, z, profile.camera)
        qw, qx, qy, qz = _normalize_quat({"yaw": str(angle), "pitch": "0", "roll": "0"})
        rec.update(
            {
                "tx": f"{tx:.6f}",
                "ty": f"{ty:.6f}",
                "tz": f"{tz:.6f}",
                "qw": f"{qw:.6f}",
                "qx": f"{qx:.6f}",
                "qy": f"{qy:.6f}",
                "qz": f"{qz:.6f}",
                "u_proj": f"{cx:.3f}",
                "v_proj": f"{cy:.3f}",
                "object_pixel_len": f"{pixel_len:.3f}",
                "z_from_geometry_len_m": f"{z_from_length:.6f}",
                "generic_geometry_pose_prior": "1",
                "geometry_pose_prior_source": "line_correspondence_schema",
                "source": (str(row.get("source", "")) + "+generic_geometry_pose_prior").strip("+"),
            }
        )
        updated += 1
        out.append(rec)
    return out, {
        "enabled": True,
        "source": str(paths["line_correspondence"]),
        "rows": updated,
        "skipped_untrusted_rows": skipped_untrusted,
        "policy": "Stage3 generic geometry pose prior from correspondence schema; Stage4 remains object-agnostic SE3 optimization",
    }


def _apply_line_contact_pose_prior(profile: CaseProfile, rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], dict[str, object]]:
    paths = stage_paths(profile)
    if not profile.data.get("line_object") or not rows:
        return rows, {"enabled": False, "reason": "not a line object"}
    if not paths["contact_candidates"].exists() or not paths["anchor_state"].exists():
        return rows, {"enabled": False, "reason": "missing contact_candidates or anchor_state"}
    from ..refinement.policies import line_contact_lock as line_adapter

    contacts = read_csv(paths["contact_candidates"])
    anchor_rows = read_csv(paths["anchor_state"])
    contacts_by_frame = line_adapter._by_frame_side(contacts, anchor_rows)
    visible_by_frame = line_adapter._line_tracks(profile)
    solved_by_frame: dict[int, tuple[float, float, float, float, float, float]] = {}
    blend_by_frame: dict[int, dict[str, float | str]] = {}
    solved = 0
    skipped = 0
    for row in rows:
        try:
            fr = int(float(str(row.get("frame", "0") or 0)))
        except Exception:
            continue
        pair = contacts_by_frame.get(fr, {})
        solution = (
            line_adapter._solve_axis_from_contacts(pair.get("left", {}), pair.get("right", {}))
            if {"left", "right"} <= set(pair)
            else None
        )
        visible = line_adapter._solve_axis_from_visible(visible_by_frame[fr]) if fr in visible_by_frame else None
        if solution is None and visible is not None:
            solution = visible
            blend_by_frame[fr] = {
                "blend_visible_weight": 1.0,
                "visible_len_px": visible[2],
                "contact_len_px": -1.0,
                "angle_gap_deg": -1.0,
                "center_gap_px": -1.0,
                "mean_palm_to_visible_line_px": -1.0,
                "blend_reason": "visible_line_fallback_no_contact_solution",
            }
        elif solution is not None:
            blended, blend_info = line_adapter._blend_solutions(solution, visible, pair)
            solution = blended
            blend_by_frame[fr] = blend_info
            if visible is None and fr in visible_by_frame:
                # A very short visible segment exists but is not long enough to
                # define the full object scale. In that case a contact-only
                # solve often flips the rod away from the image evidence; let
                # the second pass interpolate between neighboring visual poses.
                solution = None
                blend_by_frame.pop(fr, None)
        if solution is None:
            skipped += 1
            continue
        solved_by_frame[fr] = solution
        solved += 1

    out: list[dict[str, object]] = []
    filled = 0
    promoted = 0
    overlay_fallback = 0
    overlay_fallback_threshold = float(profile.data.get("line_object", {}).get("overlay_first_fallback_px", 35.0))
    for row in rows:
        rec = dict(row)
        try:
            fr = int(float(str(row.get("frame", "0") or 0)))
        except Exception:
            out.append(rec)
            continue
        solution = solved_by_frame.get(fr)
        blend_info = blend_by_frame.get(fr)
        if solution is None:
            solution = line_adapter._interp_solution(fr, solved_by_frame)
            if solution is not None:
                blend_info = {"blend_visible_weight": 0.0, "blend_reason": "temporal_contact_fill"}
                filled += 1
        if solution is not None:
            pair = contacts_by_frame.get(fr, {})
            merged = line_adapter._merge_pose({k: str(v) for k, v in rec.items()}, solution, profile, blend_info, pair)
            try:
                overlay_err = float(merged.get("line_contact_se3_overlay_err_px", "0") or 0)
            except Exception:
                overlay_err = 0.0
            if overlay_err > overlay_fallback_threshold and blend_info and str(blend_info.get("blend_reason", "")).startswith("visible_line"):
                cx, cy, pixel_len, angle, _ax, _ay = solution
                length_m = float(profile.data.get("line_object", {}).get("length_m", 1.0))
                z = max(0.5, min(8.0, profile.camera["fx"] * length_m / max(pixel_len, 1.0)))
                tx, ty, tz = backproject_uvz(cx, cy, z, profile.camera)
                half = 0.5 * angle
                merged.update(
                    {
                        "tx": f"{tx:.6f}",
                        "ty": f"{ty:.6f}",
                        "tz": f"{tz:.6f}",
                        "qw": f"{math.cos(half):.6f}",
                        "qx": "0.000000",
                        "qy": "0.000000",
                        "qz": f"{math.sin(half):.6f}",
                        "u_proj": f"{cx:.3f}",
                        "v_proj": f"{cy:.3f}",
                        "u_contact_lock": f"{cx:.3f}",
                        "v_contact_lock": f"{cy:.3f}",
                        "contact_lock_pixel_len": f"{pixel_len:.3f}",
                        "contact_lock_angle_rad": f"{angle:.9f}",
                        "line_contact_se3_refined": "0",
                        "line_contact_se3_reason": "overlay_first_fallback_after_contact_refine_error",
                        "line_contact_se3_overlay_err_px": f"{overlay_err:.6f}",
                        "z_from_pixel_len_m": f"{z:.6f}",
                    }
                )
                overlay_fallback += 1
            merged["line_contact_pose_prior"] = "1"
            merged["generic_geometry_pose_prior"] = merged.get("generic_geometry_pose_prior", rec.get("generic_geometry_pose_prior", ""))
            rec = merged
            promoted += 1
        out.append(rec)
    return out, {
        "enabled": promoted > 0,
        "rows": promoted,
        "solved_frames": solved,
        "skipped_frames": skipped,
        "temporal_filled_frames": filled,
        "overlay_first_fallback_frames": overlay_fallback,
        "overlay_first_fallback_threshold_px": overlay_fallback_threshold,
        "source": "line_object_geometry_adapter_from_contact_and_correspondence",
        "policy": "promote line contact/overlay geometry into Stage3 object_pose_init.csv; Stage4 remains generic SE3 optimizer",
    }


def build(profile: CaseProfile) -> dict[str, object]:
    paths = stage_paths(profile)
    adapter = _legacy_pose_adapter(profile)
    rows = read_csv(paths["object_pose_init"]) if paths["object_pose_init"].exists() else []
    se3_rows = _ensure_se3_rows(rows)
    se3_rows, geometry_prior = _apply_geometry_pose_prior(profile, se3_rows)
    se3_rows, line_contact_prior = _apply_line_contact_pose_prior(profile, se3_rows)
    write_csv(paths["object_pose_init"], se3_rows)
    missing = [field for field in SE3_FIELDS if se3_rows and field not in se3_rows[0]]
    metrics = {
        "component": "generic_se3_pose_init",
        "case_name": profile.case_name,
        "adapter": adapter,
        "object_pose_init": str(paths["object_pose_init"]),
        "rows": len(se3_rows),
        "se3_fields": SE3_FIELDS,
        "missing_se3_fields": missing,
        "geometry_pose_prior": geometry_prior,
        "line_contact_pose_prior": line_contact_prior,
        "policy": "fixed SE3 pose schema; weak or unobserved rotations remain identity/low-weight instead of dropping quaternion fields",
    }
    write_json(paths["stage3_metrics"], metrics)
    return metrics
