from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from ....core.base.camera import backproject_uvz
from ....core.base.config import CaseProfile
from ....core.base.io import float_or_none, read_csv, write_csv, write_json
from ....core.base.schema import stage_paths


def _pose_anchor_gate(anchor_rows: list[dict[str, str]] | None) -> dict[tuple[int, str], bool]:
    gate: dict[tuple[int, str], bool] = {}
    for row in anchor_rows or []:
        try:
            fr = int(float(row.get("frame", "0") or 0))
        except Exception:
            continue
        side = row.get("human_side", "")
        if side in {"left", "right"}:
            gate[(fr, side)] = row.get("pose_anchor_allowed") == "1"
    return gate


def _anchor_by_frame_side(anchor_rows: list[dict[str, str]] | None) -> dict[tuple[int, str], dict[str, str]]:
    out: dict[tuple[int, str], dict[str, str]] = {}
    for row in anchor_rows or []:
        try:
            fr = int(float(row.get("frame", "0") or 0))
        except Exception:
            continue
        side = row.get("human_side", "")
        if side in {"left", "right"}:
            out[(fr, side)] = row
    return out


def _by_frame_side(
    rows: list[dict[str, str]],
    anchor_rows: list[dict[str, str]] | None = None,
) -> dict[int, dict[str, dict[str, str]]]:
    pose_gate = _pose_anchor_gate(anchor_rows)
    anchor_by_key = _anchor_by_frame_side(anchor_rows)
    out: dict[int, dict[str, dict[str, str]]] = {}
    for row in rows:
        active = row.get("contact_active", "1")
        if str(active).strip().lower() in {"0", "false", "no"}:
            continue
        try:
            fr = int(float(row.get("frame", "0") or 0))
        except Exception:
            continue
        side = row.get("human_side", "")
        if side in {"left", "right"}:
            if pose_gate and not pose_gate.get((fr, side), False):
                continue
            merged = dict(row)
            anchor = anchor_by_key.get((fr, side))
            if anchor is not None:
                merged["pose_anchor_allowed"] = anchor.get("pose_anchor_allowed", "")
                merged["anchor_action"] = anchor.get("anchor_action", "")
                merged["observed_object_local_s"] = anchor.get("observed_object_local_s", merged.get("object_local_s", ""))
                merged["stable_object_local_s"] = anchor.get("stable_object_local_s", merged.get("stable_object_local_s", ""))
                merged["anchor_update_allowed"] = anchor.get("anchor_update_allowed", "")
                merged["contact_observed"] = anchor.get("contact_observed", "")
                merged["contact_persistent"] = anchor.get("contact_persistent", "")
                merged["line_observation_trusted"] = anchor.get("line_observation_trusted", merged.get("line_observation_trusted", ""))
                merged["palm_to_line_px"] = anchor.get("palm_to_line_px", merged.get("palm_to_line_px", ""))
            out.setdefault(fr, {})[side] = merged
    return out


def _gate_contact_rows_for_pose_anchors(
    contact_rows: list[dict[str, str]],
    anchor_rows: list[dict[str, str]] | None,
) -> list[dict[str, str]]:
    anchor_by_key: dict[tuple[int, str], dict[str, str]] = {}
    for row in anchor_rows or []:
        try:
            fr = int(float(row.get("frame", "0") or 0))
        except Exception:
            continue
        side = row.get("human_side", "")
        if side in {"left", "right"}:
            anchor_by_key[(fr, side)] = row
    if not anchor_by_key:
        return [dict(row) for row in contact_rows]

    out: list[dict[str, str]] = []
    for row in contact_rows:
        rec = dict(row)
        try:
            fr = int(float(rec.get("frame", "0") or 0))
        except Exception:
            fr = -1
        side = rec.get("human_side", "")
        anchor = anchor_by_key.get((fr, side))
        if anchor is not None:
            pose_allowed = "1" if anchor.get("pose_anchor_allowed") == "1" else "0"
            rec["contact_active"] = pose_allowed
            for key in (
                "contact_observed",
                "contact_persistent",
                "anchor_update_allowed",
                "pose_anchor_allowed",
                "anchor_action",
                "contact_disambiguation",
                "observed_object_local_s",
                "stable_object_local_s",
                "occlusion_state",
            ):
                if key in anchor:
                    rec[key] = anchor[key]
        out.append(rec)
    return out


def _qz(angle: float) -> tuple[float, float, float, float]:
    half = 0.5 * angle
    return math.cos(half), 0.0, 0.0, math.sin(half)


def _project(points: np.ndarray, camera: dict[str, float]) -> np.ndarray:
    z = np.clip(points[:, 2], 1e-6, None)
    return np.column_stack(
        [
            camera["fx"] * points[:, 0] / z + camera["cx"],
            camera["fy"] * points[:, 1] / z + camera["cy"],
        ]
    )


def _quat_wxyz_from_rot(rot: Rotation) -> tuple[float, float, float, float]:
    qx, qy, qz, qw = rot.as_quat()
    return float(qw), float(qx), float(qy), float(qz)


def _contact_local_s(row: dict[str, str]) -> tuple[float | None, str]:
    observed = float_or_none(row.get("observed_object_local_s"))
    if observed is None:
        observed = float_or_none(row.get("object_local_s"))
    stable = float_or_none(row.get("stable_object_local_s"))
    drift = abs(observed - stable) if observed is not None and stable is not None else float_or_none(row.get("local_s_drift"))
    palm_dist = float_or_none(row.get("palm_to_line_px"))
    trusted_line = str(row.get("line_observation_trusted", "1")).strip() == "1"
    visible = row.get("occlusion_state", "visible") == "visible"
    if observed is None:
        return stable, "stable_only"
    if stable is None:
        return observed, "observed_no_stable"
    if row.get("pose_anchor_allowed") == "1" and row.get("anchor_action") in {"update", "propagate"}:
        if (
            drift is not None
            and drift > 0.045
            and trusted_line
            and visible
            and (palm_dist is None or palm_dist <= 35.0)
        ):
            return observed, "observed_due_to_anchor_drift"
        return stable, "stable_pose_anchor"
    # Persistent fixed anchors are useful only while the current observation is
    # compatible with that anchor. During single-hand/occluded frames or endpoint
    # identity uncertainty, using a stale global median pulls the staff away from
    # the visible object. Keep the choice explicit for debugging/gates.
    if (drift is not None and drift > 0.22) or (palm_dist is not None and palm_dist > 55.0):
        return observed, "observed_due_to_anchor_drift"
    return stable, "stable_persistent_anchor"


def _contact_3d_anchors(pair: dict[str, dict[str, str]]) -> list[tuple[float, np.ndarray, str]]:
    anchors: list[tuple[float, np.ndarray, str]] = []
    for side in ("left", "right"):
        row = pair.get(side, {})
        if str(row.get("contact_active", "1")).strip().lower() in {"0", "false", "no"}:
            continue
        local_s, _source = _contact_local_s(row)
        px = float_or_none(row.get("palm_x")) or float_or_none(row.get("anchor_x_m"))
        py = float_or_none(row.get("palm_y")) or float_or_none(row.get("anchor_y_m"))
        pz = float_or_none(row.get("palm_z")) or float_or_none(row.get("anchor_z_m"))
        if local_s is None or px is None or py is None or pz is None:
            continue
        if not (0.0 <= local_s <= 1.0) or pz <= 0.0:
            continue
        anchors.append((float(local_s), np.asarray([px, py, pz], dtype=float), side))
    return anchors


def _refine_visible_pose_with_contact_se3(
    solved: tuple[float, float, float, float, float, float],
    profile: CaseProfile,
    pair: dict[str, dict[str, str]],
    base_t: tuple[float, float, float],
    base_quat_wxyz: tuple[float, float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float, float], dict[str, float | str]]:
    anchors = _contact_3d_anchors(pair)
    if not anchors:
        return base_t, base_quat_wxyz, {"line_contact_se3_refined": "0", "line_contact_se3_reason": "need_3d_anchor"}
    cx, cy, pixel_len, angle, _ax, _ay = solved
    length_m = float(profile.data.get("line_object", {}).get("length_m", 1.86))
    ux = math.cos(angle)
    uy = math.sin(angle)
    target_uv = np.asarray(
        [
            [cx - 0.5 * pixel_len * ux, cy - 0.5 * pixel_len * uy],
            [cx + 0.5 * pixel_len * ux, cy + 0.5 * pixel_len * uy],
        ],
        dtype=float,
    )
    endpoints_local = np.asarray([[-0.5 * length_m, 0.0, 0.0], [0.5 * length_m, 0.0, 0.0]], dtype=float)
    base_rot = Rotation.from_quat([base_quat_wxyz[1], base_quat_wxyz[2], base_quat_wxyz[3], base_quat_wxyz[0]])
    base_rotvec = base_rot.as_rotvec()
    base_t_arr = np.asarray(base_t, dtype=float)
    anchor_s0 = np.asarray([local_s for local_s, _palm, _side in anchors], dtype=float)
    base_lower = np.concatenate(
        [
            base_rotvec,
            base_t_arr,
        ]
    ) + np.asarray([-1.10, -1.35, -0.45, -1.25, -1.25, -3.40], dtype=float)
    base_upper = np.concatenate(
        [
            base_rotvec,
            base_t_arr,
        ]
    ) + np.asarray([1.10, 1.35, 0.45, 1.25, 1.25, 3.40], dtype=float)

    def optimize(allow_local_s_slack: bool) -> tuple[np.ndarray, object, float, float, list[str]]:
        if allow_local_s_slack:
            x0 = np.concatenate([base_rotvec, base_t_arr, anchor_s0])
            lower = np.concatenate([base_lower, np.clip(anchor_s0 - 0.35, 0.0, 1.0)])
            upper = np.concatenate([base_upper, np.clip(anchor_s0 + 0.35, 0.0, 1.0)])
            overlay_scale = 24.0
            contact_scale = 1.00
            local_s_prior = 0.02
        else:
            x0 = np.concatenate([base_rotvec, base_t_arr])
            lower = base_lower
            upper = base_upper
            overlay_scale = 3.5
            contact_scale = 1.15
            local_s_prior = 0.0

        def residual(x: np.ndarray) -> np.ndarray:
            rot = Rotation.from_rotvec(x[:3]).as_matrix()
            t = x[3:6]
            vals: list[float] = []
            pred_endpoints = endpoints_local @ rot.T + t
            pred_uv = _project(pred_endpoints, profile.camera)
            vals.extend(((pred_uv - target_uv).reshape(-1) / overlay_scale).tolist())
            for j, (local_s, palm, _side) in enumerate(anchors):
                opt_s = float(x[6 + j]) if allow_local_s_slack else local_s
                local = np.asarray([(opt_s - 0.5) * length_m, 0.0, 0.0], dtype=float)
                pred = local @ rot.T + t
                vals.extend(((pred - palm) / 0.060 * contact_scale).tolist())
                if allow_local_s_slack:
                    vals.append((opt_s - local_s) / 0.120 * local_s_prior)
            vals.extend(((x[:3] - base_rotvec) / np.asarray([0.70, 0.85, 0.32]) * 0.12).tolist())
            vals.extend(((x[3:5] - base_t_arr[:2]) / 0.45 * 0.10).tolist())
            vals.extend(((x[5:6] - base_t_arr[2:3]) / 1.60 * 0.06).tolist())
            return np.asarray(vals, dtype=float)

        res = least_squares(
            residual,
            np.clip(x0, lower, upper),
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=0.9,
            max_nfev=120 if allow_local_s_slack else 80,
        )
        opt = res.x
        rot = Rotation.from_rotvec(opt[:3]).as_matrix()
        t = opt[3:6]
        pred_endpoints = endpoints_local @ rot.T + t
        overlay_err = float(np.max(np.linalg.norm(_project(pred_endpoints, profile.camera) - target_uv, axis=1)))
        opt_anchor_s = opt[6 : 6 + len(anchors)] if allow_local_s_slack else anchor_s0
        gaps = []
        local_s_adjustments = []
        for j, (local_s, palm, side) in enumerate(anchors):
            opt_s = float(opt_anchor_s[j]) if j < len(opt_anchor_s) else local_s
            local_s_adjustments.append(f"{side}:{local_s:.4f}->{opt_s:.4f}")
            local = np.asarray([(opt_s - 0.5) * length_m, 0.0, 0.0], dtype=float)
            gaps.append(float(np.linalg.norm(local @ rot.T + t - palm)))
        return opt, res, overlay_err, max(gaps) if gaps else -1.0, local_s_adjustments

    opt, res, overlay_err, max_gap, local_s_adjustments = optimize(False)
    if len(anchors) >= 2 and max_gap > 0.18:
        opt, res, overlay_err, max_gap, local_s_adjustments = optimize(True)
    rot = Rotation.from_rotvec(opt[:3])
    t = tuple(float(x) for x in opt[3:6])
    q = _quat_wxyz_from_rot(rot)
    return t, q, {
        "line_contact_se3_refined": "1",
        "line_contact_se3_cost": float(res.cost),
        "line_contact_se3_overlay_err_px": overlay_err,
        "line_contact_se3_max_gap_m": max_gap,
        "line_contact_se3_anchor_count": float(len(anchors)),
        "line_contact_se3_local_s_adjustments": "|".join(local_s_adjustments),
    }


def _solve_axis_from_contacts(left: dict[str, str], right: dict[str, str]) -> tuple[float, float, float, float, float, float] | None:
    lu = float_or_none(left.get("palm_u"))
    lv = float_or_none(left.get("palm_v"))
    ru = float_or_none(right.get("palm_u"))
    rv = float_or_none(right.get("palm_v"))
    ls, _ls_source = _contact_local_s(left)
    rs, _rs_source = _contact_local_s(right)
    if None in {lu, lv, ru, rv, ls, rs}:
        return None
    denom = float(ls - rs)  # type: ignore[operator]
    if abs(denom) < 0.08:
        return None
    ax = float((lu - ru) / denom)  # type: ignore[operator]
    ay = float((lv - rv) / denom)  # type: ignore[operator]
    px0 = float(lu - float(ls) * ax)  # local_s=0 endpoint
    py0 = float(lv - float(ls) * ay)
    cx = px0 + 0.5 * ax
    cy = py0 + 0.5 * ay
    pixel_len = math.hypot(ax, ay)
    if not math.isfinite(pixel_len) or pixel_len < 60:
        return None
    angle = math.atan2(ay, ax)
    return cx, cy, pixel_len, angle, ax, ay


def _line_tracks(profile: CaseProfile) -> dict[int, tuple[float, float, float, float]]:
    paths = stage_paths(profile)
    corr_path = paths["line_correspondence"]
    if corr_path.exists():
        out: dict[int, tuple[float, float, float, float]] = {}
        for row in read_csv(corr_path):
            try:
                fr = int(float(row["frame"]))
                out[fr] = (
                    float(row["visible_x1"]),
                    float(row["visible_y1"]),
                    float(row["visible_x2"]),
                    float(row["visible_y2"]),
                )
            except Exception:
                continue
        if out:
            return out

    path = profile.sample_dir / "results" / "tracking" / "object_mesh_tracks_test.csv"
    grouped: dict[int, dict[str, tuple[float, float]]] = {}
    if not path.exists():
        return {}
    for row in read_csv(path):
        pid = row.get("point_id", "")
        if pid not in {"tip_a", "tip_b"}:
            continue
        try:
            fr = int(float(row["frame"]))
            grouped.setdefault(fr, {})[pid] = (float(row["x"]), float(row["y"]))
        except Exception:
            continue
    out = {}
    for fr, pts in grouped.items():
        if "tip_a" in pts and "tip_b" in pts:
            ax, ay = pts["tip_a"]
            bx, by = pts["tip_b"]
            out[fr] = (ax, ay, bx, by)
    return out


def _solve_axis_from_visible(track: tuple[float, float, float, float]) -> tuple[float, float, float, float, float, float] | None:
    ax0, ay0, bx0, by0 = track
    dx = bx0 - ax0
    dy = by0 - ay0
    pixel_len = math.hypot(dx, dy)
    if pixel_len < 90:
        return None
    cx = 0.5 * (ax0 + bx0)
    cy = 0.5 * (ay0 + by0)
    angle = math.atan2(dy, dx)
    return cx, cy, pixel_len, angle, dx, dy


def _solve_axis_from_contact_with_angle(
    pair: dict[str, dict[str, str]],
    angle: float,
) -> tuple[float, float, float, float, float, float] | None:
    left = pair.get("left", {})
    right = pair.get("right", {})
    lu = float_or_none(left.get("palm_u"))
    lv = float_or_none(left.get("palm_v"))
    ru = float_or_none(right.get("palm_u"))
    rv = float_or_none(right.get("palm_v"))
    ls, _ls_source = _contact_local_s(left)
    rs, _rs_source = _contact_local_s(right)
    if None in {lu, lv, ru, rv, ls, rs}:
        return None
    denom = float(ls - rs)  # type: ignore[operator]
    if abs(denom) < 0.08:
        return None
    ux = math.cos(angle)
    uy = math.sin(angle)
    left_x, left_y = float(lu), float(lv)
    right_x, right_y = float(ru), float(rv)
    # Estimate object image length along the visible axis from the two persistent
    # local contact coordinates. The center is the least-squares midpoint that
    # keeps the two palm anchors at their fixed local-s positions.
    pixel_len = ((left_x - right_x) * ux + (left_y - right_y) * uy) / denom
    if pixel_len < 0:
        pixel_len = -pixel_len
        angle = angle + math.pi
        ux = -ux
        uy = -uy
    if not math.isfinite(pixel_len) or pixel_len < 60:
        return None
    left_cx = left_x - (float(ls) - 0.5) * pixel_len * ux
    left_cy = left_y - (float(ls) - 0.5) * pixel_len * uy
    right_cx = right_x - (float(rs) - 0.5) * pixel_len * ux
    right_cy = right_y - (float(rs) - 0.5) * pixel_len * uy
    cx = 0.5 * (left_cx + right_cx)
    cy = 0.5 * (left_cy + right_cy)
    ax = pixel_len * math.cos(angle)
    ay = pixel_len * math.sin(angle)
    return cx, cy, pixel_len, angle, ax, ay


def _angle_delta(a: float, b: float) -> float:
    d = abs(math.atan2(math.sin(a - b), math.cos(a - b)))
    return min(d, abs(math.pi - d))


def _blend_solutions(
    contact: tuple[float, float, float, float, float, float],
    visible: tuple[float, float, float, float, float, float] | None,
    pair: dict[str, dict[str, str]],
) -> tuple[tuple[float, float, float, float, float, float], dict[str, float | str]]:
    if visible is None:
        return contact, {"blend_visible_weight": 0.0, "blend_reason": "no_visible_line"}
    cx_c, cy_c, len_c, ang_c, _ax_c, _ay_c = contact
    cx_v, cy_v, len_v, ang_v, ax_v, ay_v = visible
    left_dist = float_or_none(pair.get("left", {}).get("palm_to_line_px")) or 999.0
    right_dist = float_or_none(pair.get("right", {}).get("palm_to_line_px")) or 999.0
    palm_dist = 0.5 * (left_dist + right_dist)
    left_s, left_s_source = _contact_local_s(pair.get("left", {}))
    right_s, right_s_source = _contact_local_s(pair.get("right", {}))
    angle_gap = _angle_delta(ang_c, ang_v)
    original_angle_gap = angle_gap
    center_gap = math.hypot(cx_c - cx_v, cy_c - cy_v)
    return visible, {
        "blend_visible_weight": 1.0,
        "visible_len_px": len_v,
        "contact_len_px": len_c,
        "angle_gap_deg": math.degrees(original_angle_gap),
        "center_gap_px": center_gap,
        "mean_palm_to_visible_line_px": palm_dist,
        "left_local_s_source": left_s_source,
        "right_local_s_source": right_s_source,
        "left_local_s_used": left_s if left_s is not None else -1.0,
        "right_local_s_used": right_s if right_s is not None else -1.0,
        "blend_reason": "visible_line_primary_contact_depth_anchor",
    }


def _contact_depth_anchor(pair: dict[str, dict[str, str]], fallback_z: float | None = None) -> tuple[float | None, dict[str, float | str]]:
    weighted = []
    for side in ("left", "right"):
        row = pair.get(side, {})
        z = float_or_none(row.get("anchor_z_m")) or float_or_none(row.get("palm_z"))
        dist = float_or_none(row.get("palm_to_line_px"))
        if z is None or z <= 0.0:
            continue
        if dist is None:
            dist = 999.0
        # A palm that is close to the detected line is a strong 3D contact
        # anchor. Far palms still provide a weak prior during single-hand frames.
        weight = 1.0 / (1.0 + max(0.0, dist) / 35.0)
        weighted.append((z, weight, side, dist))
    if not weighted:
        return fallback_z, {"depth_anchor_source": "fallback_init" if fallback_z is not None else "missing"}
    total = sum(w for _z, w, _side, _dist in weighted)
    z_anchor = sum(z * w for z, w, _side, _dist in weighted) / max(total, 1e-6)
    return z_anchor, {
        "depth_anchor_source": "palm_3d_contact",
        "depth_anchor_z": z_anchor,
        "depth_anchor_sides": "+".join(side for _z, _w, side, _dist in weighted),
        "depth_anchor_mean_dist_px": sum(dist for _z, _w, _side, dist in weighted) / len(weighted),
    }


def _interp_solution(
    fr: int,
    solved_by_frame: dict[int, tuple[float, float, float, float, float, float]],
) -> tuple[float, float, float, float, float, float] | None:
    prev_frames = [x for x in solved_by_frame if x < fr]
    next_frames = [x for x in solved_by_frame if x > fr]
    if not prev_frames or not next_frames:
        return None
    a = max(prev_frames)
    b = min(next_frames)
    if b - a > 8:
        return None
    ta = solved_by_frame[a]
    tb = solved_by_frame[b]
    t = (fr - a) / float(b - a)
    cx = (1.0 - t) * ta[0] + t * tb[0]
    cy = (1.0 - t) * ta[1] + t * tb[1]
    pixel_len = (1.0 - t) * ta[2] + t * tb[2]
    delta = math.atan2(math.sin(tb[3] - ta[3]), math.cos(tb[3] - ta[3]))
    angle = ta[3] + t * delta
    ax = pixel_len * math.cos(angle)
    ay = pixel_len * math.sin(angle)
    return cx, cy, pixel_len, angle, ax, ay


def _merge_pose(
    init: dict[str, str],
    solved: tuple[float, float, float, float, float, float],
    profile: CaseProfile,
    blend_info: dict[str, float | str] | None = None,
    pair: dict[str, dict[str, str]] | None = None,
) -> dict[str, str]:
    cx, cy, pixel_len, angle, _ax, _ay = solved
    length_m = float(profile.data.get("line_object", {}).get("length_m", 1.86))
    fx = profile.camera["fx"]
    z_contact = fx * length_m / max(pixel_len, 1.0)
    z_init = float_or_none(init.get("tz")) or z_contact
    z_anchor, depth_info = _contact_depth_anchor(pair or {}, z_init)
    if z_anchor is not None:
        # Image-space overlay is primary. Contact depth is a small SE(3) depth
        # correction after the visual pose is solved, not a source for changing
        # the rendered line center/angle/scale.
        z_anchor_clamped = max(z_contact - 0.35, min(z_contact + 0.35, z_anchor))
        z = max(1.0, min(8.0, 0.86 * z_contact + 0.14 * z_anchor_clamped))
    else:
        z = max(2.2, min(7.0, 0.55 * z_contact + 0.45 * z_init))
    tx, ty, tz = backproject_uvz(cx, cy, z, profile.camera)
    qw, qx, qy, qz = _qz(angle)
    (tx, ty, tz), (qw, qx, qy, qz), se3_info = _refine_visible_pose_with_contact_se3(
        solved,
        profile,
        pair or {},
        (tx, ty, tz),
        (qw, qx, qy, qz),
    )
    out = dict(init)
    out.update(
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
            "u_contact_lock": f"{cx:.3f}",
            "v_contact_lock": f"{cy:.3f}",
            "contact_lock_pixel_len": f"{pixel_len:.3f}",
            "contact_lock_angle_rad": f"{angle:.9f}",
            "source": (init.get("source", "") + "+line_contact_lock").strip("+"),
        }
    )
    if blend_info:
        for key, value in blend_info.items():
            out[key] = f"{value:.6f}" if isinstance(value, float) else str(value)
    for key, value in depth_info.items():
        out[key] = f"{value:.6f}" if isinstance(value, float) else str(value)
    for key, value in se3_info.items():
        out[key] = f"{value:.6f}" if isinstance(value, float) else str(value)
    out["z_from_pixel_len_m"] = f"{z_contact:.6f}"
    return out


def apply(profile: CaseProfile) -> dict[str, object]:
    paths = stage_paths(profile)
    pose_rows = read_csv(paths["object_pose_init"] if paths["object_pose_init"].exists() else paths["object_pose"])
    contacts = read_csv(paths["contact_candidates"])
    anchor_rows = read_csv(paths["anchor_state"]) if paths["anchor_state"].exists() else []
    contacts_by_frame = _by_frame_side(contacts, anchor_rows)
    visible_by_frame = _line_tracks(profile)
    pose_by_frame: dict[int, dict[str, str]] = {}
    solved_by_frame: dict[int, tuple[float, float, float, float, float, float]] = {}
    blend_by_frame: dict[int, dict[str, float | str]] = {}
    solved_frames: list[int] = []
    skipped_frames: list[int] = []
    blend_rows: list[dict[str, object]] = []
    for row in pose_rows:
        try:
            fr = int(float(row.get("frame", "0") or 0))
        except Exception:
            continue
        pair = contacts_by_frame.get(fr, {})
        solved = _solve_axis_from_contacts(pair.get("left", {}), pair.get("right", {})) if {"left", "right"} <= set(pair) else None
        if solved is None:
            visible = _solve_axis_from_visible(visible_by_frame[fr]) if fr in visible_by_frame else None
            if visible is not None:
                solved_by_frame[fr] = visible
                blend_info = {
                    "blend_visible_weight": 1.0,
                    "visible_len_px": visible[2],
                    "contact_len_px": -1.0,
                    "angle_gap_deg": -1.0,
                    "center_gap_px": -1.0,
                    "mean_palm_to_visible_line_px": -1.0,
                    "blend_reason": "visible_line_fallback_no_contact_solution",
                }
                blend_by_frame[fr] = blend_info
                blend_rows.append({"frame": fr, **blend_info})
                solved_frames.append(fr)
            else:
                skipped_frames.append(fr)
        else:
            visible = _solve_axis_from_visible(visible_by_frame[fr]) if fr in visible_by_frame else None
            blended, blend_info = _blend_solutions(solved, visible, pair)
            solved_by_frame[fr] = blended
            blend_by_frame[fr] = blend_info
            blend_rows.append({"frame": fr, **blend_info})
            solved_frames.append(fr)
    filled_frames: list[int] = []
    for row in pose_rows:
        try:
            fr = int(float(row.get("frame", "0") or 0))
        except Exception:
            continue
        solved = solved_by_frame.get(fr)
        blend_info = blend_by_frame.get(fr)
        if solved is None:
            solved = _interp_solution(fr, solved_by_frame)
            if solved is not None:
                blend_info = {
                    "blend_visible_weight": 0.0,
                    "blend_reason": "temporal_contact_fill",
                }
                filled_frames.append(fr)
        pair = contacts_by_frame.get(fr, {})
        pose_by_frame[fr] = _merge_pose(row, solved, profile, blend_info, pair) if solved is not None else dict(row)
    out_rows = [pose_by_frame[int(float(row.get("frame", "0") or 0))] for row in pose_rows if int(float(row.get("frame", "0") or 0)) in pose_by_frame]
    out = write_csv(paths["object_pose"], out_rows)
    gated_contacts = _gate_contact_rows_for_pose_anchors(contacts, anchor_rows)
    contacts_out = write_csv(paths["object_contact_points"], gated_contacts) if gated_contacts else ""
    blend_csv = write_csv(profile.result_dir / "line_contact_lock_blend_debug.csv", blend_rows) if blend_rows else ""
    metrics = {
        "component": "line_contact_lock",
        "object_pose": str(out),
        "object_contact_points": str(contacts_out) if contacts_out else "",
        "contact_candidates": str(paths["contact_candidates"]),
        "rows": len(out_rows),
        "solved_frames": len(solved_frames),
        "skipped_frames": len(skipped_frames),
        "skipped_frame_examples": skipped_frames[:20],
        "temporal_filled_frames": len(filled_frames),
        "temporal_filled_frame_examples": filled_frames[:20],
        "blend_debug_csv": str(blend_csv) if blend_csv else "",
        "policy": "refine elongated object 2D pose from persistent left/right palm contacts at stable local line coordinates, blended with compatible visible line observations",
    }
    write_json(profile.result_dir / "line_contact_lock_metrics.json", metrics)
    return metrics
