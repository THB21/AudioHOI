from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from ....core.base.camera import K_FULLIMG
from ....core.base.config import CaseProfile
from ....core.base.io import read_csv, write_csv, write_json
from ....core.base.schema import stage_paths
from ....core.gates.vlm import GATE_REJECT, GATE_UNCLEAR
from ....core.gates.vlm_gates import disabled_anchor_frames, disabled_contact_frames
from ..sequence_se3_optimizer import smooth_quaternion_pose_sequence


def _f(row: dict[str, str], key: str, default: float = math.nan) -> float:
    try:
        value = row.get(key, "")
        return default if value in {"", None} else float(value)
    except Exception:
        return default


def _frame(row: dict[str, str]) -> int:
    return int(float(row.get("frame", "0") or 0))


def _angle_from_quat(row: dict[str, str]) -> float:
    qw = _f(row, "qw", 1.0)
    qz = _f(row, "qz", 0.0)
    return 2.0 * math.atan2(qz, qw)


def _qz(angle: float) -> tuple[float, float, float, float]:
    half = 0.5 * angle
    return math.cos(half), 0.0, 0.0, math.sin(half)


def _rotation_from_row(row: dict[str, str]) -> Rotation:
    q = np.asarray([_f(row, "qx", 0.0), _f(row, "qy", 0.0), _f(row, "qz", 0.0), _f(row, "qw", 1.0)], dtype=float)
    if np.linalg.norm(q) < 1e-8:
        q = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=float)
    return Rotation.from_quat(q / np.linalg.norm(q))


def _quat_strings(rot: Rotation) -> tuple[str, str, str, str]:
    qx, qy, qz, qw = rot.as_quat()
    return f"{qw:.6f}", f"{qx:.6f}", f"{qy:.6f}", f"{qz:.6f}"


def _backproject_anchor(row: dict[str, str], camera: dict[str, float]) -> np.ndarray | None:
    z = _f(row, "anchor_z_m")
    u = _f(row, "anchor_u")
    v = _f(row, "anchor_v")
    if not (math.isfinite(u) and math.isfinite(v) and math.isfinite(z) and z > 0.0):
        return None
    return np.asarray([(u - camera["cx"]) * z / camera["fx"], (v - camera["cy"]) * z / camera["fy"], z], dtype=float)


def _anchor_local_s(row: dict[str, str]) -> float:
    observed = _f(row, "observed_object_local_s")
    stable = _f(row, "stable_object_local_s", observed)
    if not math.isfinite(observed):
        return stable
    if not math.isfinite(stable):
        return observed
    drift = abs(observed - stable)
    trusted_line = str(row.get("line_observation_trusted", "1")).strip() == "1"
    visible = row.get("occlusion_state", "visible") == "visible"
    palm_dist = _f(row, "palm_to_line_px", 0.0)
    if drift > 0.045 and trusted_line and visible and (not math.isfinite(palm_dist) or palm_dist <= 35.0):
        return observed
    return stable


def _stable_anchor_gaps(
    rot: Rotation,
    t: np.ndarray,
    anchors: list[dict[str, str]],
    *,
    line_length_m: float,
    camera: dict[str, float],
) -> list[float]:
    gaps: list[float] = []
    R = rot.as_matrix()
    for anchor in anchors:
        if anchor.get("pose_anchor_allowed") != "1":
            continue
        s = _anchor_local_s(anchor)
        palm = _backproject_anchor(anchor, camera)
        if palm is None or not math.isfinite(s):
            continue
        pred = t + R @ np.asarray([(s - 0.5) * line_length_m, 0.0, 0.0], dtype=float)
        gaps.append(float(np.linalg.norm(pred - palm)))
    return gaps


def _endpoint_identity_candidates(row: dict[str, str], angle: float) -> list[tuple[Rotation, int]]:
    if row.get("line_contact_se3_refined") == "1":
        rot = _rotation_from_row(row)
        return [(rot, 0), (rot * Rotation.from_euler("z", math.pi), 1)]
    qw, qx, qy, qz = _qz(angle)
    return [(Rotation.from_quat([qx, qy, qz, qw]), 0)]


def _choose_endpoint_identity_states(
    pose_rows: list[dict[str, str]],
    frames: list[int],
    angles: np.ndarray,
    anchor_state_by_frame: dict[int, list[dict[str, str]]],
    *,
    line_length_m: float,
    camera: dict[str, float],
) -> dict[int, tuple[Rotation, int]]:
    all_candidates: list[list[tuple[Rotation, int, float]]] = []
    for i, row in enumerate(pose_rows):
        t = np.asarray([_f(row, "tx", 0.0), _f(row, "ty", 0.0), _f(row, "tz", 3.0)], dtype=float)
        anchors = anchor_state_by_frame.get(frames[i], [])
        row_candidates: list[tuple[Rotation, int, float]] = []
        for rot, flipped in _endpoint_identity_candidates(row, float(angles[i])):
            gaps = _stable_anchor_gaps(rot, t, anchors, line_length_m=line_length_m, camera=camera)
            anchor_cost = (float(np.median(gaps)) / 0.18 * 5.0) if gaps else 0.0
            flip_cost = 0.08 if flipped else 0.0
            row_candidates.append((rot, flipped, anchor_cost + flip_cost))
        all_candidates.append(row_candidates)

    costs: list[np.ndarray] = []
    parents: list[np.ndarray] = []
    for i, candidates in enumerate(all_candidates):
        cur = np.asarray([c[2] for c in candidates], dtype=float)
        parent = np.full(len(candidates), -1, dtype=int)
        if i > 0:
            prev_candidates = all_candidates[i - 1]
            prev_cost = costs[-1]
            next_cost = np.full(len(candidates), math.inf, dtype=float)
            for j, (rot, _flipped, unary) in enumerate(candidates):
                axis = rot.as_matrix() @ np.asarray([1.0, 0.0, 0.0], dtype=float)
                best = math.inf
                best_k = 0
                for k, (prev_rot, _prev_flipped, _prev_unary) in enumerate(prev_candidates):
                    prev_axis = prev_rot.as_matrix() @ np.asarray([1.0, 0.0, 0.0], dtype=float)
                    transition = 4.2 * (1.0 - float(np.dot(prev_axis, axis)))
                    value = float(prev_cost[k]) + unary + transition
                    if value < best:
                        best = value
                        best_k = k
                next_cost[j] = best
                parent[j] = best_k
            cur = next_cost
        costs.append(cur)
        parents.append(parent)

    chosen_idx = int(np.argmin(costs[-1])) if costs else 0
    out: dict[int, tuple[Rotation, int]] = {}
    for i in range(len(all_candidates) - 1, -1, -1):
        rot, flipped, _cost = all_candidates[i][chosen_idx]
        out[frames[i]] = (rot, flipped)
        chosen_idx = int(parents[i][chosen_idx]) if parents[i][chosen_idx] >= 0 else 0
    return out


def _suppress_isolated_se3_spikes(
    rows: list[dict[str, str]],
    residual_rows: list[dict[str, object]],
    audit_rows: list[dict[str, object]],
) -> None:
    if len(rows) < 3:
        return
    rotations = [_rotation_from_row(row) for row in rows]
    translations = [
        np.asarray([_f(row, "tx", 0.0), _f(row, "ty", 0.0), _f(row, "tz", 0.0)], dtype=float)
        for row in rows
    ]
    for i in range(1, len(rows) - 1):
        if rows[i].get("line_contact_se3_refined") != "1":
            continue
        if rows[i - 1].get("line_contact_se3_refined") != "1" or rows[i + 1].get("line_contact_se3_refined") != "1":
            continue
        prev_rot = rotations[i - 1]
        cur_rot = rotations[i]
        next_rot = rotations[i + 1]
        d_prev = float((prev_rot.inv() * cur_rot).magnitude())
        d_next = float((cur_rot.inv() * next_rot).magnitude())
        d_bridge = float((prev_rot.inv() * next_rot).magnitude())
        prev_axis = prev_rot.as_matrix() @ np.asarray([1.0, 0.0, 0.0], dtype=float)
        cur_axis = cur_rot.as_matrix() @ np.asarray([1.0, 0.0, 0.0], dtype=float)
        next_axis = next_rot.as_matrix() @ np.asarray([1.0, 0.0, 0.0], dtype=float)
        rotation_spike = d_prev > math.radians(65.0) and d_next > math.radians(65.0) and d_bridge < math.radians(45.0)
        depth_axis_flip = (
            abs(prev_axis[2]) > 0.16
            and abs(cur_axis[2]) > 0.22
            and abs(next_axis[2]) > 0.16
            and prev_axis[2] * next_axis[2] > 0.0
            and cur_axis[2] * prev_axis[2] < 0.0
            and d_bridge < math.radians(32.0)
        )
        if not (rotation_spike or depth_axis_flip):
            continue
        slerp = Slerp([0.0, 1.0], Rotation.concatenate([prev_rot, next_rot]))
        interp_rot = slerp([0.5])[0]
        interp_t = 0.5 * (translations[i - 1] + translations[i + 1])
        qw, qx, qy, qz = _quat_strings(interp_rot)
        rows[i].update(
            {
                "tx": f"{interp_t[0]:.6f}",
                "ty": f"{interp_t[1]:.6f}",
                "tz": f"{max(0.5, interp_t[2]):.6f}",
                "qw": qw,
                "qx": qx,
                "qy": qy,
                "qz": qz,
                "source": (rows[i].get("source", "") + "+isolated_se3_spike_interpolate").strip("+"),
            }
        )
        rotations[i] = interp_rot
        translations[i] = interp_t
        residual_rows[i]["freeze_or_propagate"] = "interpolate"
        residual_rows[i]["se3_spike_interpolated"] = "1"
        audit_rows[i]["smoothness_spike"] = "1"
        audit_rows[i]["freeze_or_propagate"] = "interpolate"
        audit_rows[i]["se3_spike_interpolated"] = "1"


def _unwrap(values: np.ndarray) -> np.ndarray:
    return np.unwrap(values)


def _interp_bad_frames(values: np.ndarray, good: np.ndarray) -> np.ndarray:
    out = values.copy()
    idx = np.arange(len(values))
    if np.count_nonzero(good) < 2:
        return out
    for dim in range(values.shape[1]):
        out[:, dim] = np.interp(idx, idx[good], values[good, dim])
    return out


def _anchor_targets(anchor_state: dict[int, list[dict[str, str]]]) -> dict[int, tuple[float, float, int]]:
    out: dict[int, tuple[float, float, int]] = {}
    for fr, rows in anchor_state.items():
        pts = []
        for row in rows:
            if row.get("pose_anchor_allowed") != "1":
                continue
            u = _f(row, "anchor_u")
            v = _f(row, "anchor_v")
            if math.isfinite(u) and math.isfinite(v):
                pts.append((u, v))
        if pts:
            arr = np.asarray(pts, dtype=float)
            out[fr] = (float(np.mean(arr[:, 0])), float(np.mean(arr[:, 1])), len(pts))
    return out


def _has_reliable_physical_contact(anchors: list[dict[str, str]]) -> bool:
    reliable = 0
    for anchor in anchors:
        if anchor.get("pose_anchor_allowed") != "1":
            continue
        dist = _f(anchor, "palm_to_line_px", math.inf)
        if math.isfinite(dist) and dist <= 18.0:
            reliable += 1
    return reliable >= 2


def _line_center_len_angle(row: dict[str, str]) -> tuple[float, float, float, float]:
    x1 = _f(row, "physical_x1", _f(row, "visible_x1"))
    y1 = _f(row, "physical_y1", _f(row, "visible_y1"))
    x2 = _f(row, "physical_x2", _f(row, "visible_x2"))
    y2 = _f(row, "physical_y2", _f(row, "visible_y2"))
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    length = _f(row, "visible_len_px", math.hypot(x2 - x1, y2 - y1))
    angle = _f(row, "visible_angle_rad", math.atan2(y2 - y1, x2 - x1))
    return cx, cy, length, angle


def _mean_rotation(rows: list[dict[str, str]]) -> Rotation:
    rots = [_rotation_from_row(row) for row in rows]
    if not rots:
        return Rotation.identity()
    base = rots[0]
    vecs = [(base.inv() * rot).as_rotvec() for rot in rots]
    return base * Rotation.from_rotvec(np.median(np.asarray(vecs, dtype=float), axis=0))


def _freeze_reliable_static_tail(
    rows: list[dict[str, str]],
    residual_rows: list[dict[str, object]],
    audit_rows: list[dict[str, object]],
    frames: list[int],
    correspondence_by_frame: dict[int, dict[str, str]],
    anchor_state_by_frame: dict[int, list[dict[str, str]]],
    config: dict[str, object],
) -> None:
    min_frames = int(config.get("static_tail_min_frames", 12))
    if len(rows) < min_frames:
        return
    center_thr = float(config.get("static_tail_center_px", 2.0))
    angle_thr = math.radians(float(config.get("static_tail_angle_deg", 0.75)))
    length_thr = float(config.get("static_tail_length_px", 3.0))
    stable = [False] * len(rows)
    prev_features: tuple[float, float, float, float] | None = None
    for i, fr in enumerate(frames):
        corr = correspondence_by_frame.get(fr, {})
        reliable = (
            str(corr.get("line_observation_trusted", "1")) == "1"
            and corr.get("occlusion_state", "visible") == "visible"
            and _has_reliable_physical_contact(anchor_state_by_frame.get(fr, []))
        )
        features = _line_center_len_angle(corr) if reliable else None
        if i == 0:
            stable[i] = bool(reliable)
        elif reliable and prev_features is not None and features is not None:
            cx, cy, length, angle = features
            pcx, pcy, plen, pangle = prev_features
            dcenter = math.hypot(cx - pcx, cy - pcy)
            dangle = abs(math.atan2(math.sin(angle - pangle), math.cos(angle - pangle)))
            stable[i] = dcenter <= center_thr and dangle <= angle_thr and abs(length - plen) <= length_thr
        if features is not None:
            prev_features = features
        else:
            prev_features = None
    start = len(rows) - 1
    while start >= 0 and stable[start]:
        start -= 1
    start += 1
    if len(rows) - start < min_frames:
        return
    tail = rows[start:]
    trans = np.asarray([[_f(row, "tx", 0.0), _f(row, "ty", 0.0), _f(row, "tz", 3.0)] for row in tail], dtype=float)
    target_t = np.median(trans, axis=0)
    target_rot = _mean_rotation(tail)
    qw, qx, qy, qz = _quat_strings(target_rot)
    for i in range(start, len(rows)):
        rows[i].update(
            {
                "tx": f"{target_t[0]:.6f}",
                "ty": f"{target_t[1]:.6f}",
                "tz": f"{max(0.5, target_t[2]):.6f}",
                "qw": qw,
                "qx": qx,
                "qy": qy,
                "qz": qz,
                "source": (rows[i].get("source", "") + "+static_tail_freeze").strip("+"),
            }
        )
        residual_rows[i]["static_tail_freeze"] = "1"
        audit_rows[i]["static_tail_freeze"] = "1"
        audit_rows[i]["freeze_or_propagate"] = "static_tail_freeze"


def smooth_line_pose_rows(
    pose_rows: list[dict[str, str]],
    correspondence_by_frame: dict[int, dict[str, str]],
    anchor_state_by_frame: dict[int, list[dict[str, str]]],
    *,
    disabled_contact: set[int] | None = None,
    disabled_anchor: set[int] | None = None,
    disabled_pose_refine: set[int] | None = None,
    line_length_m: float = 1.86,
    camera: dict[str, float] | None = None,
    sequence_se3_config: dict[str, object] | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, object]], list[dict[str, object]]]:
    frames = [_frame(r) for r in pose_rows]
    n = len(pose_rows)
    disabled_contact = disabled_contact or set()
    disabled_anchor = disabled_anchor or set()
    disabled_pose_refine = disabled_pose_refine or set()
    camera = camera or K_FULLIMG
    raw = np.zeros((n, 5), dtype=float)
    for i, row in enumerate(pose_rows):
        raw[i] = [
            _f(row, "u_contact_lock", _f(row, "u_proj", 0.0)),
            _f(row, "v_contact_lock", _f(row, "v_proj", 0.0)),
            _f(row, "contact_lock_pixel_len", 1.0),
            _f(row, "tz", 3.0),
            _angle_from_quat(row),
        ]
    raw[:, 4] = _unwrap(raw[:, 4])
    trusted = np.asarray(
        [str(correspondence_by_frame.get(fr, {}).get("line_observation_trusted", "1")) == "1" for fr in frames],
        dtype=bool,
    )
    anchor_targets = {fr: pt for fr, pt in _anchor_targets(anchor_state_by_frame).items() if fr not in disabled_contact and fr not in disabled_anchor}
    visual_spike = np.zeros(n, dtype=bool)
    contact_spike = np.zeros(n, dtype=bool)
    smoothness_spike = np.zeros(n, dtype=bool)
    for i in range(1, n):
        duv = float(np.linalg.norm(raw[i, :2] - raw[i - 1, :2]))
        dlen = abs(float(raw[i, 2] - raw[i - 1, 2]))
        dang = abs(float(raw[i, 4] - raw[i - 1, 4]))
        visual_spike[i] = duv > 80.0 or dlen > 150.0 or dang > math.radians(35.0)
        smoothness_spike[i] = duv > 120.0 or dang > math.radians(60.0)
    for i, fr in enumerate(frames):
        if fr in disabled_pose_refine:
            continue
        if fr in anchor_targets:
            au, av, count = anchor_targets[fr]
            if count < 2:
                continue
            contact_spike[i] = math.hypot(raw[i, 0] - au, raw[i, 1] - av) > 120.0
    pose_refine_allowed = np.asarray([fr not in disabled_pose_refine for fr in frames], dtype=bool)
    good = trusted & pose_refine_allowed
    init = _interp_bad_frames(raw, good)
    prior_weights = np.where(good, 0.95, 0.05)
    scales = np.asarray([120.0, 120.0, 180.0, 0.35, math.radians(25.0)], dtype=float)

    def residual(x_flat: np.ndarray) -> np.ndarray:
        x = x_flat.reshape(n, 5)
        out: list[float] = []
        out.extend((((x - raw) / scales) * prior_weights[:, None]).reshape(-1).tolist())
        for i, fr in enumerate(frames):
            target = anchor_targets.get(fr)
            if target is None or fr in disabled_pose_refine:
                continue
            au, av, count = target
            if count < 2:
                continue
            weight = 0.18 if trusted[i] else 0.32
            out.append((x[i, 0] - au) / 90.0 * weight)
            out.append((x[i, 1] - av) / 90.0 * weight)
        if n > 1:
            out.extend((np.diff(x[:, 0]) / 95.0 * 0.75).tolist())
            out.extend((np.diff(x[:, 1]) / 95.0 * 0.75).tolist())
            out.extend((np.diff(x[:, 2]) / 135.0 * 0.85).tolist())
            out.extend((np.diff(x[:, 3]) / 0.22 * 0.70).tolist())
            out.extend((np.diff(x[:, 4]) / math.radians(28.0) * 0.85).tolist())
        if n > 2:
            out.extend((np.diff(x[:, 0], n=2) / 70.0 * 2.2).tolist())
            out.extend((np.diff(x[:, 1], n=2) / 70.0 * 2.2).tolist())
            out.extend((np.diff(x[:, 2], n=2) / 110.0 * 2.4).tolist())
            out.extend((np.diff(x[:, 4], n=2) / math.radians(18.0) * 2.5).tolist())
        return np.asarray(out, dtype=float)

    res = least_squares(residual, init.reshape(-1), loss="soft_l1", f_scale=0.12, max_nfev=160)
    opt = res.x.reshape(n, 5)
    endpoint_identity = _choose_endpoint_identity_states(
        pose_rows,
        frames,
        opt[:, 4],
        anchor_state_by_frame,
        line_length_m=line_length_m,
        camera=camera,
    )
    out_rows: list[dict[str, str]] = []
    residual_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    for i, row in enumerate(pose_rows):
        out = dict(row)
        u, v, pix_len, tz, angle = opt[i]
        visual_locked = bool(trusted[i] and not frames[i] in disabled_pose_refine)
        if visual_locked:
            target = anchor_targets.get(frames[i])
            if target is not None and target[2] >= 2:
                u = float(np.clip(u, raw[i, 0] - 12.0, raw[i, 0] + 12.0))
                v = float(np.clip(v, raw[i, 1] - 12.0, raw[i, 1] + 12.0))
                pix_len = float(np.clip(pix_len, raw[i, 2] - 18.0, raw[i, 2] + 18.0))
                angle = float(np.clip(angle, raw[i, 4] - math.radians(4.0), raw[i, 4] + math.radians(4.0)))
            else:
                u, v, pix_len, tz, angle = raw[i]
        if row.get("line_contact_se3_refined") == "1":
            tx = row.get("tx", "")
            ty = row.get("ty", "")
            tz_out = row.get("tz", f"{max(0.5, tz):.6f}")
            chosen_rot, endpoint_flip_suppressed = endpoint_identity.get(frames[i], (_rotation_from_row(row), 0))
            qw, qx, qy, qz = _quat_strings(chosen_rot)
        else:
            qw_f, qx_f, qy_f, qz_f = _qz(float(angle))
            qw = f"{qw_f:.6f}"
            qx = f"{qx_f:.6f}"
            qy = f"{qy_f:.6f}"
            qz = f"{qz_f:.6f}"
            tx = row.get("tx", "")
            ty = row.get("ty", "")
            tz_out = f"{max(0.5, tz):.6f}"
            endpoint_flip_suppressed = 0
        out.update(
            {
                "tx": str(tx),
                "ty": str(ty),
                "u_proj": f"{u:.3f}",
                "v_proj": f"{v:.3f}",
                "u_contact_lock": f"{u:.3f}",
                "v_contact_lock": f"{v:.3f}",
                "contact_lock_pixel_len": f"{max(1.0, pix_len):.3f}",
                "tz": str(tz_out),
                "qw": str(qw),
                "qx": str(qx),
                "qy": str(qy),
                "qz": str(qz),
                "source": (row.get("source", "") + "+generic_line_physical_smooth").strip("+"),
            }
        )
        out_rows.append(out)
        freeze_or_propagate = "propagate" if (not trusted[i] or frames[i] in disabled_pose_refine) else "optimize"
        residual_rows.append(
            {
                "frame": frames[i],
                "raw_u": f"{raw[i, 0]:.3f}",
                "raw_v": f"{raw[i, 1]:.3f}",
                "smooth_u": f"{u:.3f}",
                "smooth_v": f"{v:.3f}",
                "raw_len_px": f"{raw[i, 2]:.3f}",
                "smooth_len_px": f"{pix_len:.3f}",
                "visual_residual_px": f"{float(np.linalg.norm(opt[i, :2] - raw[i, :2])):.6f}",
                "length_residual_px": f"{abs(float(pix_len - raw[i, 2])):.6f}",
                "residual_visual_enabled": "1" if trusted[i] and frames[i] not in disabled_pose_refine else "0",
                "residual_contact_enabled": "1" if frames[i] in anchor_targets and anchor_targets[frames[i]][2] >= 2 and frames[i] not in disabled_pose_refine else "0",
                "residual_smooth_enabled": "1",
                "freeze_or_propagate": freeze_or_propagate,
                "vlm_contact_disabled": "1" if frames[i] in disabled_contact else "0",
                "vlm_anchor_disabled": "1" if frames[i] in disabled_anchor else "0",
                "vlm_pose_refine_disabled": "1" if frames[i] in disabled_pose_refine else "0",
                "endpoint_identity_flip_suppressed": "1" if endpoint_flip_suppressed else "0",
                "se3_spike_interpolated": "0",
            }
        )
        audit_rows.append(
            {
                "frame": frames[i],
                "line_observation_trusted": "1" if trusted[i] else "0",
                "visual_spike": "1" if visual_spike[i] else "0",
                "contact_spike": "1" if contact_spike[i] else "0",
                "smoothness_spike": "1" if smoothness_spike[i] else "0",
                "freeze_or_propagate": freeze_or_propagate,
                "vlm_contact_disabled": "1" if frames[i] in disabled_contact else "0",
                "vlm_anchor_disabled": "1" if frames[i] in disabled_anchor else "0",
                "vlm_pose_refine_disabled": "1" if frames[i] in disabled_pose_refine else "0",
                "occlusion_state": correspondence_by_frame.get(frames[i], {}).get("occlusion_state", ""),
                "endpoint_identity_flip_suppressed": "1" if endpoint_flip_suppressed else "0",
                "se3_spike_interpolated": "0",
            }
        )
    _suppress_isolated_se3_spikes(out_rows, residual_rows, audit_rows)
    sequence_se3_config = sequence_se3_config or {}
    low_trust = float(sequence_se3_config.get("low_trust_weight", 0.20))
    trusted_physical_contact_trust = float(sequence_se3_config.get("trusted_physical_contact_weight", 0.85))
    trust_weight = [
        (
            low_trust
            if (not trusted[i] or audit_rows[i].get("smoothness_spike") == "1")
            else (
                trusted_physical_contact_trust
                if frames[i] in disabled_pose_refine and _has_reliable_physical_contact(anchor_state_by_frame.get(frames[i], []))
                else (low_trust if frames[i] in disabled_pose_refine else 1.0)
            )
        )
        for i in range(n)
    ]
    out_rows, se3_audit_rows = smooth_quaternion_pose_sequence(
        out_rows,
        trust_weight=trust_weight,
        prior_w=float(sequence_se3_config.get("prior_w", 0.34)),
        smooth_w=float(sequence_se3_config.get("smooth_w", 0.95)),
        accel_w=float(sequence_se3_config.get("accel_w", 4.2)),
        translation_scales=tuple(float(v) for v in sequence_se3_config.get("translation_scales", (0.065, 0.065, 0.095))),
        rotation_scale=float(sequence_se3_config.get("rotation_scale", 0.070)),
    )
    for residual, audit, se3_audit in zip(residual_rows, audit_rows, se3_audit_rows):
        for key in ("se3_velocity_spike", "se3_accel_spike", "se3_smoothed", "delta_translation_m", "delta_rotation_rad", "trust_weight"):
            residual[key] = se3_audit[key]
            audit[key] = se3_audit[key]
        residual.setdefault("static_tail_freeze", "0")
        audit.setdefault("static_tail_freeze", "0")
    _freeze_reliable_static_tail(out_rows, residual_rows, audit_rows, frames, correspondence_by_frame, anchor_state_by_frame, sequence_se3_config)
    return out_rows, residual_rows, audit_rows


def _disabled_pose_refine_frames(profile: CaseProfile) -> set[int]:
    frames: set[int] = set()
    vlm_dir = stage_paths(profile)["vlm_dir"]
    for stage in ("stage3", "stage4"):
        gate_path = vlm_dir / stage / "vlm_gates.csv"
        if not gate_path.exists():
            continue
        for row in read_csv(gate_path):
            if row.get("is_effective") != "1":
                continue
            if row.get("report_only") == "1":
                continue
            if row.get("allow_pose_refine") == "1":
                continue
            if row.get("pass_gate") not in {GATE_REJECT, GATE_UNCLEAR}:
                continue
            fr = _frame(row)
            if fr >= 0:
                frames.add(fr)
    return frames


def apply(profile: CaseProfile, *, ignore_vlm_gates: bool = False, metrics_suffix: str = "") -> dict[str, object]:
    paths = stage_paths(profile)
    if not paths["object_pose"].exists() or not paths["line_correspondence"].exists() or not paths["anchor_state"].exists():
        metrics = {"component": "generic_line_physical_smooth", "enabled": False, "reason": "missing pose/correspondence/anchor_state"}
        write_json(profile.result_dir / f"generic_line_physical_smooth_metrics{metrics_suffix}.json", metrics)
        return metrics
    pose_rows = read_csv(paths["object_pose"])
    write_csv(paths["object_pose_pre_smooth"], pose_rows)
    correspondence = {_frame(r): r for r in read_csv(paths["line_correspondence"])}
    anchor_state: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(paths["anchor_state"]):
        anchor_state[_frame(row)].append(row)
    if ignore_vlm_gates:
        disabled_contact: set[int] = set()
        disabled_anchor: set[int] = set()
        disabled_pose_refine: set[int] = set()
    else:
        disabled_contact = disabled_contact_frames(profile, stage="stage2")
        disabled_anchor = disabled_anchor_frames(profile, stage="stage1") | disabled_anchor_frames(profile, stage="stage4")
        disabled_pose_refine = _disabled_pose_refine_frames(profile)
    out_rows, residual_rows, audit_rows = smooth_line_pose_rows(
        pose_rows,
        correspondence,
        anchor_state,
        disabled_contact=disabled_contact,
        disabled_anchor=disabled_anchor,
        disabled_pose_refine=disabled_pose_refine,
        line_length_m=float(profile.data.get("line_object", {}).get("length_m", 1.86)),
        camera=profile.camera,
        sequence_se3_config=profile.data.get("sequence_se3_optimizer", {})
        if isinstance(profile.data.get("sequence_se3_optimizer", {}), dict)
        else {},
    )
    write_csv(paths["physical_smooth_residuals"], residual_rows)
    write_csv(paths["pose_jump_audit"], audit_rows)
    write_csv(paths["object_pose"], out_rows)
    metrics = {
        "component": "generic_line_physical_smooth",
        "enabled": True,
        "object_pose_pre_smooth": str(paths["object_pose_pre_smooth"]),
        "object_pose": str(paths["object_pose"]),
        "physical_smooth_residuals": str(paths["physical_smooth_residuals"]),
        "pose_jump_audit": str(paths["pose_jump_audit"]),
        "rows": len(out_rows),
        "visual_spike_frames": [r["frame"] for r in audit_rows if r.get("visual_spike") == "1"][:80],
        "propagated_frames": [r["frame"] for r in audit_rows if r.get("freeze_or_propagate") == "propagate"][:80],
        "vlm_disabled_contact_frames": sorted(disabled_contact),
        "vlm_disabled_anchor_frames": sorted(disabled_anchor),
        "vlm_disabled_pose_refine_frames": sorted(disabled_pose_refine),
        "ignore_vlm_gates": ignore_vlm_gates,
        "sequence_se3_optimizer": profile.data.get("sequence_se3_optimizer", {}),
        "policy": "mainline sequence optimizer with visual/contact/depth/smooth/acceleration/freeze residuals for line-like objects",
    }
    write_json(profile.result_dir / f"generic_line_physical_smooth_metrics{metrics_suffix}.json", metrics)
    return metrics
