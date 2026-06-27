from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from ...components.contact.build_mug_grasp_anchor_state import FIELDS, build_state
from ...core.base.config import CaseProfile
from ...core.base.io import float_or_none, read_csv, repo_path, write_csv, write_json
from ...core.semantics.provenance import resolve_mug_m17_phase, write_mug_m17_reconstruction_report
from ...core.base.schema import stage_paths
from ...core.gates.vlm_gates import apply_contact_gate_to_rows


def _matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def _object_R(yaw: float, pitch: float, roll: float):
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    ry = [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]]
    rx = [[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]]
    rz = [[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]]
    return _matmul(_matmul(ry, rx), rz)


def _decompose_yxz(R) -> tuple[float, float, float]:
    pitch = math.asin(max(-1.0, min(1.0, -R[1][2])))
    yaw = math.atan2(R[0][2], R[2][2])
    roll = math.atan2(R[1][0], R[1][1])
    return yaw, pitch, roll


def _quat_from_matrix(m):
    tr = m[0][0] + m[1][1] + m[2][2]
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        return [0.25 * s, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s]
    if m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
        return [(m[2][1] - m[1][2]) / s, 0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s]
    if m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
        return [(m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s]
    s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
    return [(m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s]


def _matrix_from_quat(q):
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def _slerp(q0, q1, alpha: float):
    dot = sum(a * b for a, b in zip(q0, q1))
    if dot < 0.0:
        q1 = [-v for v in q1]
        dot = -dot
    if dot > 0.9995:
        q = [(1.0 - alpha) * a + alpha * b for a, b in zip(q0, q1)]
        n = math.sqrt(sum(v * v for v in q))
        return [v / n for v in q]
    theta_0 = math.acos(max(-1.0, min(1.0, dot)))
    theta = theta_0 * alpha
    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)
    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return [s0 * a + s1 * b for a, b in zip(q0, q1)]


def _rotation_step_deg(a, b) -> float:
    tr = sum(a[i][j] * b[i][j] for i in range(3) for j in range(3))
    # trace(a.T @ b) for rotation matrices.
    tr = sum(a[k][i] * b[k][i] for i in range(3) for k in range(3))
    return math.degrees(math.acos(max(-1.0, min(1.0, (tr - 1.0) * 0.5))))


def _median(vals: list[float]) -> float:
    vals = sorted(vals)
    n = len(vals)
    return vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])


def _detect_rotation_window(rows: list[dict[str, str]]) -> tuple[int, int, list[tuple[int, float]], str]:
    steps = []
    prev = None
    for row in rows:
        fr = int(float(row["frame"]))
        cur = _object_R(float(row["yaw"]), float(row["pitch"]), float(row["roll"]))
        if prev is not None:
            steps.append((fr, _rotation_step_deg(prev, cur)))
        prev = cur
    values = [s for _fr, s in steps]
    med = _median(values)
    mad = _median([abs(v - med) for v in values])
    threshold = max(12.0, med + 8.0 * 1.4826 * mad)
    candidates = [(fr, step) for fr, step in steps if 50 <= fr <= 75 and step >= threshold]
    if not candidates:
        first = int(float(rows[0]["frame"]))
        return first, first, steps, f"no_rotation_jump threshold={threshold:.3f}"
    jump_frame, jump_step = max(candidates, key=lambda item: item[1])
    return jump_frame - 1, jump_frame + 9, steps, f"auto_rotation_jump jump={jump_frame} step={jump_step:.3f} threshold={threshold:.3f}"


def _detect_static_frame(profile: CaseProfile, rows: list[dict[str, str]]) -> tuple[int, str]:
    stage_contact_state = stage_paths(profile)["contact_state"]
    path = stage_contact_state if stage_contact_state.exists() else profile.baseline.get("contact_state_csv")
    if path:
        try:
            run_start = None
            run_len = 0
            for row in read_csv(path):
                fr = int(float(row["frame"]))
                if fr < 150:
                    continue
                support_conf = float_or_none(row.get("support_conf")) or 0.0
                support_gap = float_or_none(row.get("signed_support_gap_px")) or -1e9
                accel = float_or_none(row.get("object_acceleration")) or 0.0
                floor_state = int(float(row.get("floor_contact_state", "0") or 0))
                good = floor_state == 1 or (support_conf >= 0.99 and support_gap >= -10.0 and accel <= 15.0)
                if good:
                    run_start = fr if run_start is None else run_start
                    run_len += 1
                    if run_len >= 8:
                        return int(run_start), "auto_table_static support_conf>=0.99 support_gap>=-10.0 run=8"
                else:
                    run_start = None
                    run_len = 0
        except Exception:
            pass
    return int(float(rows[-1]["frame"])), "fallback_last_frame"


def _build_mug_pose(profile: CaseProfile, out_path) -> dict[str, object]:
    paths = stage_paths(profile)
    pose_source = paths["object_pose_init"] if paths["object_pose_init"].exists() else repo_path(profile.baseline["m18_pose_csv"])
    rows = read_csv(pose_source)
    by_frame = {int(float(r["frame"])): r for r in rows}
    start, end, steps, rot_reason = _detect_rotation_window(rows)
    static_frame, static_reason = _detect_static_frame(profile, rows)
    q0 = _quat_from_matrix(_object_R(float(by_frame[start]["yaw"]), float(by_frame[start]["pitch"]), float(by_frame[start]["roll"])))
    q1 = _quat_from_matrix(_object_R(float(by_frame[end]["yaw"]), float(by_frame[end]["pitch"]), float(by_frame[end]["roll"])))
    fields = list(rows[0].keys())
    for key in ["m45_pose_source", "m45_rotation_smooth_start", "m45_rotation_smooth_end", "m45_static_frame", "m45_yaw_input", "m45_pitch_input", "m45_roll_input"]:
        if key not in fields:
            fields.append(key)
    out_rows = []
    for row in rows:
        fr = int(float(row["frame"]))
        new = dict(row)
        source = "unchanged"
        if start < fr < end:
            alpha = (fr - start) / (end - start)
            yaw, pitch, roll = _decompose_yxz(_matrix_from_quat(_slerp(q0, q1, alpha)))
            new["m45_yaw_input"] = row.get("yaw", "")
            new["m45_pitch_input"] = row.get("pitch", "")
            new["m45_roll_input"] = row.get("roll", "")
            new["yaw"] = f"{yaw:.9f}"
            new["pitch"] = f"{pitch:.9f}"
            new["roll"] = f"{roll:.9f}"
            new["yaw_deg"] = f"{math.degrees(yaw):.6f}"
            new["pitch_deg"] = f"{math.degrees(pitch):.6f}"
            new["roll_deg"] = f"{math.degrees(roll):.6f}"
            source = "rotation_slerp"
        new["m45_pose_source"] = source
        new["m45_rotation_smooth_start"] = str(start)
        new["m45_rotation_smooth_end"] = str(end)
        new["m45_static_frame"] = str(static_frame)
        out_rows.append(new)
    ref = next((r for r in out_rows if int(float(r["frame"])) == static_frame), None)
    if ref:
        for row in out_rows:
            if int(float(row["frame"])) > static_frame:
                for key in ["x", "y", "z", "yaw", "pitch", "roll", "scale", "yaw_deg", "pitch_deg", "roll_deg"]:
                    if key in ref:
                        row[key] = ref[key]
                row["m45_pose_source"] = "table_static_release"
    write_csv(out_path, out_rows, fields)
    return {
        "rotation_smooth_start": start,
        "rotation_smooth_end": end,
        "rotation_detection": rot_reason,
        "static_frame": static_frame,
        "static_detection": static_reason,
        "pose_source": str(pose_source),
    }


def _phase_anchors() -> list[tuple[int, float]]:
    return [
        (1, 0.447322), (20, -4.6), (26, 22.0), (30, 58.0), (35, 92.0),
        (45, 145.0), (50, 154.0), (55, 160.0), (57, 162.0), (60, 156.0),
        (62, 152.0), (65, 146.0), (75, 132.0), (84, 125.0), (86, 115.0),
        (89, 100.0), (95, 75.0), (100, 70.0), (105, 90.0), (107, 85.0),
        (110, 65.0), (112, 55.0), (115, 45.0), (120, 30.0), (126, 15.0),
        (135, 5.0), (145, -20.0), (150, -23.0), (180, -31.902183), (240, -31.744569),
    ]


def _pchip_slopes(x: list[float], y: list[float]) -> list[float]:
    n = len(x)
    h = [x[i + 1] - x[i] for i in range(n - 1)]
    d = [(y[i + 1] - y[i]) / h[i] for i in range(n - 1)]
    m = [0.0] * n
    for k in range(1, n - 1):
        if d[k - 1] == 0.0 or d[k] == 0.0 or (d[k - 1] < 0.0) != (d[k] < 0.0):
            m[k] = 0.0
        else:
            w1 = 2.0 * h[k] + h[k - 1]
            w2 = h[k] + 2.0 * h[k - 1]
            m[k] = (w1 + w2) / (w1 / d[k - 1] + w2 / d[k])

    def endpoint(h0, h1, d0, d1):
        val = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
        if (val < 0.0) != (d0 < 0.0):
            return 0.0
        if (d0 < 0.0) != (d1 < 0.0) and abs(val) > abs(3.0 * d0):
            return 3.0 * d0
        return val

    m[0] = endpoint(h[0], h[1], d[0], d[1]) if n > 2 else d[0]
    m[-1] = endpoint(h[-1], h[-2], d[-1], d[-2]) if n > 2 else d[-1]
    return m


def _pchip_eval(x: list[float], y: list[float], xs: list[float]) -> list[float]:
    m = _pchip_slopes(x, y)
    out = []
    for xv in xs:
        if xv <= x[0]:
            k = 0
        elif xv >= x[-1]:
            k = len(x) - 2
        else:
            k = 0
            while k + 1 < len(x) and x[k + 1] < xv:
                k += 1
        h = x[k + 1] - x[k]
        t = (xv - x[k]) / h
        out.append(
            (2 * t**3 - 3 * t**2 + 1) * y[k]
            + (t**3 - 2 * t**2 + t) * h * m[k]
            + (-2 * t**3 + 3 * t**2) * y[k + 1]
            + (t**3 - t**2) * h * m[k + 1]
        )
    return out


def _wrap_pi(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def _build_mug_phase(profile: CaseProfile, out_path) -> dict[str, object]:
    phase_source, phase_info = resolve_mug_m17_phase(profile)
    write_mug_m17_reconstruction_report(profile, phase_info)
    rows = read_csv(phase_source)
    frames = [int(float(r["frame"])) for r in rows]
    available = set(frames)
    anchors = [(fr, deg) for fr, deg in _phase_anchors() if fr in available]
    desired = _pchip_eval([float(fr) for fr, _deg in anchors], [deg for _fr, deg in anchors], [float(fr) for fr in frames])
    out_deg = list(desired)
    max_step = 7.0
    for i in range(1, len(out_deg)):
        delta = out_deg[i] - out_deg[i - 1]
        if abs(delta) > max_step:
            out_deg[i] = out_deg[i - 1] + math.copysign(max_step, delta)
    for i in range(len(out_deg) - 2, -1, -1):
        delta = out_deg[i] - out_deg[i + 1]
        if abs(delta) > max_step:
            out_deg[i] = out_deg[i + 1] + math.copysign(max_step, delta)
    fields = list(rows[0].keys())
    for key in ["m43_phase_rad", "m43_phase_deg", "m43_source"]:
        if key not in fields:
            fields.append(key)
    out_rows = []
    for row, deg in zip(rows, out_deg):
        rad = _wrap_pi(math.radians(deg))
        new = dict(row)
        new["m17_phase_rad"] = f"{rad:.9f}"
        new["m17_phase_deg"] = f"{math.degrees(rad):.6f}"
        new["m43_phase_rad"] = f"{rad:.9f}"
        new["m43_phase_deg"] = f"{math.degrees(rad):.6f}"
        new["m43_source"] = "smooth_entry_physical_phase_no_hide"
        out_rows.append(new)
    write_csv(out_path, out_rows, fields)
    return {
        "anchors": len(anchors),
        "max_abs_velocity_deg": max(abs(out_deg[i] - out_deg[i - 1]) for i in range(1, len(out_deg))),
        "phase_source": str(phase_source),
        "phase_reconstruction": phase_info,
    }


def apply(profile: CaseProfile) -> dict[str, object]:
    paths = stage_paths(profile)
    pose_info = _build_mug_pose(profile, paths["object_pose"])
    phase_info = _build_mug_phase(profile, paths["object_phase"])
    grasp_dir = profile.result_dir / "mug_grasp_anchor_state_internal"
    grasp_dir.mkdir(parents=True, exist_ok=True)
    vlm_keyframes = profile.sample_dir / "annotations/vlm_mug_contact_keyframes/mug_contact_keyframe_annotations.csv"
    contact_rows, contact_summary = build_state(paths["object_local_points"], vlm_keyframes)
    contact_rows, vlm_gate_summary = apply_contact_gate_to_rows(profile, contact_rows, stage="stage2")
    with Path(paths["object_contact_points"]).open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(contact_rows)
    internal_csv = grasp_dir / "mug_grasp_anchor_state.csv"
    internal_json = grasp_dir / "mug_grasp_anchor_state_summary.json"
    with internal_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(contact_rows)
    internal_json.write_text(json.dumps(contact_summary, indent=2) + "\n")
    metrics = {
        "component": "stable_grasp_anchor",
        "object_pose": str(paths["object_pose"]),
        "object_phase": str(paths["object_phase"]),
        "object_contact_points": str(paths["object_contact_points"]),
        "object_contact_points_source": str(internal_csv),
        "object_contact_points_builder": "generic_contact_pipeline/components/contact/build_mug_grasp_anchor_state.py",
        "hidden_handle_policy": "keep_previous_stable_grasp_anchor",
        "pose_policy": "m18_rotation_slerp_table_freeze_clean_python",
        "pose_info": pose_info,
        "phase_policy": "m17_phase_pchip_smooth_entry_no_hide_clean_python",
        "phase_info": phase_info,
        "grasp_anchor_summary": contact_summary,
        "vlm_gate_summary": vlm_gate_summary,
    }
    write_json(paths["stage4_metrics"], metrics)
    return metrics
