from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def _f(row: dict[str, str], key: str, default: float = math.nan) -> float:
    try:
        value = row.get(key, "")
        return default if value in {"", None} else float(value)
    except Exception:
        return default


def _frame(row: dict[str, str]) -> int:
    return int(float(row.get("frame", "0") or 0))


def _rotation_from_row(row: dict[str, str]) -> Rotation:
    q = np.asarray([_f(row, "qx", 0.0), _f(row, "qy", 0.0), _f(row, "qz", 0.0), _f(row, "qw", 1.0)], dtype=float)
    if np.linalg.norm(q) < 1e-8:
        q = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=float)
    return Rotation.from_quat(q / np.linalg.norm(q))


def _quat_strings(rot: Rotation) -> tuple[str, str, str, str]:
    qx, qy, qz, qw = rot.as_quat()
    return f"{qw:.6f}", f"{qx:.6f}", f"{qy:.6f}", f"{qz:.6f}"


def _canonical_rotvecs(rows: list[dict[str, str]]) -> np.ndarray:
    rots: list[Rotation] = []
    prev_q: np.ndarray | None = None
    for row in rows:
        q = _rotation_from_row(row).as_quat()
        if prev_q is not None and float(np.dot(prev_q, q)) < 0.0:
            q = -q
        prev_q = q
        rots.append(Rotation.from_quat(q))
    if not rots:
        return np.zeros((0, 3), dtype=float)
    raw = np.asarray([rot.as_rotvec() for rot in rots], dtype=float)
    out = raw.copy()
    for i in range(1, len(out)):
        best = out[i]
        best_norm = float(np.linalg.norm(best - out[i - 1]))
        axis_norm = float(np.linalg.norm(raw[i]))
        if axis_norm > 1e-8:
            axis = raw[i] / axis_norm
            for k in (-2, -1, 1, 2):
                cand = raw[i] + axis * (2.0 * math.pi * k)
                dist = float(np.linalg.norm(cand - out[i - 1]))
                if dist < best_norm:
                    best = cand
                    best_norm = dist
        out[i] = best
    return out


def _optimize_axis(
    init: np.ndarray,
    trust_weight: np.ndarray,
    *,
    smooth_weight: np.ndarray | None = None,
    accel_weight: np.ndarray | None = None,
    axis_scale: float,
    prior_w: float,
    smooth_w: float,
    accel_w: float,
) -> np.ndarray:
    if len(init) < 3:
        return init.copy()
    smooth = np.ones(max(0, len(init) - 1), dtype=float) if smooth_weight is None else np.asarray(smooth_weight, dtype=float)
    accel = np.ones(max(0, len(init) - 2), dtype=float) if accel_weight is None else np.asarray(accel_weight, dtype=float)
    if smooth.shape[0] != max(0, len(init) - 1):
        raise ValueError("smooth_weight length must be len(rows)-1")
    if accel.shape[0] != max(0, len(init) - 2):
        raise ValueError("accel_weight length must be len(rows)-2")

    def residual(x: np.ndarray) -> np.ndarray:
        vals: list[float] = []
        vals.extend(((x - init) / axis_scale * prior_w * trust_weight).tolist())
        if len(x) > 1:
            vals.extend((np.diff(x) / axis_scale * smooth_w * smooth).tolist())
        if len(x) > 2:
            vals.extend((np.diff(x, n=2) / axis_scale * accel_w * accel).tolist())
        return np.asarray(vals, dtype=float)

    res = least_squares(residual, init.copy(), loss="soft_l1", f_scale=0.04, max_nfev=160)
    return res.x


def _optimize_joint_se3(
    trans0: np.ndarray,
    rot0: np.ndarray,
    trust: np.ndarray,
    smooth: np.ndarray,
    accel: np.ndarray,
    anchor_constraints: Sequence[Sequence[tuple[Sequence[float], Sequence[float], float]]],
    *,
    translation_scales: tuple[float, float, float],
    rotation_scale: float,
    prior_w: float,
    smooth_w: float,
    accel_w: float,
    anchor_w: float,
) -> tuple[np.ndarray, np.ndarray, list[float], list[int]]:
    n = len(trans0)
    if n == 0:
        return trans0.copy(), rot0.copy(), [], []
    scales_t = np.asarray(translation_scales, dtype=float)
    x0 = np.concatenate([trans0.reshape(-1), rot0.reshape(-1)])

    def unpack(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return x[: 3 * n].reshape(n, 3), x[3 * n :].reshape(n, 3)

    def residual(x: np.ndarray) -> np.ndarray:
        trans, rotvec = unpack(x)
        vals: list[float] = []
        vals.extend((((trans - trans0) / scales_t) * (prior_w * trust[:, None])).reshape(-1).tolist())
        vals.extend(((rotvec - rot0) / rotation_scale * (prior_w * trust[:, None])).reshape(-1).tolist())
        if n > 1:
            vals.extend(((np.diff(trans, axis=0) / scales_t) * (smooth_w * smooth[:, None])).reshape(-1).tolist())
            vals.extend((np.diff(rotvec, axis=0) / rotation_scale * (smooth_w * smooth[:, None])).reshape(-1).tolist())
        if n > 2:
            vals.extend(((np.diff(trans, n=2, axis=0) / scales_t) * (accel_w * accel[:, None])).reshape(-1).tolist())
            vals.extend((np.diff(rotvec, n=2, axis=0) / rotation_scale * (accel_w * accel[:, None])).reshape(-1).tolist())
        for i, frame_constraints in enumerate(anchor_constraints):
            if not frame_constraints:
                continue
            rot = Rotation.from_rotvec(rotvec[i]).as_matrix()
            for local, target, weight in frame_constraints:
                local_arr = np.asarray(local, dtype=float)
                target_arr = np.asarray(target, dtype=float)
                if local_arr.shape != (3,) or target_arr.shape != (3,):
                    continue
                if not (np.isfinite(local_arr).all() and np.isfinite(target_arr).all()):
                    continue
                pred = trans[i] + rot @ local_arr
                vals.extend(((pred - target_arr) / 0.075 * anchor_w * max(0.05, float(weight))).tolist())
        return np.asarray(vals, dtype=float)

    res = least_squares(residual, x0, loss="soft_l1", f_scale=0.08, max_nfev=220)
    trans, rotvec = unpack(res.x)
    anchor_error: list[float] = []
    anchor_count: list[int] = []
    for i, frame_constraints in enumerate(anchor_constraints):
        errs: list[float] = []
        if frame_constraints:
            rot = Rotation.from_rotvec(rotvec[i]).as_matrix()
            for local, target, _weight in frame_constraints:
                local_arr = np.asarray(local, dtype=float)
                target_arr = np.asarray(target, dtype=float)
                if local_arr.shape == (3,) and target_arr.shape == (3,) and np.isfinite(local_arr).all() and np.isfinite(target_arr).all():
                    errs.append(float(np.linalg.norm(trans[i] + rot @ local_arr - target_arr)))
        anchor_error.append(float(np.median(errs)) if errs else 0.0)
        anchor_count.append(len(errs))
    return trans, rotvec, anchor_error, anchor_count


def smooth_quaternion_pose_sequence(
    rows: list[dict[str, str]],
    *,
    trust_weight: Sequence[float] | None = None,
    smooth_weight: Sequence[float] | None = None,
    accel_weight: Sequence[float] | None = None,
    anchor_constraints: Sequence[Sequence[tuple[Sequence[float], Sequence[float], float]]] | None = None,
    translation_scales: tuple[float, float, float] = (0.065, 0.065, 0.095),
    rotation_scale: float = 0.070,
    prior_w: float = 0.30,
    smooth_w: float = 1.35,
    accel_w: float = 5.8,
    anchor_w: float = 1.0,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if not rows:
        return [], []
    n = len(rows)
    trust = np.asarray(list(trust_weight) if trust_weight is not None else [1.0] * n, dtype=float)
    if trust.shape[0] != n:
        raise ValueError("trust_weight length must match rows")
    trust = np.clip(trust, 0.0, 8.0)
    smooth = np.asarray(list(smooth_weight) if smooth_weight is not None else [1.0] * max(0, n - 1), dtype=float)
    accel = np.asarray(list(accel_weight) if accel_weight is not None else [1.0] * max(0, n - 2), dtype=float)
    if smooth.shape[0] != max(0, n - 1):
        raise ValueError("smooth_weight length must be len(rows)-1")
    if accel.shape[0] != max(0, n - 2):
        raise ValueError("accel_weight length must be len(rows)-2")
    trans0 = np.asarray([[_f(row, "tx", 0.0), _f(row, "ty", 0.0), _f(row, "tz", 3.0)] for row in rows], dtype=float)
    rot0 = _canonical_rotvecs(rows)
    anchor_constraints = anchor_constraints or [[] for _ in range(n)]
    if len(anchor_constraints) != n:
        raise ValueError("anchor_constraints length must match rows")
    if any(anchor_constraints):
        trans, rotvec, anchor_error, anchor_count = _optimize_joint_se3(
            trans0,
            rot0,
            trust,
            smooth,
            accel,
            anchor_constraints,
            translation_scales=translation_scales,
            rotation_scale=rotation_scale,
            prior_w=prior_w,
            smooth_w=smooth_w,
            accel_w=accel_w,
            anchor_w=anchor_w,
        )
    else:
        trans = trans0.copy()
        rotvec = rot0.copy()
        anchor_error = [0.0] * n
        anchor_count = [0] * n
        for j, scale in enumerate(translation_scales):
            trans[:, j] = _optimize_axis(
                trans0[:, j],
                trust,
                smooth_weight=smooth,
                accel_weight=accel,
                axis_scale=scale,
                prior_w=prior_w,
                smooth_w=smooth_w,
                accel_w=accel_w,
            )
        for j in range(3):
            rotvec[:, j] = _optimize_axis(
                rot0[:, j],
                trust,
                smooth_weight=smooth,
                accel_weight=accel,
                axis_scale=rotation_scale,
                prior_w=prior_w,
                smooth_w=smooth_w,
                accel_w=accel_w,
            )

    out_rows: list[dict[str, str]] = []
    audit_rows: list[dict[str, str]] = []
    vel0 = np.zeros(n, dtype=float)
    acc0 = np.zeros(n, dtype=float)
    if n > 1:
        vel0[1:] = np.linalg.norm(np.diff(trans0, axis=0), axis=1) + np.linalg.norm(np.diff(rot0, axis=0), axis=1) * 0.10
    if n > 2:
        acc0[1:-1] = np.linalg.norm(np.diff(trans0, n=2, axis=0), axis=1) + np.linalg.norm(np.diff(rot0, n=2, axis=0), axis=1) * 0.10
    vel = np.zeros(n, dtype=float)
    acc = np.zeros(n, dtype=float)
    if n > 1:
        vel[1:] = np.linalg.norm(np.diff(trans, axis=0), axis=1) + np.linalg.norm(np.diff(rotvec, axis=0), axis=1) * 0.10
    if n > 2:
        acc[1:-1] = np.linalg.norm(np.diff(trans, n=2, axis=0), axis=1) + np.linalg.norm(np.diff(rotvec, n=2, axis=0), axis=1) * 0.10

    raw_vel_thr = max(float(np.percentile(vel0, 75) + 2.5 * np.percentile(np.abs(vel0 - np.median(vel0)), 75)), 0.08)
    raw_acc_thr = max(float(np.percentile(acc0, 75) + 2.5 * np.percentile(np.abs(acc0 - np.median(acc0)), 75)), 0.10)
    vel_thr = max(float(np.percentile(vel, 75) + 2.5 * np.percentile(np.abs(vel - np.median(vel)), 75)), 0.08)
    acc_thr = max(float(np.percentile(acc, 75) + 2.5 * np.percentile(np.abs(acc - np.median(acc)), 75)), 0.10)

    for i, row in enumerate(rows):
        out = dict(row)
        rot = Rotation.from_rotvec(rotvec[i])
        qw, qx, qy, qz = _quat_strings(rot)
        out.update(
            {
                "tx": f"{trans[i, 0]:.6f}",
                "ty": f"{trans[i, 1]:.6f}",
                "tz": f"{max(0.5, trans[i, 2]):.6f}",
                "qw": qw,
                "qx": qx,
                "qy": qy,
                "qz": qz,
                "source": (row.get("source", "") + "+sequence_se3_optimizer").strip("+"),
            }
        )
        out_rows.append(out)
        delta_t = float(np.linalg.norm(trans[i] - trans0[i]))
        delta_r = float((Rotation.from_rotvec(rot0[i]).inv() * rot).magnitude())
        raw_velocity_spike = vel0[i] > raw_vel_thr
        raw_accel_spike = acc0[i] > raw_acc_thr
        audit_rows.append(
            {
                "frame": str(_frame(row)),
                "se3_velocity_spike": "1" if vel[i] > vel_thr else "0",
                "se3_accel_spike": "1" if acc[i] > acc_thr else "0",
                "raw_se3_velocity_spike": "1" if raw_velocity_spike else "0",
                "raw_se3_accel_spike": "1" if raw_accel_spike else "0",
                "raw_velocity_score": f"{vel0[i]:.6f}",
                "raw_accel_score": f"{acc0[i]:.6f}",
                "final_velocity_score": f"{vel[i]:.6f}",
                "final_accel_score": f"{acc[i]:.6f}",
                "se3_smoothed": "1" if delta_t > 1e-5 or delta_r > 1e-5 else "0",
                "delta_translation_m": f"{delta_t:.6f}",
                "delta_rotation_rad": f"{delta_r:.6f}",
                "trust_weight": f"{trust[i]:.6f}",
                "anchor_residual_enabled": "1" if anchor_count[i] else "0",
                "anchor_residual_count": str(anchor_count[i]),
                "anchor_residual_m": f"{anchor_error[i]:.6f}",
            }
        )
    return out_rows, audit_rows
