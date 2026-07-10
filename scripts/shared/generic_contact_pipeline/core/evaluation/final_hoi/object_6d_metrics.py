from __future__ import annotations

import math

from .schemas import EvaluationPaths, MetricBlock
from .utils import f, mean, max_or_none, read_rows, write_json, write_rows


def _quat_norm(row: dict[str, str]) -> float | None:
    vals = [f(row.get(k)) for k in ("qw", "qx", "qy", "qz")]
    if any(v is None for v in vals):
        return None
    return math.sqrt(sum(float(v) * float(v) for v in vals))


def _translation(row: dict[str, str]) -> tuple[float, float, float] | None:
    vals = [f(row.get(k)) for k in ("tx", "ty", "tz")]
    if any(v is None for v in vals):
        return None
    return float(vals[0]), float(vals[1]), float(vals[2])


def _quat(row: dict[str, str]) -> tuple[float, float, float, float] | None:
    vals = [f(row.get(k)) for k in ("qw", "qx", "qy", "qz")]
    if any(v is None for v in vals):
        return None
    return float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])


def _translation_steps(rows: list[dict[str, str]]) -> list[float]:
    pts = [_translation(row) for row in rows]
    out: list[float] = []
    for a, b in zip(pts, pts[1:]):
        if a is None or b is None:
            continue
        out.append(math.sqrt(sum((b[i] - a[i]) ** 2 for i in range(3))))
    return out


def _rotation_steps(rows: list[dict[str, str]]) -> list[float]:
    qs = [_quat(row) for row in rows]
    out: list[float] = []
    for a, b in zip(qs, qs[1:]):
        if a is None or b is None:
            continue
        an = math.sqrt(sum(v * v for v in a)) or 1.0
        bn = math.sqrt(sum(v * v for v in b)) or 1.0
        dot = abs(sum(a[i] * b[i] for i in range(4)) / (an * bn))
        out.append(2.0 * math.acos(max(-1.0, min(1.0, dot))))
    return out


def _second_diff(values: list[float]) -> list[float]:
    return [abs(values[i] - values[i - 1]) for i in range(1, len(values))]


def compute_object_6d_metrics(paths: EvaluationPaths) -> MetricBlock:
    pose_path = paths.object_pose_csv
    rows = read_rows(pose_path)
    fields = set(rows[0]) if rows else set()
    se3_valid = {"tx", "ty", "tz", "qw", "qx", "qy", "qz"}.issubset(fields)
    n = len(rows)
    trans_valid = sum(1 for row in rows if _translation(row) is not None)
    rot_norms = [_quat_norm(row) for row in rows]
    rot_valid = sum(1 for value in rot_norms if value is not None and abs(value - 1.0) < 0.05)
    t_steps = _translation_steps(rows)
    r_steps = _rotation_steps(rows)
    length_values = [f(row.get("length_m"), f(row.get("object_length_m"), f(row.get("geometry_length_m")))) for row in rows]
    jumps = read_rows(paths.result_dir / "pose_jump_audit.csv")
    jump_count = (
        sum(
            1
            for row in jumps
            if any(str(row.get(key, "")).lower() in {"1", "1.0", "true"} for key in ("visual_spike", "contact_spike", "smoothness_spike"))
        )
        if jumps
        else None
    )
    static_drift = max_or_none(f(row.get("static_tail_drift_m"), f(row.get("static_drift_m"))) for row in jumps) if jumps else None
    metrics = {
        "se3_valid": se3_valid,
        "n_frames": n,
        "translation_valid_rate": trans_valid / n if n else 0.0,
        "rotation_valid_rate": rot_valid / n if n else 0.0,
        "quat_norm_error_mean": mean(abs((value or 1.0) - 1.0) for value in rot_norms) or 0.0,
        "translation_velocity_mean": mean(t_steps),
        "translation_velocity_max": max_or_none(t_steps),
        "translation_acceleration_mean": mean(_second_diff(t_steps)),
        "rotation_velocity_mean": mean(r_steps),
        "rotation_velocity_max": max_or_none(r_steps),
        "rotation_acceleration_mean": mean(_second_diff(r_steps)),
        "jump_count": jump_count,
        "static_tail_drift_m": static_drift,
        "geometry_spread_m": (max(v for v in length_values if v is not None) - min(v for v in length_values if v is not None))
        if any(v is not None for v in length_values)
        else None,
    }
    out_json = write_json(paths.evaluation_dir / "object_6d_metrics.json", metrics)
    out_csv = write_rows(paths.evaluation_dir / "object_6d_metrics.csv", [metrics])
    return MetricBlock("object_6d", metrics, {"json": str(out_json), "csv": str(out_csv)})
