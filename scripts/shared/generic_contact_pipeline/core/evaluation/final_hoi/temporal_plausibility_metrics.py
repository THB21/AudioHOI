from __future__ import annotations

import json
import math
from typing import Any

from .schemas import EvaluationPaths, MetricBlock
from .utils import f, max_or_none, mean, read_rows, write_json, write_rows


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


def _norm3(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((b[i] - a[i]) ** 2 for i in range(3)))


def _quat_angle(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    an = math.sqrt(sum(v * v for v in a)) or 1.0
    bn = math.sqrt(sum(v * v for v in b)) or 1.0
    dot = abs(sum(a[i] * b[i] for i in range(4)) / (an * bn))
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


def _median(values: list[float]) -> float | None:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return None
    n = len(vals)
    mid = n // 2
    return vals[mid] if n % 2 else 0.5 * (vals[mid - 1] + vals[mid])


def _percentile(values: list[float], q: float) -> float | None:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return None
    idx = min(len(vals) - 1, max(0, round((len(vals) - 1) * q)))
    return vals[idx]


def _robust_threshold(values: list[float], *, floor: float, q: float = 0.95) -> float | None:
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return None
    med = _median(vals) or 0.0
    mad = _median([abs(v - med) for v in vals]) or 0.0
    perc = _percentile(vals, q) or 0.0
    return max(floor, med + 3.0 * mad, perc)


def _event_frames(paths: EvaluationPaths, n_frames: int) -> set[int]:
    rows = read_rows(paths.audio_contact_csv)
    frames: set[int] = set()
    for row in rows:
        for key in ("refined_frame", "frame"):
            value = f(row.get(key))
            if value is None:
                continue
            frame = int(round(value))
            if 1 <= frame <= n_frames:
                frames.add(frame)
                break
    return frames


def _window(frames: set[int], n_frames: int, radius: int = 3) -> set[int]:
    out: set[int] = set()
    for frame in frames:
        for candidate in range(frame - radius, frame + radius + 1):
            if 1 <= candidate <= n_frames:
                out.add(candidate)
    return out


def _intervals(frames: list[int]) -> str:
    if not frames:
        return "[]"
    frames = sorted(set(frames))
    out: list[dict[str, int]] = []
    start = prev = frames[0]
    for frame in frames[1:]:
        if frame == prev + 1:
            prev = frame
            continue
        out.append({"start": start, "end": prev})
        start = prev = frame
    out.append({"start": start, "end": prev})
    return json.dumps(out, separators=(",", ":"))


def compute_temporal_plausibility_metrics(paths: EvaluationPaths) -> MetricBlock:
    rows = read_rows(paths.object_pose_csv)
    n = len(rows)
    if n < 4:
        metrics: dict[str, Any] = {
            "temporal_metric_status": "missing_or_too_short_pose",
            "translation_spike_count": None,
            "rotation_spike_count": None,
            "event_aligned_spike_count": None,
            "non_event_spike_count": None,
            "high_speed_recall": None,
            "oversmooth_rate": None,
            "static_tail_drift_m": None,
            "temporal_failure_intervals": "[]",
        }
        return MetricBlock(
            "temporal_plausibility",
            metrics,
            {
                "json": str(write_json(paths.evaluation_dir / "temporal_plausibility_metrics.json", metrics)),
                "csv": str(write_rows(paths.evaluation_dir / "temporal_plausibility_metrics.csv", [metrics])),
            },
            ["temporal_plausibility_missing_pose"],
        )

    translations = [_translation(row) for row in rows]
    quats = [_quat(row) for row in rows]
    t_vel: list[float] = []
    r_vel: list[float] = []
    vel_frames: list[int] = []
    for idx in range(1, n):
        if translations[idx - 1] is not None and translations[idx] is not None:
            t_vel.append(_norm3(translations[idx - 1], translations[idx]))  # type: ignore[arg-type]
            vel_frames.append(idx + 1)
        if quats[idx - 1] is not None and quats[idx] is not None:
            r_vel.append(_quat_angle(quats[idx - 1], quats[idx]))  # type: ignore[arg-type]

    t_accel = [abs(t_vel[i] - t_vel[i - 1]) for i in range(1, len(t_vel))]
    r_accel = [abs(r_vel[i] - r_vel[i - 1]) for i in range(1, len(r_vel))]
    accel_frames = vel_frames[1:]
    t_thr = _robust_threshold(t_accel, floor=0.02)
    r_thr = _robust_threshold(r_accel, floor=0.05)
    event_frames = _event_frames(paths, n)
    event_window = _window(event_frames, n)

    translation_spikes = [frame for frame, value in zip(accel_frames, t_accel) if t_thr is not None and value > t_thr]
    rotation_spikes = [frame for frame, value in zip(accel_frames, r_accel) if r_thr is not None and value > r_thr]
    all_spikes = sorted(set(translation_spikes) | set(rotation_spikes))
    event_aligned = [frame for frame in all_spikes if frame in event_window]
    non_event = [frame for frame in all_spikes if frame not in event_window]

    event_high_speed = 0
    event_preserved = 0
    event_oversmooth = 0
    accel_floor = _percentile(t_accel, 0.75) or 0.0
    for frame in event_frames:
        window_values = [value for fidx, value in zip(accel_frames, t_accel) if abs(fidx - frame) <= 3]
        if not window_values:
            continue
        event_high_speed += 1
        peak = max(window_values)
        if peak >= accel_floor:
            event_preserved += 1
        else:
            event_oversmooth += 1

    static_tail_drift = None
    tail_n = max(5, min(30, n // 10))
    tail = [p for p in translations[-tail_n:] if p is not None]
    tail_vel = t_vel[-max(1, tail_n - 1) :]
    tail_median_vel = _median(tail_vel) or 0.0
    low_motion_threshold = max(0.01, (_median(t_vel) or 0.0) * 0.5)
    if len(tail) >= 2 and tail_median_vel <= low_motion_threshold:
        ref = tail[0]
        static_tail_drift = max(_norm3(ref, p) for p in tail[1:])

    metrics = {
        "temporal_metric_status": "ok",
        "translation_spike_count": len(translation_spikes),
        "rotation_spike_count": len(rotation_spikes),
        "event_aligned_spike_count": len(event_aligned),
        "non_event_spike_count": len(non_event),
        "translation_spike_threshold": t_thr,
        "rotation_spike_threshold_rad": r_thr,
        "translation_acceleration_max": max_or_none(t_accel),
        "rotation_acceleration_max": max_or_none(r_accel),
        "high_speed_event_windows": event_high_speed,
        "high_speed_recall": (event_preserved / event_high_speed) if event_high_speed else None,
        "oversmooth_rate": (event_oversmooth / event_high_speed) if event_high_speed else None,
        "static_tail_drift_m": static_tail_drift,
        "temporal_failure_intervals": _intervals(non_event),
        "event_frame_count": len(event_frames),
        "event_window_frame_count": len(event_window),
        "temporal_accel_at_events": mean(value for frame, value in zip(accel_frames, t_accel) if frame in event_window),
        "temporal_accel_in_flight": mean(value for frame, value in zip(accel_frames, t_accel) if frame not in event_window),
        "source_pose": str(paths.object_pose_csv),
        "source_events": str(paths.audio_contact_csv) if paths.audio_contact_csv.exists() else "missing_audio_contact_csv",
    }
    out_json = write_json(paths.evaluation_dir / "temporal_plausibility_metrics.json", metrics)
    out_csv = write_rows(paths.evaluation_dir / "temporal_plausibility_metrics.csv", [metrics])
    return MetricBlock("temporal_plausibility", metrics, {"json": str(out_json), "csv": str(out_csv)})
