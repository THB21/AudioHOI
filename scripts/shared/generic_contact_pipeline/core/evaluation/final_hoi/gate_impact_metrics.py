from __future__ import annotations

import math
from typing import Any

from .schemas import EvaluationPaths, MetricBlock
from .utils import f, max_or_none, mean, read_rows, write_json, write_rows


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "1.0", "true", "yes", "pass", "enabled", "update", "propagate", "freeze"}


def _translation(row: dict[str, str]) -> tuple[float, float, float] | None:
    vals = [f(row.get(k)) for k in ("tx", "ty", "tz")]
    if any(v is None for v in vals):
        vals = [f(row.get(k)) for k in ("x", "y", "z")]
    if any(v is None for v in vals):
        return None
    return float(vals[0]), float(vals[1]), float(vals[2])


def _quat(row: dict[str, str]) -> tuple[float, float, float, float] | None:
    vals = [f(row.get(k)) for k in ("qw", "qx", "qy", "qz")]
    if any(v is None for v in vals):
        return None
    return float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])


def _dist(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((b[i] - a[i]) ** 2 for i in range(3)))


def _angle(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    an = math.sqrt(sum(v * v for v in a)) or 1.0
    bn = math.sqrt(sum(v * v for v in b)) or 1.0
    dot = abs(sum(a[i] * b[i] for i in range(4)) / (an * bn))
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


def _pose_deltas(pre_rows: list[dict[str, str]], final_rows: list[dict[str, str]]) -> tuple[list[float], list[float]]:
    final_by_frame = {str(row.get("frame", idx + 1)): row for idx, row in enumerate(final_rows)}
    t_deltas: list[float] = []
    r_deltas: list[float] = []
    for idx, pre in enumerate(pre_rows):
        final = final_by_frame.get(str(pre.get("frame", idx + 1)))
        if final is None:
            continue
        pt = _translation(pre)
        ft = _translation(final)
        if pt is not None and ft is not None:
            t_deltas.append(_dist(pt, ft))
        pq = _quat(pre)
        fq = _quat(final)
        if pq is not None and fq is not None:
            r_deltas.append(_angle(pq, fq))
    return t_deltas, r_deltas


def _gate_source_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        key = str(row.get("source_gate", row.get("pass_gate", ""))).strip() or "unknown"
        out[key] = out.get(key, 0) + 1
    return out


def _stage_gate_active(row: dict[str, str]) -> bool:
    gate = str(row.get("pass_gate", "")).strip().lower()
    repair = str(row.get("repair_action", "")).strip().lower()
    return (
        gate in {"reject", "unclear"}
        or _truthy(row.get("residual_reweight"))
        or _truthy(row.get("rerun_stage"))
        or repair
        not in {
            "",
            "keep_outputs",
            "no_effective_vlm_gate",
            "accept_candidate",
        }
    )


def compute_gate_impact_metrics(paths: EvaluationPaths) -> MetricBlock:
    trace_dir = paths.gate_trace_dir
    gate_rows = read_rows(trace_dir / "vlm_trace" / "04_gating" / "gate_timeline.csv")
    stage_gate_rows = read_rows(trace_dir / "stage_audit" / "stage_audit_gates.csv")
    optimizer_rows = read_rows(trace_dir / "optimizer_decisions.csv")
    anchor_rows = read_rows(trace_dir / "anchor_state.csv")
    residual_rows = read_rows(trace_dir / "physical_smooth_residuals.csv")
    pre_rows = read_rows(trace_dir / "object_pose_pre_smooth.csv")
    trace_final_rows = read_rows(trace_dir / "object_pose.csv")
    final_rows = trace_final_rows or read_rows(paths.object_pose_csv)

    t_deltas, r_deltas = _pose_deltas(pre_rows, final_rows) if pre_rows and final_rows else ([], [])
    residual_switch_fields = [
        "visual_residual_enabled",
        "contact_anchor_residual_enabled",
        "depth_residual_enabled",
        "geometry_residual_enabled",
        "velocity_residual_enabled",
        "acceleration_residual_enabled",
        "static_freeze_residual_enabled",
        "boundary_freeze_interpolation_enabled",
        "static_tail_freeze_enabled",
    ]
    residual_enabled_total = sum(_truthy(row.get(field)) for row in optimizer_rows for field in residual_switch_fields)
    reoptimized = sum(_truthy(row.get("feedback_reoptimized")) for row in optimizer_rows)
    reweighted = sum(bool(str(row.get("feedback_reweight_reason", "")).strip()) for row in optimizer_rows)
    freeze_frames = sum(
        _truthy(row.get("static_freeze_residual_enabled"))
        or _truthy(row.get("boundary_freeze_interpolation_enabled"))
        or _truthy(row.get("static_tail_freeze_enabled"))
        for row in optimizer_rows
    )
    anchor_updates = sum(_truthy(row.get("anchor_update_allowed")) for row in anchor_rows)
    anchor_blocks = sum(not _truthy(row.get("anchor_update_allowed")) for row in anchor_rows if row)
    pose_anchor_allowed = sum(_truthy(row.get("pose_anchor_allowed")) for row in anchor_rows)
    anchor_residual_frames = sum(_truthy(row.get("anchor_residual_enabled")) for row in residual_rows)
    gate_timeline_source = "frame_timeline" if gate_rows else ("stage_audit_fallback" if stage_gate_rows else "missing")
    if gate_rows:
        gate_event_count = len(gate_rows)
        gate_active = sum(_truthy(row.get("active")) for row in gate_rows)
        gate_reject_unclear = sum(str(row.get("source_gate", "")).strip().lower() in {"reject", "unclear"} for row in gate_rows)
        gate_source_counts = _gate_source_counts(gate_rows)
    else:
        gate_event_count = len(stage_gate_rows)
        gate_active = sum(_stage_gate_active(row) for row in stage_gate_rows)
        gate_reject_unclear = sum(str(row.get("pass_gate", "")).strip().lower() in {"reject", "unclear"} for row in stage_gate_rows)
        gate_source_counts: dict[str, int] = {}
        for row in stage_gate_rows:
            key = str(row.get("pass_gate", "")).strip() or "unknown"
            gate_source_counts[key] = gate_source_counts.get(key, 0) + 1
    stage_residual_reweights = sum(_truthy(row.get("residual_reweight")) for row in stage_gate_rows)

    missing = []
    if not gate_rows and not stage_gate_rows:
        missing.append("missing_gate_timeline")
    if not optimizer_rows:
        missing.append("missing_optimizer_decisions")
    if not anchor_rows:
        missing.append("missing_anchor_state")
    if not residual_rows:
        missing.append("missing_physical_smooth_residuals")
    if not pre_rows:
        missing.append("missing_object_pose_pre_smooth")
    if not final_rows:
        missing.append("missing_object_pose")

    ok_status = "ok:stage_audit_fallback" if gate_timeline_source == "stage_audit_fallback" else "ok"
    metrics: dict[str, Any] = {
        "gate_impact_status": ok_status if not missing else "partial:" + "|".join(missing),
        "gate_timeline_source": gate_timeline_source,
        "gate_event_count": gate_event_count,
        "gate_active_count": gate_active,
        "gate_reject_unclear_count": gate_reject_unclear,
        "gate_source_counts": ";".join(f"{k}:{v}" for k, v in sorted(gate_source_counts.items())),
        "stage_audit_gate_count": len(stage_gate_rows),
        "stage_residual_reweight_count": stage_residual_reweights,
        "optimizer_reoptimized_frames": reoptimized,
        "optimizer_reweighted_frames": reweighted,
        "residual_enabled_total": residual_enabled_total,
        "anchor_update_allowed_count": anchor_updates,
        "anchor_update_blocked_count": anchor_blocks,
        "pose_anchor_allowed_count": pose_anchor_allowed,
        "anchor_residual_enabled_frames": anchor_residual_frames,
        "freeze_interpolation_frames": freeze_frames,
        "pose_delta_translation_mean_m": mean(t_deltas),
        "pose_delta_translation_max_m": max_or_none(t_deltas),
        "pose_delta_rotation_mean_rad": mean(r_deltas),
        "pose_delta_rotation_max_rad": max_or_none(r_deltas),
        "pose_delta_frame_count": len(t_deltas),
        "gate_trace_source": str(trace_dir),
        "pose_delta_source": "gate_trace_pre_smooth_to_gate_trace_final" if trace_final_rows else "gate_trace_pre_smooth_to_evaluated_final_pose",
        "source_gate_timeline": str(trace_dir / "vlm_trace" / "04_gating" / "gate_timeline.csv"),
        "source_optimizer_decisions": str(trace_dir / "optimizer_decisions.csv"),
    }
    out_json = write_json(paths.evaluation_dir / "gate_impact_metrics.json", metrics)
    out_csv = write_rows(paths.evaluation_dir / "gate_impact_metrics.csv", [metrics])
    return MetricBlock("gate_impact", metrics, {"json": str(out_json), "csv": str(out_csv)}, missing)
