from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from ..base.config import CaseProfile
from ..base.io import repo_relative_value
from ..measurements import Line2DMeasurement, MetricDepthMeasurement, Point2DMeasurement, adapt_configured_supplemental_measurements, adapt_legacy_observation_rows
from .types import (
    FactorEnergySummary,
    FactorGap,
    FactorInputRef,
    FactorKind,
    FactorSourceRef,
    FactorSpec,
)


ENERGY_TO_KIND = {
    "E_2d": FactorKind.POINT_REPROJECTION,
    "E_visual": FactorKind.POINT_REPROJECTION,
    "E_mask": FactorKind.MASK_SILHOUETTE,
    "E_depth": FactorKind.METRIC_DEPTH,
    "E_contact": FactorKind.CONTACT_DISTANCE,
    "E_support": FactorKind.SUPPORT_AND_PENETRATION,
    "E_penetration": FactorKind.SUPPORT_AND_PENETRATION,
    "E_smooth": FactorKind.TEMPORAL_VELOCITY,
    "E_temporal": FactorKind.TEMPORAL_ACCELERATION,
    "E_static": FactorKind.STATIC_FREEZE,
    "E_audio": FactorKind.AUDIO_EVENT_PRIOR,
    "E_prior": FactorKind.POSE_PRIOR,
    "E_reg": FactorKind.REGULARIZATION,
}
SUPPORTED_ENERGY_TERMS = tuple(ENERGY_TO_KIND)
AUDIT_ALIAS_TO_CANONICAL = {
    "E_2d": "E_visual",
    "E_depth": "E_support",
    "E_temporal": "E_smooth",
    "E_prior": "E_reg",
}


@dataclass(frozen=True)
class FactorAdaptationResult:
    factors: tuple[FactorSpec, ...]
    energy_summaries: tuple[FactorEnergySummary, ...]
    gaps: tuple[FactorGap, ...]
    mapped_fields: tuple[str, ...]
    unmapped_nonempty_fields: tuple[str, ...]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _number(value: str | None) -> float | None:
    if value in {"", None}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _nonempty_fields(rows: list[dict[str, str]]) -> set[str]:
    if not rows:
        return set()
    return {field for field in rows[0] if any(row.get(field, "") not in {"", None} for row in rows)}


def _active_count(rows: list[dict[str, str]], field: str) -> int:
    return sum(1 for row in rows if (_number(row.get(field)) or 0.0) != 0.0)


def _has_active_support_evidence(rows: list[dict[str, str]]) -> bool:
    truthy = {"1", "1.0", "true", "yes", "active"}
    for row in rows:
        direct = any(str(row.get(field, "")).strip().lower() in truthy for field in ("floor_contact_state", "plane_support_state"))
        contact_active = str(row.get("contact_active", "")).strip().lower() in truthy
        semantic_support = row.get("human_part", "").strip().lower() == "floor" or row.get("object_part", "").strip().lower() == "floor_support"
        if direct or (contact_active and semantic_support):
            return True
    return False


def _energy_total(rows: list[dict[str, str]], field: str) -> float:
    return sum(_number(row.get(field)) or 0.0 for row in rows)


def _columns_are_identical(rows: list[dict[str, str]], left: str, right: str) -> bool:
    return bool(rows) and all(row.get(left, "") == row.get(right, "") for row in rows)


def _source(path: Path, fields: tuple[str, ...], producer: str) -> FactorSourceRef:
    return FactorSourceRef(str(repo_relative_value(path)), fields, producer)


def _input_refs(kind: FactorKind) -> tuple[FactorInputRef, ...]:
    refs: list[FactorInputRef] = [FactorInputRef("state", "StateSpec", "root")]
    if kind in {FactorKind.POINT_REPROJECTION, FactorKind.MASK_SILHOUETTE, FactorKind.LINE_REPROJECTION}:
        refs.append(FactorInputRef("measurement", "MeasurementIR", "visual_observation"))
    if kind in {FactorKind.METRIC_DEPTH, FactorKind.DEPTH_ORDER}:
        refs.append(FactorInputRef("measurement", "MeasurementIR", "depth_observation"))
    if kind in {FactorKind.CONTACT_DISTANCE, FactorKind.SUPPORT_AND_PENETRATION}:
        refs.append(FactorInputRef("constraint", "ContactConstraintIR", "contact_or_support"))
    if kind == FactorKind.AUDIO_EVENT_PRIOR:
        refs.append(FactorInputRef("measurement", "AudioEventIR", "audio_events"))
        refs.append(FactorInputRef("constraint", "ContactConstraintIR", "audio_contact_phase"))
    if kind == FactorKind.PERIODIC_PHASE_PRIOR:
        refs.append(FactorInputRef("measurement", "MeasurementIR", "periodic_feature_observation"))
        refs.append(FactorInputRef("state", "StateSpec", "body_yaw_zero_observable_axial_angle_in_phase"))
    if kind == FactorKind.GAUGE_CONSTRAINT:
        refs.append(FactorInputRef("state", "StateSpec", "gauge_constraints"))
    return tuple(refs)


def _weight_source(term: str) -> str:
    return "legacy_non_invasive_loss_audit_term:" + term


def _factor_id(term: str, kind: FactorKind) -> str:
    return f"{kind.value}:{term}"


def _stage_factor_id(stage_label: str, term: str, kind: FactorKind) -> str:
    return f"{kind.value}:{stage_label}:{term}"


def _chair_private_solver_gap_is_resolved(profile: CaseProfile, result_dir: Path) -> bool:
    """Return true only when chair Stage4 provenance proves current-run generic readiness."""
    if profile.case_name != "chair":
        return False
    try:
        from ..solver.chair_diagnostics import build_chair_contact_diagnostics, validate_chair_contact_diagnostics

        diagnostics = build_chair_contact_diagnostics(result_dir)
    except Exception:
        return False
    return diagnostics.get("compatibility_gap_status") == "nonblocking" and not validate_chair_contact_diagnostics(diagnostics)


def _mug_periodic_phase_prior_is_resolved(result_dir: Path) -> tuple[bool, int, dict[str, object]]:
    seed_dir = result_dir / "observation_seed"
    phase_csv = seed_dir / "axial_phase.csv"
    report_json = seed_dir / "observation_seed_report.json"
    if not phase_csv.exists() or not report_json.exists():
        return False, 0, {}
    report = _read_json(report_json)
    if report.get("historical_solved_seed_used") is not False:
        return False, 0, report
    if report.get("policy") != "observation_derived_body_pose_and_axial_phase":
        return False, 0, report
    phase = report.get("phase", {})
    if not isinstance(phase, dict) or phase.get("phase_gauge") != "body_yaw_zero_observable_axial_angle_in_phase":
        return False, 0, report
    phase_rows = _read_csv(phase_csv)
    if not phase_rows:
        return False, 0, report
    return True, len(phase_rows), report


def adapt_factor_rows(profile: CaseProfile, result_dir: Path) -> FactorAdaptationResult:
    loss_dir = result_dir / "loss_analysis"
    per_frame = loss_dir / "per_frame_residuals.csv"
    loss_trace = loss_dir / "loss_trace.csv"
    per_frame_rows = _read_csv(per_frame)
    trace_rows = _read_csv(loss_trace)
    loss_summary = _read_json(loss_dir / "loss_summary.json")
    stage3_metrics = _read_json(result_dir / "stage3_metrics.json")
    stage4_metrics = _read_json(result_dir / "stage4_metrics.json")
    contact_csv = result_dir / "object_contact_points.csv"
    contact_rows = _read_csv(contact_csv)
    observation_csv = result_dir / "object_observations.csv"
    observation_rows = _read_csv(observation_csv)
    has_support_evidence = _has_active_support_evidence(contact_rows)
    active_terms = set(loss_summary.get("active_terms", [])) if isinstance(loss_summary.get("active_terms", []), list) else set()

    mapped = {"frame", "time", "source", "E_total"}
    factors: list[FactorSpec] = []
    summaries: list[FactorEnergySummary] = []
    for term, kind in ENERGY_TO_KIND.items():
        if not per_frame_rows or term not in per_frame_rows[0]:
            continue
        canonical_term = AUDIT_ALIAS_TO_CANONICAL.get(term)
        if canonical_term and _columns_are_identical(per_frame_rows, term, canonical_term):
            mapped.add(term)
            continue
        total = _energy_total(per_frame_rows, term)
        active = _active_count(per_frame_rows, term)
        if term == "E_support" and not has_support_evidence:
            mapped.add(term)
            continue
        if kind == FactorKind.AUDIO_EVENT_PRIOR and term in active_terms and active == 0 and trace_rows:
            active = int(float(trace_rows[-1].get("active_audio_frames", "0") or 0))
        if total == 0.0 and active == 0:
            mapped.add(term)
            continue
        mapped.add(term)
        summaries.append(FactorEnergySummary(term, kind, active, total))
        factors.append(
            FactorSpec(
                factor_id=_factor_id(term, kind),
                kind=kind,
                frame_count=active,
                input_refs=_input_refs(kind),
                residual_unit="legacy_energy",
                weight_source=_weight_source(term),
                gate_source=(
                    "audio/contact/static gates in per_frame_residuals.csv"
                    if kind == FactorKind.AUDIO_EVENT_PRIOR
                    else "vlm/contact/static gates in per_frame_residuals.csv"
                ),
                residual_source=_source(per_frame, (term,), "non_invasive_loss_audit"),
            )
        )

    for summary_path in sorted(loss_dir.glob("*_residuals_summary.json")):
        summary = _read_json(summary_path)
        stage_label = str(summary.get("stage_label") or summary_path.stem)
        terms = summary.get("terms", {})
        rows = int(summary.get("rows") or 0)
        if not isinstance(terms, dict):
            continue
        for term, raw_total in sorted(terms.items()):
            if term in {"E_total", "E_visual"} or term not in ENERGY_TO_KIND:
                continue
            total = float(raw_total or 0.0)
            if total == 0.0:
                continue
            kind = ENERGY_TO_KIND[term]
            summaries.append(FactorEnergySummary(f"{stage_label}:{term}", kind, rows, total))
            factors.append(
                FactorSpec(
                    factor_id=_stage_factor_id(stage_label, term, kind),
                    kind=kind,
                    frame_count=rows,
                    input_refs=_input_refs(kind),
                    residual_unit="legacy_energy",
                    weight_source=f"legacy_optimizer_style_residual_report:{stage_label}:{term}",
                    gate_source=None,
                    residual_source=_source(summary_path, ("terms", term), "optimizer_style_residual_report"),
                )
            )

    generic_problem = profile.data.get("generic_object_problem", {})
    if observation_rows:
        typed_observations = list(adapt_legacy_observation_rows(
            profile.case_name,
            observation_rows,
            str(repo_relative_value(observation_csv)),
        ).measurements)
        supplemental = adapt_configured_supplemental_measurements(profile, result_dir)
        typed_observations.extend(supplemental.measurements)
        problem_config = profile.data.get("generic_object_problem", {})
        measurement_factors = set(problem_config.get("measurement_factors", ())) if isinstance(problem_config, dict) else set()
        measurement_roles = problem_config.get("measurement_roles", {}) if isinstance(problem_config, dict) else {}
        typed_factor_specs = (
            ("point_reprojection", FactorKind.POINT_REPROJECTION, Point2DMeasurement, "pixel_point_delta"),
            ("metric_depth", FactorKind.METRIC_DEPTH, MetricDepthMeasurement, "metric_depth_delta"),
        )
        for factor_name, factor_kind, measurement_type, residual_unit in typed_factor_specs:
            if factor_name not in measurement_factors:
                continue
            configured_roles = set(measurement_roles.get(factor_name, ())) if isinstance(measurement_roles, dict) else set()
            selected = tuple(
                measurement
                for measurement in typed_observations
                if isinstance(measurement, measurement_type)
                and (not configured_roles or measurement.meta.feature.semantic_role in configured_roles)
            )
            if not selected:
                continue
            factors = [factor for factor in factors if factor.kind != factor_kind]
            summaries = [summary for summary in summaries if summary.kind != factor_kind]
            source_fields = tuple(sorted({field for measurement in selected for field in measurement.meta.source.fields}))
            factors.append(
                FactorSpec(
                    factor_id=f"{factor_name}:measurement_ir",
                    kind=factor_kind,
                    frame_count=len(selected),
                    input_refs=_input_refs(factor_kind),
                    residual_unit=residual_unit,
                    weight_source=f"factor_runtime_configuration:{factor_name}",
                    gate_source="VisibilityState activation",
                    residual_source=_source(Path(selected[0].meta.source.artifact), source_fields, f"measurement_ir_{factor_name}"),
                )
            )
            summaries.append(FactorEnergySummary(f"measurement_ir:{factor_name}", factor_kind, len(selected), 0.0))
        line_measurements = tuple(
            measurement
            for measurement in typed_observations
            if isinstance(measurement, Line2DMeasurement)
        )
        if line_measurements:
            factors = [
                factor
                for factor in factors
                if not (
                    factor.kind == FactorKind.POINT_REPROJECTION
                    and factor.residual_source.producer == "non_invasive_loss_audit"
                )
            ]
            summaries = [
                summary
                for summary in summaries
                if not (
                    summary.kind == FactorKind.POINT_REPROJECTION
                    and summary.term in {"E_2d", "E_visual"}
                )
            ]
            source_fields = tuple(
                sorted(
                    {
                        field
                        for measurement in line_measurements
                        for field in measurement.meta.source.fields
                    }
                )
            )
            line_source_path = Path(line_measurements[0].meta.source.artifact)
            factors.append(
                FactorSpec(
                    factor_id="line_reprojection:measurement_ir",
                    kind=FactorKind.LINE_REPROJECTION,
                    frame_count=len(line_measurements),
                    input_refs=_input_refs(FactorKind.LINE_REPROJECTION),
                    residual_unit="pixel_line_delta",
                    weight_source="factor_runtime_configuration:line_reprojection",
                    gate_source="VisibilityState activation",
                    residual_source=_source(
                        line_source_path,
                        source_fields,
                        "measurement_ir_line_observation",
                    ),
                )
            )
            summaries.append(
                FactorEnergySummary(
                    "measurement_ir:line_reprojection",
                    FactorKind.LINE_REPROJECTION,
                    len(line_measurements),
                    0.0,
                )
            )

    sequence_factors = set(generic_problem.get("sequence_factors", ())) if isinstance(generic_problem, dict) else set()
    for factor_name, factor_kind, order in (
        ("temporal_velocity", FactorKind.TEMPORAL_VELOCITY, 1),
        ("temporal_acceleration", FactorKind.TEMPORAL_ACCELERATION, 2),
    ):
        if factor_name not in sequence_factors:
            continue
        factors = [factor for factor in factors if factor.kind != factor_kind]
        summaries = [summary for summary in summaries if summary.kind != factor_kind]
        temporal_rows = max(0, len(observation_rows) - order)
        factors.append(
            FactorSpec(
                factor_id=f"{factor_name}:state_sequence",
                kind=factor_kind,
                frame_count=temporal_rows,
                input_refs=_input_refs(factor_kind),
                residual_unit=f"state_{'first' if order == 1 else 'second'}_difference",
                weight_source=f"factor_runtime_configuration:{factor_name}",
                gate_source="InteractionState ContactMode/MotionMode activation",
                residual_source=_source(
                    observation_csv,
                    ("frame", "time"),
                    f"state_spec_sequence_{factor_name}_prior",
                ),
            )
        )
        summaries.append(
            FactorEnergySummary(
                f"state_spec:sequence_{factor_name}_prior",
                factor_kind,
                temporal_rows,
                0.0,
            )
        )

    if isinstance(generic_problem, dict) and generic_problem.get("contact_source") in {"interaction_state", "event_directional"}:
        contact_source = str(generic_problem["contact_source"])
        contact_state_csv = result_dir / str(generic_problem.get("contact_state_artifact", "contact_state_frames.csv"))
        contact_state_rows = _read_csv(contact_state_csv)
        contact_source_csv = (
            result_dir / str(generic_problem["contact_event_artifact"])
            if contact_source == "event_directional"
            else contact_state_csv
        )
        contact_source_rows = _read_csv(contact_source_csv)
        active_contact_rows = sum(
            1
            for row in contact_source_rows
            if (
                row.get("contact_type") not in {"floor_contact_event", "plane_support_contact_event"}
                if contact_source == "event_directional"
                else int(float(row.get("human_contact_state", row.get("anchor_contact_state", "0")) or 0)) == 1
            )
        )
        if active_contact_rows:
            factors = [factor for factor in factors if factor.kind != FactorKind.CONTACT_DISTANCE]
            summaries = [summary for summary in summaries if summary.kind != FactorKind.CONTACT_DISTANCE]
            fields = tuple(
                field
                for field in (
                    "frame",
                    "time",
                    "human_contact_state",
                    "anchor_contact_state",
                    "contact_label",
                    "anchor_score",
                )
                if field in contact_source_rows[0]
            )
            factors.append(
                FactorSpec(
                    factor_id="contact_distance:interaction_state",
                    kind=FactorKind.CONTACT_DISTANCE,
                    frame_count=active_contact_rows,
                    input_refs=_input_refs(FactorKind.CONTACT_DISTANCE),
                    residual_unit="metric_surface_distance",
                    weight_source="factor_runtime_configuration:contact_distance",
                    gate_source="InteractionState human_active",
                    residual_source=_source(contact_source_csv, fields, f"contact_{contact_source}_adapter"),
                )
            )
            summaries.append(
                FactorEnergySummary(
                    "interaction_state:contact_distance",
                    FactorKind.CONTACT_DISTANCE,
                    active_contact_rows,
                    0.0,
                )
            )

    if not any(factor.kind == FactorKind.POSE_PRIOR for factor in factors):
        factors.append(
            FactorSpec(
                factor_id="pose_prior:state_initializer",
                kind=FactorKind.POSE_PRIOR,
                frame_count=len(per_frame_rows),
                input_refs=_input_refs(FactorKind.POSE_PRIOR),
                residual_unit="state_spec_initializer_delta",
                weight_source="state_spec_initializer_prior",
                gate_source=None,
                residual_source=_source(
                    result_dir / "stage3_metrics.json",
                    ("adapter", "state_model", "initializer"),
                    "state_spec_initializer_prior",
                ),
            )
        )
        summaries.append(FactorEnergySummary("state_spec:initializer_prior", FactorKind.POSE_PRIOR, len(per_frame_rows), 0.0))

    if not any(factor.kind == FactorKind.TEMPORAL_ACCELERATION for factor in factors):
        acceleration_rows = max(0, len(per_frame_rows) - 2)
        factors.append(
            FactorSpec(
                factor_id="temporal_acceleration:state_sequence",
                kind=FactorKind.TEMPORAL_ACCELERATION,
                frame_count=acceleration_rows,
                input_refs=_input_refs(FactorKind.TEMPORAL_ACCELERATION),
                residual_unit="state_second_difference",
                weight_source="state_spec_sequence_acceleration_prior",
                gate_source="InteractionState ContactMode/MotionMode activation",
                residual_source=_source(
                    result_dir / "stage3_metrics.json",
                    ("adapter", "state_model", "temporal_acceleration"),
                    "state_spec_sequence_acceleration_prior",
                ),
            )
        )
        summaries.append(
            FactorEnergySummary(
                "state_spec:sequence_acceleration_prior",
                FactorKind.TEMPORAL_ACCELERATION,
                acceleration_rows,
                0.0,
            )
        )

    has_contact_factor = any(factor.kind == FactorKind.CONTACT_DISTANCE for factor in factors)
    active_contact_rows = sum(1 for row in contact_rows if row.get("contact_active") == "1")
    if active_contact_rows and not has_contact_factor:
        fields = tuple(field for field in ("contact_active", "contact_conf", "anchor_score", "palm_to_line_px", "local_s_drift") if contact_rows and field in contact_rows[0])
        factors.append(
            FactorSpec(
                factor_id="contact_distance:contact_constraint_shadow",
                kind=FactorKind.CONTACT_DISTANCE,
                frame_count=active_contact_rows,
                input_refs=_input_refs(FactorKind.CONTACT_DISTANCE),
                residual_unit="contact_constraint",
                weight_source="legacy_contact_confidence_or_anchor_score",
                gate_source="contact_active in object_contact_points.csv",
                residual_source=_source(contact_csv, fields or ("contact_active",), "contact_constraint_shadow"),
            )
        )
        summaries.append(FactorEnergySummary("object_contact_points:contact_active", FactorKind.CONTACT_DISTANCE, active_contact_rows, 0.0))

    adapter = stage3_metrics.get("adapter", {}) if isinstance(stage3_metrics.get("adapter"), dict) else {}
    if profile.case_name == "chair" or adapter.get("component") == "semantic_graph_6d" or adapter.get("solver"):
        for joint_id, field in (
            ("joint.front_to_rear", "rear_joint_angle"),
            ("joint.front_to_seat", "seat_joint_angle"),
        ):
            factors.append(
                FactorSpec(
                    factor_id=f"joint_limit:{joint_id}",
                    kind=FactorKind.JOINT_LIMIT,
                    frame_count=len(per_frame_rows),
                    input_refs=(
                        FactorInputRef("state", "StateSpec", joint_id),
                        FactorInputRef("geometry", "GeometryDescriptor", "articulated_urdf"),
                    ),
                    residual_unit="radian_bound_violation",
                    weight_source=f"state_spec_bound:{joint_id}",
                    gate_source=None,
                    residual_source=_source(
                        result_dir / "stage3_metrics.json",
                        ("adapter", "semantic_graph_6d", field),
                        "state_spec_joint_limit_shadow",
                    ),
                )
            )
            summaries.append(FactorEnergySummary(f"state_spec_bound:{joint_id}", FactorKind.JOINT_LIMIT, len(per_frame_rows), 0.0))
    mug_phase_prior_resolved = False
    if profile.case_name == "mug":
        mug_phase_prior_resolved, mug_phase_rows, _mug_phase_report = _mug_periodic_phase_prior_is_resolved(result_dir)
        if mug_phase_prior_resolved:
            factors.append(
                FactorSpec(
                    factor_id="periodic_phase_prior:observation_seed_axial_phase",
                    kind=FactorKind.PERIODIC_PHASE_PRIOR,
                    frame_count=mug_phase_rows,
                    input_refs=_input_refs(FactorKind.PERIODIC_PHASE_PRIOR),
                    residual_unit="radian_phase_prior",
                    weight_source="observation_derived_body_pose_and_axial_phase",
                    gate_source="visible/interpolated handle phase in observation_seed/axial_phase.csv",
                    residual_source=_source(
                        result_dir / "observation_seed/axial_phase.csv",
                        ("m17_phase_rad", "m43_phase_rad", "vlm_visibility", "source"),
                        "projected_periodic_phase_prior_shadow",
                    ),
                )
            )
            summaries.append(
                FactorEnergySummary(
                    "observation_seed:periodic_phase_prior",
                    FactorKind.PERIODIC_PHASE_PRIOR,
                    mug_phase_rows,
                    0.0,
                )
            )

    if per_frame_rows and any(row.get("vlm_contact_gate", "") for row in per_frame_rows):
        mapped.add("vlm_contact_gate")
    if per_frame_rows and any(row.get("vlm_anchor_gate", "") for row in per_frame_rows):
        mapped.add("vlm_anchor_gate")
    if per_frame_rows and any(row.get("contact_active", "") for row in per_frame_rows):
        mapped.add("contact_active")
    if per_frame_rows and any(row.get("static_active", "") for row in per_frame_rows):
        mapped.add("static_active")
    for field in ("failure_label", "contact_state", "visibility_state"):
        if per_frame_rows and any(row.get(field, "") for row in per_frame_rows):
            mapped.add(field)

    gaps: list[FactorGap] = []
    phase = adapter.get("phase_reconstruction", {}) if isinstance(adapter.get("phase_reconstruction"), dict) else {}
    if phase.get("snapshot_fallback_used") and not (profile.case_name == "mug" and mug_phase_prior_resolved):
        gaps.append(
            FactorGap(
                "phase_snapshot_fallback",
                "known_gap",
                "legacy phase reconstruction uses fallback/snapshot evidence; not promoted to PosePriorFactor",
                str(repo_relative_value(result_dir / "stage3_metrics.json")),
            )
        )
    if (adapter.get("component") == "semantic_graph_6d" or adapter.get("solver")) and not _chair_private_solver_gap_is_resolved(profile, result_dir):
        gaps.append(
            FactorGap(
                "semantic_graph_solver_private",
                "known_gap",
                "semantic graph 6D seed remains solver-private and is represented only as StateSpec shadow in this branch",
                str(repo_relative_value(result_dir / "stage3_metrics.json")),
            )
        )
    generic_problem = profile.data.get("generic_object_problem", {})
    line_capability_promoted = (
        isinstance(generic_problem, dict)
        and generic_problem.get("initializer") == "line_s_two_site"
    )
    if stage4_metrics.get("line_object_special_refinement") and not line_capability_promoted:
        gaps.append(
            FactorGap(
                "line_contact_lock_special_refinement",
                "known_gap",
                "line_contact_lock is a compatibility seed/refinement path; factor registry records equivalent contact/temporal terms but does not consume it",
                str(repo_relative_value(result_dir / "stage4_metrics.json")),
            )
        )

    if trace_rows:
        mapped.update(field for field in ("iteration", "active_frames", "active_contact_frames", "active_audio_frames") if field in trace_rows[0])
    active_terms = loss_summary.get("active_terms", [])
    if isinstance(active_terms, list):
        unsupported = sorted(str(term) for term in active_terms if str(term) not in {"E_total", *SUPPORTED_ENERGY_TERMS})
        for term in unsupported:
            gaps.append(
                FactorGap(
                    f"unsupported_loss_term:{term}",
                    "deferred",
                    "loss_summary active term is not mapped to first-pass FactorKind",
                    str(repo_relative_value(loss_dir / "loss_summary.json")),
                )
            )

    nonempty = _nonempty_fields(per_frame_rows)
    return FactorAdaptationResult(tuple(factors), tuple(summaries), tuple(gaps), tuple(sorted(mapped)), tuple(sorted(nonempty - mapped)))


def artifact_sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
