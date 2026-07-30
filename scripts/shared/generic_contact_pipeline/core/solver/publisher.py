"""Single atomic publisher for generic object sequence results."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ..state import StateSpec
from .optimization import GenericSequenceSolveResult
from .optimization import SequenceOptimizationProblem, build_runtime_residual_blocks


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv_atomic(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


@dataclass(frozen=True)
class ObjectPublicationGate:
    passed: bool
    gate_ids: tuple[str, ...]
    blocking_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.gate_ids:
            raise ValueError("object publication requires explicit hard gate ids")
        if self.passed == bool(self.blocking_reasons):
            raise ValueError("object publication pass state and blocking reasons are inconsistent")


@dataclass(frozen=True)
class ObjectPublicationResult:
    status: str
    candidate_path: str
    candidate_sha256: str
    accepted_path: str | None
    accepted_sha256: str | None
    solve_attempt_id: str
    case_dispatch_used: bool
    human_state_optimized: bool


class AcceptedObjectOutputPublisher:
    """Write candidate state always and canonical object_pose.csv only after hard gates."""

    candidate_name = "generic_object_pose_candidate.csv"
    accepted_name = "object_pose.csv"

    def publish(
        self,
        *,
        result: GenericSequenceSolveResult,
        state_spec: StateSpec,
        template_rows: Sequence[Mapping[str, object]],
        candidate_dir: Path,
        accepted_result_dir: Path,
        gate: ObjectPublicationGate,
    ) -> ObjectPublicationResult:
        if candidate_dir.resolve() == accepted_result_dir.resolve():
            raise ValueError("generic object candidate and accepted directories must differ")
        by_frame = {int(row["frame"]): dict(row) for row in template_rows}
        if tuple(sorted(by_frame)) != result.frames:
            raise ValueError("object publication template rows must align with solved frames")
        state_fields = tuple(field for dof in state_spec.dofs for field in dof.source_fields)
        if len(state_fields) != len(result.states[0]):
            raise ValueError("object publication StateSpec fields must match solved state width")
        rows: list[dict[str, object]] = []
        for frame, state in zip(result.frames, result.states):
            row = by_frame[frame]
            row.update({field: f"{float(value):.9f}" for field, value in zip(state_fields, state)})
            row["source"] = "generic_sequence_executor"
            row["generic_solve_attempt_id"] = result.solve_attempt_id
            rows.append(row)
        fields = list(template_rows[0])
        for field in (*state_fields, "source", "generic_solve_attempt_id"):
            if field not in fields:
                fields.append(field)
        candidate_path = candidate_dir / self.candidate_name
        _write_csv_atomic(candidate_path, rows, fields)
        accepted_path: Path | None = None
        if gate.passed:
            accepted_path = accepted_result_dir / self.accepted_name
            _write_csv_atomic(accepted_path, rows, fields)
        return ObjectPublicationResult(
            status="accepted" if accepted_path is not None else "candidate_blocked",
            candidate_path=str(candidate_path),
            candidate_sha256=_sha256(candidate_path),
            accepted_path=str(accepted_path) if accepted_path is not None else None,
            accepted_sha256=_sha256(accepted_path) if accepted_path is not None else None,
            solve_attempt_id=result.solve_attempt_id,
            case_dispatch_used=False,
            human_state_optimized=False,
        )


def evaluate_object_publication_gate(
    problem: SequenceOptimizationProblem,
    result: GenericSequenceSolveResult,
    *,
    normalized_contact_rmse_limit: float = 2.0,
    normalized_projection_rmse_limit: float = 2.0,
    contact_gap_p95_m_limit: float = 0.08,
    contact_gap_max_m_limit: float = 0.20,
    trusted_contact_confidence_min: float = 0.50,
    projection_p95_px_limit: float = 24.0,
    stationarity_limit: float = 0.10,
) -> tuple[ObjectPublicationGate, dict[str, object]]:
    """Evaluate case-independent solver and geometry-factor publication gates."""

    states = {frame: state for frame, state in zip(result.frames, result.states)}
    blocks = build_runtime_residual_blocks(
        problem.residual_execution_plan,
        problem.residual_input_builder(states),
        problem.factor_ids,
    )
    solver_stationary = (
        result.termination_status == 0
        and math.isfinite(result.optimality)
        and result.optimality <= stationarity_limit
    )
    solver_termination_accepted = result.success or solver_stationary
    metrics: dict[str, object] = {
        "solver_success": result.success,
        "solver_stationary": solver_stationary,
        "solver_termination_accepted": solver_termination_accepted,
        "stationarity_limit": stationarity_limit,
        "termination_status": result.termination_status,
        "optimality": result.optimality,
        "active_bound_count": result.active_bound_count,
        "initial_squared_error": result.initial_squared_error,
        "final_squared_error": result.final_squared_error,
        "objective_nonincrease": result.final_squared_error <= result.initial_squared_error + 1e-9,
    }
    gate_ids = ["solver_converged_or_stationary", "objective_nonincrease"]
    blocking: list[str] = []
    if not solver_termination_accepted:
        blocking.append("solver_not_converged_or_stationary")
    if not metrics["objective_nonincrease"]:
        blocking.append("objective_increased")
    residual_inputs = problem.residual_input_builder(states)

    def active_rows(raw_weight: object, row_count: int) -> np.ndarray:
        weights = np.asarray(raw_weight, dtype=float)
        if weights.ndim == 0:
            return np.full(row_count, float(weights) > 0.0, dtype=bool)
        weights = weights.reshape(-1)
        if len(weights) != row_count:
            raise ValueError("factor metric weights must match raw row count")
        return weights > 0.0

    for factor_id, values in blocks:
        if factor_id.startswith("contact_distance:"):
            gate_id, limit = "contact_distance_normalized_rmse", normalized_contact_rmse_limit
        elif factor_id.startswith(("line_reprojection:", "point_reprojection:")):
            gate_id, limit = "projection_normalized_rmse", normalized_projection_rmse_limit
        else:
            continue
        rmse = float((float(values @ values) / int(values.size)) ** 0.5)
        metrics[gate_id] = rmse
        metrics[f"{gate_id}_limit"] = limit
        if gate_id not in gate_ids:
            gate_ids.append(gate_id)
        if not math.isfinite(rmse) or rmse > limit:
            blocking.append(f"{gate_id}_failed")
        raw_inputs = residual_inputs[factor_id]
        if factor_id.startswith("contact_distance:"):
            gaps_m = np.linalg.norm(
                np.asarray(raw_inputs["anchors"], dtype=float)
                - np.asarray(raw_inputs["targets"], dtype=float),
                axis=1,
            )
            active_gaps_m = gaps_m[active_rows(raw_inputs["weight"], len(gaps_m))]
            if not len(active_gaps_m):
                blocking.append("contact_distance_has_no_active_rows")
                continue
            p95_m = float(np.quantile(active_gaps_m, 0.95))
            confidence = np.asarray(
                raw_inputs.get("sample_confidence", np.ones(len(gaps_m))), dtype=float
            ).reshape(-1)
            trusted_mask = (
                active_rows(raw_inputs["weight"], len(gaps_m))
                & (confidence >= trusted_contact_confidence_min)
            )
            trusted_gaps_m = gaps_m[trusted_mask]
            if not len(trusted_gaps_m):
                blocking.append("contact_distance_has_no_trusted_rows")
                continue
            max_m = float(np.max(trusted_gaps_m))
            metrics.update(
                {
                    "contact_gap_all_rows_p95_m": float(np.quantile(gaps_m, 0.95)),
                    "contact_gap_all_rows_max_m": float(np.max(gaps_m)),
                    "contact_gap_p95_m": p95_m,
                    "contact_gap_p95_m_limit": contact_gap_p95_m_limit,
                    "contact_gap_max_m": max_m,
                    "contact_gap_max_m_limit": contact_gap_max_m_limit,
                    "trusted_contact_confidence_min": trusted_contact_confidence_min,
                }
            )
            gate_ids.extend(("contact_gap_p95_m", "contact_gap_max_m"))
            if p95_m > contact_gap_p95_m_limit:
                blocking.append("contact_gap_p95_m_failed")
            if max_m > contact_gap_max_m_limit:
                blocking.append("contact_gap_max_m_failed")
        elif factor_id.startswith("line_reprojection:"):
            predicted = np.asarray(raw_inputs["predicted"], dtype=float)
            target = np.asarray(raw_inputs["target"], dtype=float)
            if raw_inputs.get("constraint_mode", "endpoints") == "axis_line":
                direction = target[:, 1, :] - target[:, 0, :]
                offsets = predicted - target[:, :1, :]
                errors_by_line_px = np.abs(
                    direction[:, None, 0] * offsets[:, :, 1]
                    - direction[:, None, 1] * offsets[:, :, 0]
                ) / np.linalg.norm(direction, axis=1)[:, None]
            elif bool(raw_inputs.get("allow_endpoint_swap", False)):
                direct = np.sum((predicted - target) ** 2, axis=(1, 2))
                swapped = predicted[:, ::-1, :]
                predicted = np.where((np.sum((swapped - target) ** 2, axis=(1, 2)) < direct)[:, None, None], swapped, predicted)
                errors_by_line_px = np.linalg.norm(predicted - target, axis=2)
            else:
                errors_by_line_px = np.linalg.norm(predicted - target, axis=2)
            errors_px = errors_by_line_px.reshape(-1)
            active_line_rows = active_rows(raw_inputs["weight"], len(predicted))
            active_errors_px = errors_by_line_px[active_line_rows].reshape(-1)
            if not len(active_errors_px):
                blocking.append("line_reprojection_has_no_active_rows")
                continue
            p95_px = float(np.quantile(active_errors_px, 0.95))
            metrics.update(
                {
                    "projection_all_rows_p95_px": float(np.quantile(errors_px, 0.95)),
                    "projection_p95_px": p95_px,
                    "projection_p95_px_limit": projection_p95_px_limit,
                    "projection_max_px": float(np.max(active_errors_px)),
                }
            )
            gate_ids.append("projection_p95_px")
            if p95_px > projection_p95_px_limit:
                blocking.append("projection_p95_px_failed")
        elif factor_id.startswith("point_reprojection:"):
            predicted = np.asarray(raw_inputs["predicted"], dtype=float)
            target = np.asarray(raw_inputs["target"], dtype=float)
            errors_px = np.linalg.norm(predicted - target, axis=1)
            active_errors_px = errors_px[active_rows(raw_inputs["weight"], len(errors_px))]
            if not len(active_errors_px):
                blocking.append("point_reprojection_has_no_active_rows")
                continue
            p95_px = float(np.quantile(active_errors_px, 0.95))
            metrics.update(
                {
                    "point_projection_all_rows_p95_px": float(np.quantile(errors_px, 0.95)),
                    "point_projection_p95_px": p95_px,
                    "point_projection_p95_px_limit": projection_p95_px_limit,
                    "point_projection_max_px": float(np.max(active_errors_px)),
                }
            )
            gate_ids.append("point_projection_p95_px")
            if p95_px > projection_p95_px_limit:
                blocking.append("point_projection_p95_px_failed")
    gate = ObjectPublicationGate(
        passed=not blocking,
        gate_ids=tuple(gate_ids),
        blocking_reasons=tuple(blocking),
    )
    return gate, metrics


def object_publication_record(
    result: ObjectPublicationResult,
    gate: ObjectPublicationGate,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": result.status,
        "candidate_path": result.candidate_path,
        "candidate_sha256": result.candidate_sha256,
        "accepted_path": result.accepted_path,
        "accepted_sha256": result.accepted_sha256,
        "solve_attempt_id": result.solve_attempt_id,
        "hard_gate": {
            "passed": gate.passed,
            "gate_ids": list(gate.gate_ids),
            "blocking_reasons": list(gate.blocking_reasons),
        },
        "case_dispatch_used": result.case_dispatch_used,
        "human_state_optimized": result.human_state_optimized,
    }
