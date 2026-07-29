from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .factor_residuals import FactorResidualEvaluator
from .runtime import GenericExecutorAttemptLedger


SUPPORTED_RESIDUAL_REFS = {
    "shadow_residual::point_reprojection": "FactorResidualEvaluator.point_reprojection",
    "shadow_residual::contact_distance": "FactorResidualEvaluator.contact_distance",
    "shadow_residual::metric_depth": "FactorResidualEvaluator.metric_depth",
    "shadow_residual::support_and_penetration": "FactorResidualEvaluator.support_penetration",
    "shadow_residual::pose_prior": "FactorResidualEvaluator.pose_prior",
    "shadow_residual::temporal_velocity": "FactorResidualEvaluator.temporal_delta",
    "shadow_residual::temporal_acceleration": "FactorResidualEvaluator.temporal_delta",
    "shadow_residual::joint_limit": "FactorResidualEvaluator.joint_limit",
    "shadow_residual::gauge_constraint": "FactorResidualEvaluator.gauge_constraint",
    "shadow_residual::regularization": "FactorResidualEvaluator.regularization",
    "shadow_residual::periodic_phase_prior": "FactorResidualEvaluator.periodic_phase_prior",
    "shadow_residual::audio_event_prior": "FactorResidualEvaluator.audio_event_prior",
}


@dataclass(frozen=True)
class ResidualBoundaryRecord:
    factor_id: str
    residual_fn_ref: str
    evaluator_ref: str
    status: str

    def __post_init__(self) -> None:
        if not self.factor_id or not self.residual_fn_ref or not self.evaluator_ref:
            raise ValueError("ResidualBoundaryRecord requires factor id, residual fn ref, and evaluator ref")
        if self.status not in {"supported_not_executed", "pending_generic_residual"}:
            raise ValueError("invalid residual boundary status")


@dataclass(frozen=True)
class ResidualGapRecord:
    gap_id: str
    residual_fn_ref: str
    factor_ids: tuple[str, ...]
    reason: str
    status: str = "pending_generic_residual"

    def __post_init__(self) -> None:
        if not self.gap_id or not self.residual_fn_ref or not self.factor_ids or not self.reason:
            raise ValueError("ResidualGapRecord requires gap id, residual ref, factor ids, and reason")
        if self.status != "pending_generic_residual":
            raise ValueError("residual gap status must be pending_generic_residual")


@dataclass(frozen=True)
class GenericResidualBoundary:
    schema_version: int
    attempt_id: str
    status: str
    supported_count: int
    pending_count: int
    records: tuple[ResidualBoundaryRecord, ...]
    pending_gap_records: tuple[ResidualGapRecord, ...]
    case_dispatch_used: bool
    residuals_executed: bool
    solver_executed: bool
    accepted_outputs_written: bool
    canonical_sha256: str

    def __post_init__(self) -> None:
        if not self.attempt_id.startswith("generic-attempt-"):
            raise ValueError("residual boundary must attach to a generic attempt")
        if self.status != "planned_not_executed":
            raise ValueError("residual boundary must remain planned_not_executed")
        if self.supported_count + self.pending_count != len(self.records):
            raise ValueError("residual boundary counts must match records")
        if self.pending_count and not self.pending_gap_records:
            raise ValueError("pending residuals require gap records")
        if self.case_dispatch_used or self.residuals_executed or self.solver_executed or self.accepted_outputs_written:
            raise ValueError("residual boundary must not dispatch, execute residuals, solve, or write accepted outputs")


@dataclass(frozen=True)
class ResidualExecutionPlanRecord:
    factor_id: str
    residual_fn_ref: str
    evaluator_ref: str
    input_ids: tuple[str, ...]
    gate_provenance: tuple[str, ...]
    status: str

    def __post_init__(self) -> None:
        if not self.factor_id or not self.residual_fn_ref or not self.evaluator_ref:
            raise ValueError("ResidualExecutionPlanRecord requires factor id, residual ref, and evaluator ref")
        if not self.input_ids:
            raise ValueError("residual execution plan records require input ids")
        if self.status not in {"ready_not_executed", "blocked_pending_residual"}:
            raise ValueError("invalid residual execution plan status")


@dataclass(frozen=True)
class GenericResidualExecutionPlan:
    schema_version: int
    attempt_id: str
    status: str
    boundary_sha256: str
    record_count: int
    ready_count: int
    blocked_count: int
    records: tuple[ResidualExecutionPlanRecord, ...]
    case_dispatch_used: bool
    residuals_executed: bool
    solver_executed: bool
    accepted_outputs_written: bool
    canonical_sha256: str

    def __post_init__(self) -> None:
        if not self.attempt_id.startswith("generic-attempt-"):
            raise ValueError("residual execution plan must attach to a generic attempt")
        if self.status != "planned_not_executed":
            raise ValueError("residual execution plan must remain planned_not_executed")
        if self.record_count != len(self.records):
            raise ValueError("residual execution plan record count mismatch")
        if self.ready_count + self.blocked_count != self.record_count:
            raise ValueError("residual execution plan ready/blocked count mismatch")
        if self.case_dispatch_used or self.residuals_executed or self.solver_executed or self.accepted_outputs_written:
            raise ValueError("residual execution plan must not dispatch, execute residuals, solve, or write accepted outputs")


@dataclass(frozen=True)
class ResidualDryRunRecord:
    factor_id: str
    evaluator_ref: str
    status: str
    residual_count: int
    rms: float
    residual_sha256: str

    def __post_init__(self) -> None:
        if not self.factor_id or not self.evaluator_ref or not self.status:
            raise ValueError("ResidualDryRunRecord requires factor id, evaluator ref, and status")
        if self.status not in {"executed", "skipped_missing_inputs", "blocked_pending_residual"}:
            raise ValueError("invalid residual dry-run status")
        if self.residual_count < 0 or self.rms < 0.0:
            raise ValueError("invalid residual dry-run metrics")


@dataclass(frozen=True)
class GenericResidualDryRunLedger:
    schema_version: int
    execution_plan_sha256: str
    status: str
    record_count: int
    executed_count: int
    skipped_count: int
    records: tuple[ResidualDryRunRecord, ...]
    case_dispatch_used: bool
    residuals_executed: bool
    solver_executed: bool
    accepted_outputs_written: bool
    canonical_sha256: str

    def __post_init__(self) -> None:
        if self.status != "residuals_executed_dry_run":
            raise ValueError("residual dry-run ledger must use residuals_executed_dry_run status")
        if self.record_count != len(self.records):
            raise ValueError("residual dry-run record count mismatch")
        if self.executed_count + self.skipped_count != self.record_count:
            raise ValueError("residual dry-run executed/skipped count mismatch")
        if self.case_dispatch_used or self.solver_executed or self.accepted_outputs_written:
            raise ValueError("residual dry-run must not dispatch, solve, or write accepted outputs")
        if self.residuals_executed != (self.executed_count > 0):
            raise ValueError("residuals_executed must reflect whether any residual block executed")


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def residual_boundary_record(record: ResidualBoundaryRecord) -> dict[str, object]:
    return asdict(record)


def residual_gap_record(record: ResidualGapRecord) -> dict[str, object]:
    return asdict(record)


def residual_boundary_ledger_record(boundary: GenericResidualBoundary) -> dict[str, object]:
    payload = asdict(boundary)
    payload["records"] = [residual_boundary_record(record) for record in boundary.records]
    payload["pending_gap_records"] = [residual_gap_record(record) for record in boundary.pending_gap_records]
    return payload


def residual_execution_plan_record(record: ResidualExecutionPlanRecord) -> dict[str, object]:
    return asdict(record)


def residual_execution_plan_ledger_record(plan: GenericResidualExecutionPlan) -> dict[str, object]:
    payload = asdict(plan)
    payload["records"] = [residual_execution_plan_record(record) for record in plan.records]
    return payload


def residual_dry_run_record(record: ResidualDryRunRecord) -> dict[str, object]:
    return asdict(record)


def residual_dry_run_ledger_record(ledger: GenericResidualDryRunLedger) -> dict[str, object]:
    payload = asdict(ledger)
    payload["records"] = [residual_dry_run_record(record) for record in ledger.records]
    return payload


def build_generic_residual_boundary(
    attempt: GenericExecutorAttemptLedger,
    compiled_factor_shadow: dict[str, object],
) -> GenericResidualBoundary:
    records = compiled_factor_shadow.get("records", [])
    if not isinstance(records, list):
        records = []
    boundary_records: list[ResidualBoundaryRecord] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        factor_id = str(record.get("factor_id", ""))
        residual_fn_ref = str(record.get("residual_fn_ref", ""))
        evaluator_ref = SUPPORTED_RESIDUAL_REFS.get(residual_fn_ref, "pending_generic_residual")
        status = "supported_not_executed" if residual_fn_ref in SUPPORTED_RESIDUAL_REFS else "pending_generic_residual"
        boundary_records.append(
            ResidualBoundaryRecord(
                factor_id=factor_id,
                residual_fn_ref=residual_fn_ref,
                evaluator_ref=evaluator_ref,
                status=status,
            )
        )
    supported = sum(1 for record in boundary_records if record.status == "supported_not_executed")
    pending = sum(1 for record in boundary_records if record.status == "pending_generic_residual")
    pending_by_ref: dict[str, list[str]] = {}
    for record in boundary_records:
        if record.status == "pending_generic_residual":
            pending_by_ref.setdefault(record.residual_fn_ref, []).append(record.factor_id)
    pending_gap_records = tuple(
        ResidualGapRecord(
            gap_id=f"missing_generic_residual:{residual_ref.removeprefix('shadow_residual::')}",
            residual_fn_ref=residual_ref,
            factor_ids=tuple(sorted(factor_ids)),
            reason="no FactorResidualEvaluator capability is registered for this compiled residual_fn_ref",
        )
        for residual_ref, factor_ids in sorted(pending_by_ref.items())
    )
    payload = {
        "schema_version": 1,
        "attempt_id": attempt.attempt_id,
        "status": "planned_not_executed",
        "supported_count": supported,
        "pending_count": pending,
        "records": [residual_boundary_record(record) for record in boundary_records],
        "pending_gap_records": [residual_gap_record(record) for record in pending_gap_records],
        "case_dispatch_used": False,
        "residuals_executed": False,
        "solver_executed": False,
        "accepted_outputs_written": False,
    }
    return GenericResidualBoundary(
        schema_version=1,
        attempt_id=attempt.attempt_id,
        status="planned_not_executed",
        supported_count=supported,
        pending_count=pending,
        records=tuple(boundary_records),
        pending_gap_records=pending_gap_records,
        case_dispatch_used=False,
        residuals_executed=False,
        solver_executed=False,
        accepted_outputs_written=False,
        canonical_sha256=_canonical_hash(payload),
    )


def build_generic_residual_execution_plan(
    attempt: GenericExecutorAttemptLedger,
    compiled_factor_shadow: dict[str, object],
    boundary: GenericResidualBoundary,
) -> GenericResidualExecutionPlan:
    if boundary.attempt_id != attempt.attempt_id:
        raise ValueError("residual execution plan must match attempt ledger")
    records = compiled_factor_shadow.get("records", [])
    if not isinstance(records, list):
        records = []
    boundary_by_factor = {record.factor_id: record for record in boundary.records}
    plan_records: list[ResidualExecutionPlanRecord] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        factor_id = str(record.get("factor_id", ""))
        boundary_record = boundary_by_factor.get(factor_id)
        if boundary_record is None:
            raise ValueError(f"missing residual boundary record for factor {factor_id}")
        input_ids = tuple(str(item) for item in record.get("input_ids", []) if item)
        gate_provenance = tuple(str(item) for item in record.get("gate_provenance", []) if item)
        plan_records.append(
            ResidualExecutionPlanRecord(
                factor_id=factor_id,
                residual_fn_ref=boundary_record.residual_fn_ref,
                evaluator_ref=boundary_record.evaluator_ref,
                input_ids=input_ids,
                gate_provenance=gate_provenance,
                status="ready_not_executed"
                if boundary_record.status == "supported_not_executed"
                else "blocked_pending_residual",
            )
        )
    ready = sum(1 for record in plan_records if record.status == "ready_not_executed")
    blocked = sum(1 for record in plan_records if record.status == "blocked_pending_residual")
    payload = {
        "schema_version": 1,
        "attempt_id": attempt.attempt_id,
        "status": "planned_not_executed",
        "boundary_sha256": boundary.canonical_sha256,
        "record_count": len(plan_records),
        "ready_count": ready,
        "blocked_count": blocked,
        "records": [residual_execution_plan_record(record) for record in plan_records],
        "case_dispatch_used": False,
        "residuals_executed": False,
        "solver_executed": False,
        "accepted_outputs_written": False,
    }
    return GenericResidualExecutionPlan(
        schema_version=1,
        attempt_id=attempt.attempt_id,
        status="planned_not_executed",
        boundary_sha256=boundary.canonical_sha256,
        record_count=len(plan_records),
        ready_count=ready,
        blocked_count=blocked,
        records=tuple(plan_records),
        case_dispatch_used=False,
        residuals_executed=False,
        solver_executed=False,
        accepted_outputs_written=False,
        canonical_sha256=_canonical_hash(payload),
    )


def _array_kwargs(raw_kwargs: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    for key, value in raw_kwargs.items():
        if isinstance(value, list):
            kwargs[key] = np.asarray(value, dtype=float)
        else:
            kwargs[key] = value
    return kwargs


def _residual_hash(residual: np.ndarray) -> str:
    payload = residual.astype(float).reshape(-1).tolist()
    return _canonical_hash(payload)


def build_generic_residual_dry_run(
    execution_plan: GenericResidualExecutionPlan | dict[str, Any],
    residual_inputs: dict[str, dict[str, Any]],
) -> GenericResidualDryRunLedger:
    evaluator = FactorResidualEvaluator()
    dry_run_records: list[ResidualDryRunRecord] = []
    if isinstance(execution_plan, dict):
        plan_records = [
            ResidualExecutionPlanRecord(
                factor_id=str(record.get("factor_id", "")),
                residual_fn_ref=str(record.get("residual_fn_ref", "")),
                evaluator_ref=str(record.get("evaluator_ref", "")),
                input_ids=tuple(str(item) for item in record.get("input_ids", []) if item),
                gate_provenance=tuple(str(item) for item in record.get("gate_provenance", []) if item),
                status=str(record.get("status", "")),
            )
            for record in execution_plan.get("records", [])
            if isinstance(record, dict)
        ]
        execution_plan_sha256 = str(execution_plan.get("canonical_sha256", ""))
    else:
        plan_records = list(execution_plan.records)
        execution_plan_sha256 = execution_plan.canonical_sha256
    for plan_record in plan_records:
        if plan_record.status != "ready_not_executed":
            dry_run_records.append(
                ResidualDryRunRecord(
                    factor_id=plan_record.factor_id,
                    evaluator_ref=plan_record.evaluator_ref,
                    status="blocked_pending_residual",
                    residual_count=0,
                    rms=0.0,
                    residual_sha256="",
                )
            )
            continue
        raw_kwargs = residual_inputs.get(plan_record.factor_id)
        if raw_kwargs is None:
            dry_run_records.append(
                ResidualDryRunRecord(
                    factor_id=plan_record.factor_id,
                    evaluator_ref=plan_record.evaluator_ref,
                    status="skipped_missing_inputs",
                    residual_count=0,
                    rms=0.0,
                    residual_sha256="",
                )
            )
            continue
        method_name = plan_record.evaluator_ref.removeprefix("FactorResidualEvaluator.")
        method = getattr(evaluator, method_name)
        residual = np.asarray(method(**_array_kwargs(raw_kwargs)), dtype=float).reshape(-1)
        rms = float(np.sqrt(np.mean(residual * residual))) if residual.size else 0.0
        dry_run_records.append(
            ResidualDryRunRecord(
                factor_id=plan_record.factor_id,
                evaluator_ref=plan_record.evaluator_ref,
                status="executed",
                residual_count=int(residual.size),
                rms=rms,
                residual_sha256=_residual_hash(residual),
            )
        )
    executed = sum(1 for record in dry_run_records if record.status == "executed")
    skipped = len(dry_run_records) - executed
    payload = {
        "schema_version": 1,
        "execution_plan_sha256": execution_plan_sha256,
        "status": "residuals_executed_dry_run",
        "record_count": len(dry_run_records),
        "executed_count": executed,
        "skipped_count": skipped,
        "records": [residual_dry_run_record(record) for record in dry_run_records],
        "case_dispatch_used": False,
        "residuals_executed": executed > 0,
        "solver_executed": False,
        "accepted_outputs_written": False,
    }
    return GenericResidualDryRunLedger(
        schema_version=1,
        execution_plan_sha256=execution_plan_sha256,
        status="residuals_executed_dry_run",
        record_count=len(dry_run_records),
        executed_count=executed,
        skipped_count=skipped,
        records=tuple(dry_run_records),
        case_dispatch_used=False,
        residuals_executed=executed > 0,
        solver_executed=False,
        accepted_outputs_written=False,
        canonical_sha256=_canonical_hash(payload),
    )
