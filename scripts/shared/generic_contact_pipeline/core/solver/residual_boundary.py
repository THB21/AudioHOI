from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from .runtime import GenericExecutorAttemptLedger


SUPPORTED_RESIDUAL_REFS = {
    "shadow_residual::point_reprojection": "FactorResidualEvaluator.point_reprojection",
    "shadow_residual::contact_distance": "FactorResidualEvaluator.contact_distance",
    "shadow_residual::pose_prior": "FactorResidualEvaluator.pose_prior",
    "shadow_residual::temporal_velocity": "FactorResidualEvaluator.temporal_delta",
    "shadow_residual::temporal_acceleration": "FactorResidualEvaluator.temporal_delta",
    "shadow_residual::joint_limit": "FactorResidualEvaluator.joint_limit",
    "shadow_residual::gauge_constraint": "FactorResidualEvaluator.gauge_constraint",
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
class GenericResidualBoundary:
    schema_version: int
    attempt_id: str
    status: str
    supported_count: int
    pending_count: int
    records: tuple[ResidualBoundaryRecord, ...]
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
        if self.case_dispatch_used or self.residuals_executed or self.solver_executed or self.accepted_outputs_written:
            raise ValueError("residual boundary must not dispatch, execute residuals, solve, or write accepted outputs")


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def residual_boundary_record(record: ResidualBoundaryRecord) -> dict[str, object]:
    return asdict(record)


def residual_boundary_ledger_record(boundary: GenericResidualBoundary) -> dict[str, object]:
    payload = asdict(boundary)
    payload["records"] = [residual_boundary_record(record) for record in boundary.records]
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
    payload = {
        "schema_version": 1,
        "attempt_id": attempt.attempt_id,
        "status": "planned_not_executed",
        "supported_count": supported,
        "pending_count": pending,
        "records": [residual_boundary_record(record) for record in boundary_records],
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
        case_dispatch_used=False,
        residuals_executed=False,
        solver_executed=False,
        accepted_outputs_written=False,
        canonical_sha256=_canonical_hash(payload),
    )
