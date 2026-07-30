from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from .optimization import (
    GenericSequenceSolveResult,
    SequenceOptimizationParameters,
    SequenceOptimizationProblem,
    solve_sequence_optimization,
)
from .problem_contract import SequenceProblemContract


@dataclass(frozen=True)
class GenericExecutorRuntimePlan:
    schema_version: int
    executor_id: str
    status: str
    sequence_contract_sha256: str
    compiled_factor_count: int
    compiled_factor_ids: tuple[str, ...]
    case_dispatch_used: bool
    solver_executed: bool
    accepted_outputs_written: bool
    canonical_sha256: str

    def __post_init__(self) -> None:
        if self.executor_id != "generic_sequence_executor":
            raise ValueError("runtime plan must target generic_sequence_executor")
        if self.status != "not_executed":
            raise ValueError("runtime plan is a boundary contract and must not execute")
        if self.case_dispatch_used:
            raise ValueError("generic runtime boundary must not use case dispatch")
        if self.solver_executed or self.accepted_outputs_written:
            raise ValueError("runtime boundary plan must not execute or write accepted outputs")
        if self.compiled_factor_count <= 0 or len(self.compiled_factor_ids) != self.compiled_factor_count:
            raise ValueError("runtime plan compiled factor ids/count mismatch")


@dataclass(frozen=True)
class GenericExecutorPrepareResult:
    schema_version: int
    executor_id: str
    status: str
    sequence_contract_sha256: str
    runtime_plan_sha256: str
    compiled_factor_count: int
    case_dispatch_used: bool
    solver_executed: bool
    accepted_outputs_written: bool
    canonical_sha256: str

    def __post_init__(self) -> None:
        if self.executor_id != "generic_sequence_executor":
            raise ValueError("prepare result must target generic_sequence_executor")
        if self.status != "prepared_not_executed":
            raise ValueError("prepare result must not execute")
        if self.case_dispatch_used:
            raise ValueError("generic executor prepare must not use case dispatch")
        if self.solver_executed or self.accepted_outputs_written:
            raise ValueError("generic executor prepare must not solve or write accepted outputs")


@dataclass(frozen=True)
class GenericExecutorAttemptLedger:
    schema_version: int
    attempt_id: str
    executor_id: str
    status: str
    sequence_contract_sha256: str
    runtime_plan_sha256: str
    prepare_sha256: str
    compiled_factor_count: int
    case_dispatch_used: bool
    residual_evaluation_status: str
    solver_executed: bool
    accepted_outputs_written: bool
    canonical_sha256: str

    def __post_init__(self) -> None:
        if self.executor_id != "generic_sequence_executor":
            raise ValueError("attempt ledger must target generic_sequence_executor")
        if not self.attempt_id.startswith("generic-attempt-"):
            raise ValueError("attempt_id must be deterministic generic-attempt id")
        if self.status != "attempt_planned_not_executed":
            raise ValueError("attempt ledger must remain planned/not executed")
        if self.residual_evaluation_status != "not_executed":
            raise ValueError("residual evaluation must remain not_executed in this branch")
        if self.case_dispatch_used:
            raise ValueError("generic attempt ledger must not use case dispatch")
        if self.solver_executed or self.accepted_outputs_written:
            raise ValueError("generic attempt ledger must not solve or write accepted outputs")


class GenericSequenceExecutor:
    executor_id = "generic_sequence_executor"

    def prepare(
        self,
        contract: SequenceProblemContract,
        runtime_plan: GenericExecutorRuntimePlan,
        compiled_factor_shadow: dict[str, object],
    ) -> GenericExecutorPrepareResult:
        if runtime_plan.sequence_contract_sha256 != contract.canonical_sha256:
            raise ValueError("runtime plan must match SequenceProblemContract")
        records = compiled_factor_shadow.get("records", [])
        if not isinstance(records, list):
            records = []
        compiled_factor_ids = tuple(str(record.get("factor_id")) for record in records if isinstance(record, dict) and record.get("factor_id"))
        if compiled_factor_ids != contract.compiled_factor_ids:
            raise ValueError("compiled factors must match SequenceProblemContract")
        payload = {
            "schema_version": 1,
            "executor_id": self.executor_id,
            "status": "prepared_not_executed",
            "sequence_contract_sha256": contract.canonical_sha256,
            "runtime_plan_sha256": runtime_plan.canonical_sha256,
            "compiled_factor_count": len(compiled_factor_ids),
            "case_dispatch_used": False,
            "solver_executed": False,
            "accepted_outputs_written": False,
        }
        return GenericExecutorPrepareResult(
            **payload,
            canonical_sha256=_canonical_hash(payload),
        )

    def plan_attempt(
        self,
        contract: SequenceProblemContract,
        runtime_plan: GenericExecutorRuntimePlan,
        prepared: GenericExecutorPrepareResult,
    ) -> GenericExecutorAttemptLedger:
        if runtime_plan.sequence_contract_sha256 != contract.canonical_sha256:
            raise ValueError("runtime plan must match SequenceProblemContract")
        if prepared.sequence_contract_sha256 != contract.canonical_sha256:
            raise ValueError("prepare result must match SequenceProblemContract")
        if prepared.runtime_plan_sha256 != runtime_plan.canonical_sha256:
            raise ValueError("prepare result must match runtime plan")
        payload = {
            "schema_version": 1,
            "executor_id": self.executor_id,
            "status": "attempt_planned_not_executed",
            "sequence_contract_sha256": contract.canonical_sha256,
            "runtime_plan_sha256": runtime_plan.canonical_sha256,
            "prepare_sha256": prepared.canonical_sha256,
            "compiled_factor_count": contract.compiled_factor_count,
            "case_dispatch_used": False,
            "residual_evaluation_status": "not_executed",
            "solver_executed": False,
            "accepted_outputs_written": False,
        }
        attempt_hash = _canonical_hash(payload)
        return GenericExecutorAttemptLedger(
            **payload,
            attempt_id=f"generic-attempt-{attempt_hash[:12]}",
            canonical_sha256=_canonical_hash({"attempt_id": f"generic-attempt-{attempt_hash[:12]}", **payload}),
        )

    def solve(
        self,
        problem: SequenceOptimizationProblem,
        parameters: SequenceOptimizationParameters = SequenceOptimizationParameters(),
    ) -> GenericSequenceSolveResult:
        return solve_sequence_optimization(problem, parameters)


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def runtime_plan_record(plan: GenericExecutorRuntimePlan) -> dict[str, object]:
    return asdict(plan)


def prepare_result_record(result: GenericExecutorPrepareResult) -> dict[str, object]:
    return asdict(result)


def attempt_ledger_record(ledger: GenericExecutorAttemptLedger) -> dict[str, object]:
    return asdict(ledger)


def build_generic_executor_runtime_plan(
    contract: SequenceProblemContract,
    compiled_factor_shadow: dict[str, object],
) -> GenericExecutorRuntimePlan:
    records = compiled_factor_shadow.get("records", [])
    if not isinstance(records, list):
        records = []
    compiled_factor_ids = tuple(str(record.get("factor_id")) for record in records if isinstance(record, dict) and record.get("factor_id"))
    if compiled_factor_ids != contract.compiled_factor_ids:
        raise ValueError("runtime plan compiled factors must match SequenceProblemContract")
    payload = {
        "schema_version": 1,
        "executor_id": "generic_sequence_executor",
        "status": "not_executed",
        "sequence_contract_sha256": contract.canonical_sha256,
        "compiled_factor_count": contract.compiled_factor_count,
        "compiled_factor_ids": compiled_factor_ids,
        "case_dispatch_used": False,
        "solver_executed": False,
        "accepted_outputs_written": False,
    }
    return GenericExecutorRuntimePlan(
        **payload,
        canonical_sha256=_canonical_hash(payload),
    )
