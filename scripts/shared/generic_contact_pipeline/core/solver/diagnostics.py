from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..base.config import CaseProfile
from .problem import build_sequence_problem_shadow
from .validation import validate_sequence_problem_shadow


NONBLOCKING_COMPATIBILITY_GAPS = {
    "line_contact_lock_special_refinement",
    "unsupported_loss_term:E_audio",
}


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _phase_record(
    phase_id: str,
    status: str,
    *,
    reason: str,
    reads: list[str] | None = None,
    writes: list[str] | None = None,
    diagnostics: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "phase_id": phase_id,
        "status": status,
        "reason": reason,
        "reads": reads or [],
        "writes": writes or [],
        "diagnostics": diagnostics or {},
    }


def build_sequence_solver_shadow_diagnostics(profile: CaseProfile, result_dir: Path) -> dict[str, object]:
    problem = build_sequence_problem_shadow(profile, result_dir)
    validation_errors = validate_sequence_problem_shadow(problem)
    gap_ids = list(problem["inputs"]["factor_shadow"]["gap_ids"])
    nonblocking_gaps = [gap for gap in gap_ids if gap in NONBLOCKING_COMPATIBILITY_GAPS]
    blocking_gaps = [gap for gap in gap_ids if gap and gap not in NONBLOCKING_COMPATIBILITY_GAPS]
    ready_for_future_shadow_solve = not validation_errors and not blocking_gaps
    phases = [
        _phase_record(
            "assemble_problem",
            "pass" if not validation_errors else "failed",
            reason="typed measurement/contact/factor inputs were assembled without running a solver",
            reads=[
                "sequence_problem_contract:canonical",
                "runtime_plan:canonical",
                problem["inputs"]["measurement_shadow"]["source"]["path"],
                problem["inputs"]["contact_constraint_shadow"]["source"]["path"],
                "interaction_state_shadow:canonical",
                "factor_activation_shadow:canonical",
                "compiled_factor_shadow:canonical",
                "factor_shadow:canonical",
            ],
            diagnostics={
                "validation_error_count": len(validation_errors),
                "sequence_contract_sha256": problem["sequence_problem_contract"]["canonical_sha256"],
                "runtime_plan_sha256": problem["runtime_plan"]["canonical_sha256"],
                "case_dispatch_used": problem["runtime_plan"]["case_dispatch_used"],
                "measurement_count": problem["inputs"]["measurement_shadow"]["count"],
                "contact_count": problem["inputs"]["contact_constraint_shadow"]["count"],
                "interaction_frame_count": problem["inputs"]["interaction_state_shadow"]["frame_count"],
                "factor_activation_record_count": problem["inputs"]["factor_activation_shadow"]["record_count"],
                "compiled_factor_count": problem["inputs"]["compiled_factor_shadow"]["count"],
                "factor_count": problem["inputs"]["factor_shadow"]["factor_count"],
            },
        ),
        _phase_record(
            "initialize_state",
            "not_executed",
            reason="initializer is intentionally deferred; baseline and legacy solved poses are forbidden inputs",
            diagnostics={
                "baseline_pose_read": False,
                "initializer_policy": problem["attempt_plan"]["initializer_policy"],
                "required_dofs": problem["state_contract"]["required_dofs"],
            },
        ),
        _phase_record(
            "solve_sequence",
            "blocked_by_known_gaps" if blocking_gaps else "not_executed",
            reason=(
                "known legacy gap must be migrated before generic solve"
                if blocking_gaps
                else "solver execution is deferred to a later branch"
            ),
            diagnostics={
                "blocking_gap_ids": blocking_gaps,
                "nonblocking_gap_ids": nonblocking_gaps,
                "factor_kinds": problem["inputs"]["factor_shadow"]["factor_kinds"],
                "solver_executed": False,
            },
        ),
        _phase_record(
            "evaluate_candidate",
            "not_executed",
            reason="no candidate pose is produced in shadow diagnostics",
            diagnostics={
                "accepted_outputs_written": False,
                "candidate_outputs": [],
            },
        ),
    ]
    status = "ready_for_future_shadow_solve" if ready_for_future_shadow_solve else "blocked_by_known_gaps"
    canonical_payload = {
        "problem_sha256": problem["canonical_sha256"],
        "status": status,
        "phases": phases,
        "validation_errors": validation_errors,
        "blocking_gap_ids": blocking_gaps,
        "nonblocking_gap_ids": nonblocking_gaps,
    }
    return {
        "schema_version": 1,
        "mode": "generic_sequence_solver_shadow_diagnostics",
        "sample_id": profile.case_name,
        "problem_sha256": problem["canonical_sha256"],
        "attempt_id": problem["attempt_plan"]["attempt_id"],
        "status": status,
        "solver_executed": False,
        "accepted_outputs_written": False,
        "baseline_pose_read": False,
        "validation_errors": validation_errors,
        "blocking_gap_ids": blocking_gaps,
        "nonblocking_gap_ids": nonblocking_gaps,
        "phases": phases,
        "canonical_sha256": _canonical_hash(canonical_payload),
    }
