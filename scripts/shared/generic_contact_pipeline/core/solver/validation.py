from __future__ import annotations

from pathlib import Path

from ..base.io import REPO


def _check_repo_relative_existing(path_value: object, label: str) -> list[str]:
    path = str(path_value or "")
    if not path:
        return [f"{label}: missing path"]
    if Path(path).is_absolute():
        return [f"{label}: path must be repo-relative"]
    if not (REPO / path).exists():
        return [f"{label}: path does not exist: {path}"]
    return []


def validate_sequence_problem_shadow(problem: dict[str, object]) -> list[str]:
    errors: list[str] = []
    if problem.get("mode") != "generic_sequence_solver_shadow":
        errors.append("sequence problem mode must be generic_sequence_solver_shadow")
    if problem.get("solver_executed") is not False:
        errors.append("generic sequence solver must not execute in this shadow branch")
    if problem.get("accepted_outputs_written") is not False:
        errors.append("shadow branch must not write accepted outputs")
    if problem.get("baseline_pose_read") is not False:
        errors.append("shadow branch must not read baseline or legacy solved poses")

    state_contract = problem.get("state_contract", {})
    if not isinstance(state_contract, dict):
        errors.append("state_contract must be recorded")
    else:
        if state_contract.get("baseline_pose_read") is not False:
            errors.append("state_contract must not read baseline pose")
        if not state_contract.get("spec_id"):
            errors.append("state_contract missing spec_id")
        if not state_contract.get("required_dofs"):
            errors.append("state_contract missing required_dofs")

    sequence_contract = problem.get("sequence_problem_contract", {})
    if not isinstance(sequence_contract, dict):
        errors.append("sequence_problem_contract must be recorded")
    else:
        if sequence_contract.get("consumed_by_solver") is not False:
            errors.append("sequence_problem_contract must remain unconsumed")
        if not sequence_contract.get("canonical_sha256"):
            errors.append("sequence_problem_contract missing canonical_sha256")
        if not isinstance(state_contract, dict) or sequence_contract.get("state_spec_id") != state_contract.get("spec_id"):
            errors.append("sequence_problem_contract state_spec_id must match state_contract spec_id")

    runtime_plan = problem.get("runtime_plan", {})
    if not isinstance(runtime_plan, dict):
        errors.append("runtime_plan must be recorded")
    else:
        if runtime_plan.get("executor_id") != "generic_sequence_executor":
            errors.append("runtime_plan executor_id must be generic_sequence_executor")
        if runtime_plan.get("status") != "not_executed":
            errors.append("runtime_plan must remain not_executed")
        if runtime_plan.get("case_dispatch_used") is not False:
            errors.append("runtime_plan must not use case dispatch")
        if runtime_plan.get("solver_executed") is not False:
            errors.append("runtime_plan must not execute solver")
        if runtime_plan.get("accepted_outputs_written") is not False:
            errors.append("runtime_plan must not write accepted outputs")
        if isinstance(sequence_contract, dict) and runtime_plan.get("sequence_contract_sha256") != sequence_contract.get("canonical_sha256"):
            errors.append("runtime_plan sequence_contract_sha256 must match sequence_problem_contract")

    executor_prepare = problem.get("executor_prepare", {})
    if not isinstance(executor_prepare, dict):
        errors.append("executor_prepare must be recorded")
    else:
        if executor_prepare.get("executor_id") != "generic_sequence_executor":
            errors.append("executor_prepare executor_id must be generic_sequence_executor")
        if executor_prepare.get("status") != "prepared_not_executed":
            errors.append("executor_prepare must remain prepared_not_executed")
        if executor_prepare.get("case_dispatch_used") is not False:
            errors.append("executor_prepare must not use case dispatch")
        if executor_prepare.get("solver_executed") is not False:
            errors.append("executor_prepare must not execute solver")
        if executor_prepare.get("accepted_outputs_written") is not False:
            errors.append("executor_prepare must not write accepted outputs")
        if isinstance(sequence_contract, dict) and executor_prepare.get("sequence_contract_sha256") != sequence_contract.get("canonical_sha256"):
            errors.append("executor_prepare sequence_contract_sha256 must match sequence_problem_contract")
        if isinstance(runtime_plan, dict) and executor_prepare.get("runtime_plan_sha256") != runtime_plan.get("canonical_sha256"):
            errors.append("executor_prepare runtime_plan_sha256 must match runtime_plan")

    attempt_ledger = problem.get("attempt_ledger", {})
    if not isinstance(attempt_ledger, dict):
        errors.append("attempt_ledger must be recorded")
    else:
        if not str(attempt_ledger.get("attempt_id", "")).startswith("generic-attempt-"):
            errors.append("attempt_ledger attempt_id must be deterministic generic-attempt id")
        if attempt_ledger.get("executor_id") != "generic_sequence_executor":
            errors.append("attempt_ledger executor_id must be generic_sequence_executor")
        if attempt_ledger.get("status") != "attempt_planned_not_executed":
            errors.append("attempt_ledger must remain attempt_planned_not_executed")
        if attempt_ledger.get("residual_evaluation_status") != "not_executed":
            errors.append("attempt_ledger residual evaluation must remain not_executed")
        if attempt_ledger.get("case_dispatch_used") is not False:
            errors.append("attempt_ledger must not use case dispatch")
        if attempt_ledger.get("solver_executed") is not False:
            errors.append("attempt_ledger must not execute solver")
        if attempt_ledger.get("accepted_outputs_written") is not False:
            errors.append("attempt_ledger must not write accepted outputs")
        if isinstance(sequence_contract, dict) and attempt_ledger.get("sequence_contract_sha256") != sequence_contract.get("canonical_sha256"):
            errors.append("attempt_ledger sequence_contract_sha256 must match sequence_problem_contract")
        if isinstance(runtime_plan, dict) and attempt_ledger.get("runtime_plan_sha256") != runtime_plan.get("canonical_sha256"):
            errors.append("attempt_ledger runtime_plan_sha256 must match runtime_plan")
        if isinstance(executor_prepare, dict) and attempt_ledger.get("prepare_sha256") != executor_prepare.get("canonical_sha256"):
            errors.append("attempt_ledger prepare_sha256 must match executor_prepare")

    inputs = problem.get("inputs", {})
    if not isinstance(inputs, dict):
        errors.append("inputs must be recorded")
    else:
        measurement = inputs.get("measurement_shadow", {})
        contact = inputs.get("contact_constraint_shadow", {})
        if isinstance(measurement, dict):
            source = measurement.get("source", {})
            if isinstance(source, dict):
                errors.extend(_check_repo_relative_existing(source.get("path"), "measurement_shadow.source"))
            if measurement.get("consumed_by_solver") is not False:
                errors.append("measurement_shadow must remain unconsumed")
        else:
            errors.append("measurement_shadow must be recorded")
        if isinstance(contact, dict):
            source = contact.get("source", {})
            if isinstance(source, dict):
                errors.extend(_check_repo_relative_existing(source.get("path"), "contact_constraint_shadow.source"))
            if contact.get("consumed_by_solver") is not False:
                errors.append("contact_constraint_shadow must remain unconsumed")
        else:
            errors.append("contact_constraint_shadow must be recorded")
        interaction = inputs.get("interaction_state_shadow", {})
        if isinstance(interaction, dict):
            if interaction.get("consumed_by_solver") is not False:
                errors.append("interaction_state_shadow must remain unconsumed")
            if not interaction.get("frame_count"):
                errors.append("interaction_state_shadow missing frame_count")
            if not interaction.get("canonical_sha256"):
                errors.append("interaction_state_shadow missing canonical_sha256")
            metrics = interaction.get("metrics", {})
            if not isinstance(metrics, dict):
                errors.append("interaction_state_shadow metrics must be recorded")
            elif metrics.get("final_pose_read") is not False:
                errors.append("interaction_state_shadow must not read final or baseline pose")
        else:
            errors.append("interaction_state_shadow must be recorded")
        factor = inputs.get("factor_shadow", {})
        if not isinstance(factor, dict):
            errors.append("factor_shadow must be recorded")
        elif factor.get("consumed_by_solver") is not False:
            errors.append("factor_shadow must remain unconsumed")
        activation = inputs.get("factor_activation_shadow", {})
        if isinstance(activation, dict):
            if activation.get("consumed_by_solver") is not False:
                errors.append("factor_activation_shadow must remain unconsumed")
            if not activation.get("record_count"):
                errors.append("factor_activation_shadow missing records")
            if not activation.get("canonical_sha256"):
                errors.append("factor_activation_shadow missing canonical_sha256")
            if not isinstance(activation.get("by_policy"), dict):
                errors.append("factor_activation_shadow missing policy summary")
        else:
            errors.append("factor_activation_shadow must be recorded")
        compiled = inputs.get("compiled_factor_shadow", {})
        if isinstance(compiled, dict):
            if compiled.get("consumed_by_solver") is not False:
                errors.append("compiled_factor_shadow must remain unconsumed")
            if not compiled.get("count"):
                errors.append("compiled_factor_shadow missing records")
            if not compiled.get("canonical_sha256"):
                errors.append("compiled_factor_shadow missing canonical_sha256")
            if isinstance(factor, dict) and compiled.get("count") != factor.get("factor_count"):
                errors.append("compiled_factor_shadow count must match factor_shadow count")
        else:
            errors.append("compiled_factor_shadow must be recorded")
        if isinstance(sequence_contract, dict):
            if isinstance(measurement, dict) and sequence_contract.get("measurement_count") != measurement.get("count"):
                errors.append("sequence_problem_contract measurement_count must match measurement_shadow count")
            if isinstance(contact, dict) and sequence_contract.get("contact_constraint_count") != contact.get("count"):
                errors.append("sequence_problem_contract contact_constraint_count must match contact_constraint_shadow count")
            if isinstance(interaction, dict) and sequence_contract.get("interaction_frame_count") != interaction.get("frame_count"):
                errors.append("sequence_problem_contract interaction_frame_count must match interaction_state_shadow frame_count")
            if isinstance(compiled, dict) and sequence_contract.get("compiled_factor_count") != compiled.get("count"):
                errors.append("sequence_problem_contract compiled_factor_count must match compiled_factor_shadow count")
        if isinstance(runtime_plan, dict) and isinstance(compiled, dict):
            if runtime_plan.get("compiled_factor_count") != compiled.get("count"):
                errors.append("runtime_plan compiled_factor_count must match compiled_factor_shadow count")
        if isinstance(executor_prepare, dict) and isinstance(compiled, dict):
            if executor_prepare.get("compiled_factor_count") != compiled.get("count"):
                errors.append("executor_prepare compiled_factor_count must match compiled_factor_shadow count")
        if isinstance(attempt_ledger, dict) and isinstance(compiled, dict):
            if attempt_ledger.get("compiled_factor_count") != compiled.get("count"):
                errors.append("attempt_ledger compiled_factor_count must match compiled_factor_shadow count")

    attempt_plan = problem.get("attempt_plan", {})
    if not isinstance(attempt_plan, dict):
        errors.append("attempt_plan must be recorded")
    else:
        if attempt_plan.get("mode") != "shadow_plan_only":
            errors.append("attempt_plan mode must be shadow_plan_only")
        if attempt_plan.get("initializer_status") != "not_executed":
            errors.append("initializer must not execute in this branch")
        if attempt_plan.get("writes") != []:
            errors.append("shadow attempt must not plan accepted writes")
        if not str(attempt_plan.get("attempt_id", "")).startswith("generic-attempt-"):
            errors.append("attempt_id must be deterministic generic-attempt id")

    factor_ids: set[str] = set()
    problem_payload = problem.get("problem", {})
    requirements = problem_payload.get("factor_requirements", []) if isinstance(problem_payload, dict) else []
    if not requirements:
        errors.append("problem must include factor requirements")
    for index, requirement in enumerate(requirements):
        if not isinstance(requirement, dict):
            errors.append(f"factor_requirement[{index}] must be a record")
            continue
        factor_id = str(requirement.get("factor_id", ""))
        if not factor_id:
            errors.append(f"factor_requirement[{index}] missing factor_id")
        elif factor_id in factor_ids:
            errors.append(f"duplicate factor requirement: {factor_id}")
        factor_ids.add(factor_id)
        if requirement.get("consumed_by_solver") is not False:
            errors.append(f"{factor_id}: requirement must not be marked consumed")
    return errors
