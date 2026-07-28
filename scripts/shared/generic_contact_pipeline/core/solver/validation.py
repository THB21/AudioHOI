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
        if not str(attempt_plan.get("attempt_id", "")).startswith("shadow-"):
            errors.append("attempt_id must be deterministic shadow id")

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
