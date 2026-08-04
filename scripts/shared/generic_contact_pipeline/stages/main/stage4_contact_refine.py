from __future__ import annotations

from collections.abc import Mapping

from ...core.base.config import CaseProfile
from ...core.base.io import REPO, write_csv, write_json
from ...core.base.schema import stage_paths
from ...core.factors import factor_arbitration_ledger_record
from ...core.gates.interval_candidate_selection import (
    compose_interval_selected_result,
    load_interval_candidate_selection,
)
from ...components.render.backends.urdf_solid import render_candidate_overlay_evidence
from ...core.solver import (
    AcceptedObjectOutputPublisher,
    GenericSequenceExecutor,
    ObjectPublicationGate,
    SequenceOptimizationParameters,
    capability_object_problem_preparation_record,
    evaluate_object_publication_gate,
    object_publication_record,
    project_contact_facing_states,
    smooth_adjacent_states,
    repair_rotation_step_outliers,
    reevaluate_solve_result,
    prepare_capability_object_problem,
    update_isolated_attempt_evidence,
    write_isolated_sequence_attempt,
    build_runtime_residual_blocks,
    residual_execution_plan_record,
)


def _write_multimodal_factor_ledgers(profile, prepared, solve_result, candidate_dir) -> None:
    problem = prepared.preparation.problem
    states = dict(zip(solve_result.frames, solve_result.states))
    residual_inputs = problem.residual_input_builder(states)
    blocks = dict(build_runtime_residual_blocks(
        problem.residual_execution_plan,
        residual_inputs,
        problem.factor_ids,
    ))
    execution_plan = problem.residual_execution_plan
    plan_records = (
        execution_plan.get("records", ())
        if isinstance(execution_plan, Mapping)
        else getattr(execution_plan, "records", ())
    )
    records = [
        dict(record)
        if isinstance(record, Mapping)
        else residual_execution_plan_record(record)
        for record in plan_records
    ]
    write_json(
        candidate_dir / "factor_ledger.json",
        {
            "schema_version": 1,
            "records": records,
            "case_dispatch_used": False,
            "baseline_pose_read": False,
            "human_state_optimized": False,
        },
    )
    consumption = []
    for item in prepared.evidence_consumption:
        record = dict(item)
        values = blocks.get(str(record["factor_id"]))
        if values is not None:
            record["residual_count"] = int(values.size)
            record["residual_squared_error"] = float(values @ values)
            record["residual_mean_abs"] = float(abs(values).mean())
            record["residual_max_abs"] = float(abs(values).max())
        record["activation_status"] = "active"
        consumption.append(record)
    flags = set(profile.data.get("ablation_flags", ()))
    if "disable_vlm_semantic_evidence" in flags:
        consumption.append({"evidence_kind": "semantic_relation", "activation_status": "disabled_by_ablation"})
    if "disable_audio_events" in flags:
        consumption.append({"evidence_kind": "audio_event", "activation_status": "disabled_by_ablation"})
    write_json(
        candidate_dir / "evidence_consumption.json",
        {
            "schema_version": 1,
            "records": consumption,
            "case_dispatch_used": False,
            "baseline_pose_read": False,
            "human_state_optimized": False,
        },
    )
    semantic_rows = [record for record in consumption if record.get("evidence_kind") == "semantic_relation" and record.get("factor_id")]
    audio_rows = [record for record in consumption if record.get("evidence_kind") == "audio_event" and record.get("factor_id")]
    write_csv(candidate_dir / "semantic_factor_residuals.csv", semantic_rows)
    write_csv(candidate_dir / "audio_factor_residuals.csv", audio_rows)


def _quaternion_groups(prepared: object) -> tuple[tuple[int, int, int, int], ...]:
    groups: list[tuple[int, int, int, int]] = []
    offset = 0
    state_spec = prepared.state_adaptation.state_spec
    for dof in state_spec.dofs:
        if dof.kind.value == "rotation_so3":
            names = tuple(field.rsplit(".", 1)[-1] for field in dof.source_fields)
            groups.append(tuple(offset + names.index(name) for name in ("qw", "qx", "qy", "qz")))
        offset += dof.dimension
    return tuple(groups)


def _solve_prepared(prepared: object, config: dict[str, object], max_evaluations: int):
    robust_loss = str(config.get("robust_loss", "soft_l1"))
    robust_scale = float(config.get("robust_scale", 1.0))
    solve_result = GenericSequenceExecutor().solve(
        prepared.preparation.problem,
        SequenceOptimizationParameters(
            robust_loss=robust_loss,
            robust_scale=robust_scale,
            max_function_evaluations=max_evaluations,
        ),
    )
    solve_result = smooth_adjacent_states(
        prepared.preparation.problem,
        solve_result,
        translation_passes=int(getattr(prepared, "adjacent_translation_smoothing_passes", 0)),
        rotation_passes=int(getattr(prepared, "adjacent_rotation_smoothing_passes", 0)),
    )
    solve_result = repair_rotation_step_outliers(
        prepared.preparation.problem,
        solve_result,
        maximum_step_deg=float(getattr(prepared, "maximum_rotation_step_deg", 0.0)),
    )
    facing_config = config.get("contact_facing", {})
    return project_contact_facing_states(
        prepared.preparation.problem,
        solve_result,
        getattr(prepared, "contact_facing_projection", None),
        smoothing_passes=int(
            facing_config.get("projection_smoothing_passes", 0)
            if isinstance(facing_config, dict)
            else 0
        ),
        turn_trigger_half_window_frames=int(
            facing_config.get("turn_trigger_half_window_frames", 0)
            if isinstance(facing_config, dict)
            else 0
        ),
        turn_trigger_span_deg=float(
            facing_config.get("turn_trigger_span_deg", 0.0)
            if isinstance(facing_config, dict)
            else 0.0
        ),
        latch_exact_alignment_after_turn=bool(
            facing_config.get("latch_exact_alignment_after_turn", False)
            if isinstance(facing_config, dict)
            else False
        ),
        turn_alignment_ramp_frames=int(
            facing_config.get("turn_alignment_ramp_frames", 0)
            if isinstance(facing_config, dict)
            else 0
        ),
    )


def _write_role_candidate(
    profile: CaseProfile,
    prepared: object,
    solve_result: object,
    role_dir,
) -> tuple[object, dict[str, object], dict[str, object], str]:
    attempt_dir = write_isolated_sequence_attempt(
        profile.result_dir / "generic_sequence_solver_attempts",
        prepared.preparation.problem,
        solve_result,
    )
    gate, hard_metrics = evaluate_object_publication_gate(
        prepared.preparation.problem,
        solve_result,
    )
    update_isolated_attempt_evidence(
        attempt_dir,
        hard_metrics=hard_metrics,
        vlm_gates=factor_arbitration_ledger_record(prepared.factor_arbitration),
    )
    role_gate = ObjectPublicationGate(
        passed=False,
        gate_ids=("isolated_role_candidate",),
        blocking_reasons=("role_candidate_requires_interval_selection",),
    )
    publication = AcceptedObjectOutputPublisher().publish(
        result=solve_result,
        state_spec=prepared.state_adaptation.state_spec,
        template_rows=list(prepared.template_rows),
        candidate_dir=role_dir,
        accepted_result_dir=profile.result_dir,
        gate=role_gate,
    )
    evidence = render_candidate_overlay_evidence(
        profile,
        REPO / publication.candidate_path,
        role_dir,
    )
    return gate, hard_metrics, evidence, str(attempt_dir)


def run(profile: CaseProfile) -> dict[str, object]:
    """Run the single production object solver and hard-gated publisher."""

    config = profile.data.get("generic_object_problem")
    if not isinstance(config, dict):
        raise ValueError("Stage 4 requires generic_object_problem capability configuration")
    result_dir = profile.result_dir
    candidate_dir = result_dir / "generic_stage4_candidate"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    prepared = prepare_capability_object_problem(
        profile=profile,
        result_dir=result_dir,
        repository_root=REPO,
        body_models_root=REPO / "third-party/GVHMR/inputs/checkpoints/body_models",
        factor_arbitration_mode="auto",
        mask_artifact_bbox_policy_override=False,
    )
    preparation_record = capability_object_problem_preparation_record(prepared)
    write_json(candidate_dir / "generic_problem_preparation.json", preparation_record)
    max_evaluations = int(config.get("max_function_evaluations", 200))
    stable_result = _solve_prepared(prepared, config, max_evaluations)
    stable_dir = candidate_dir / "stable"
    stable_gate, stable_hard_metrics, stable_evidence, stable_attempt_dir = _write_role_candidate(
        profile,
        prepared,
        stable_result,
        stable_dir,
    )
    challenger_prepared = None
    challenger_result = None
    challenger_evidence: dict[str, object] = {"status": "not_applicable"}
    challenger_attempt_dir = ""
    if getattr(prepared, "bounded_gap_smoothing_frames", ()):
        challenger_prepared = prepare_capability_object_problem(
            profile=profile,
            result_dir=result_dir,
            repository_root=REPO,
            body_models_root=REPO / "third-party/GVHMR/inputs/checkpoints/body_models",
            factor_arbitration_mode="auto",
            mask_artifact_bbox_policy_override=True,
            line_constraint_mode_override="endpoints",
        )
        challenger_result = _solve_prepared(challenger_prepared, config, max_evaluations)
        _, _, challenger_evidence, challenger_attempt_dir = _write_role_candidate(
            profile,
            challenger_prepared,
            challenger_result,
            candidate_dir / "occlusion_challenger",
        )
    interval_ledger = load_interval_candidate_selection(result_dir=result_dir)
    composition = None
    solve_result = stable_result
    if challenger_result is not None:
        composition = compose_interval_selected_result(
            stable_result,
            challenger_result,
            interval_ledger,
            transition_frames=int(config.get("interval_candidate_transition_frames", 6)),
            quaternion_groups=_quaternion_groups(prepared),
        )
        if composition.result is not None:
            solve_result = composition.result
            if solve_result.solve_attempt_id != stable_result.solve_attempt_id:
                solve_result = reevaluate_solve_result(
                    prepared.preparation.problem,
                    solve_result,
                    policy="vlm_interval_candidate_selection",
                )
    write_json(
        candidate_dir / "attempt_roles.json",
        {
            "schema_version": 1,
            "stable_attempt_id": stable_result.solve_attempt_id,
            "stable_attempt_dir": stable_attempt_dir,
            "stable_evidence": stable_evidence,
            "challenger_attempt_id": (
                challenger_result.solve_attempt_id if challenger_result is not None else None
            ),
            "challenger_attempt_dir": challenger_attempt_dir or None,
            "challenger_evidence": challenger_evidence,
            "completion_frames": list(getattr(prepared, "bounded_gap_smoothing_frames", ())),
            "baseline_pose_read": False,
            "accepted_outputs_written": False,
        },
    )
    if composition is not None:
        write_json(candidate_dir / "interval_selection_provenance.json", composition.provenance)
    _write_multimodal_factor_ledgers(profile, prepared, solve_result, candidate_dir)
    assert prepared.case_dispatch_used is False
    assert prepared.baseline_pose_read is False
    assert prepared.human_state_optimized is False
    assert candidate_dir.resolve() != (result_dir / "object_pose.csv").resolve()
    attempt_dir = write_isolated_sequence_attempt(
        result_dir / "generic_sequence_solver_attempts",
        prepared.preparation.problem,
        solve_result,
    )
    gate, hard_metrics = evaluate_object_publication_gate(prepared.preparation.problem, solve_result)
    arbitration = factor_arbitration_ledger_record(prepared.factor_arbitration)
    if bool(arbitration.get("blocking", False)):
        gate = ObjectPublicationGate(
            passed=False,
            gate_ids=(*gate.gate_ids, "vlm_factor_arbitration_clear"),
            blocking_reasons=(*gate.blocking_reasons, "vlm_factor_arbitration_unclear"),
        )
    if interval_ledger.blocking:
        gate = ObjectPublicationGate(
            passed=False,
            gate_ids=(*gate.gate_ids, "vlm_interval_selection_clear"),
            blocking_reasons=(*gate.blocking_reasons, "vlm_interval_selection_rejected"),
        )
    update_isolated_attempt_evidence(
        attempt_dir,
        hard_metrics=hard_metrics,
        vlm_gates=arbitration,
    )
    publication = AcceptedObjectOutputPublisher().publish(
        result=solve_result,
        state_spec=prepared.state_adaptation.state_spec,
        template_rows=list(prepared.template_rows),
        candidate_dir=candidate_dir,
        accepted_result_dir=result_dir,
        gate=gate,
    )
    candidate_vlm_evidence: dict[str, object] = {}
    if publication.candidate_path is not None:
        candidate_vlm_evidence = render_candidate_overlay_evidence(
            profile,
            REPO / publication.candidate_path,
            candidate_dir / "vlm_evidence",
        )
    publication_record = object_publication_record(publication, gate)
    publication_record.update(
        {
            "attempt_dir": str(attempt_dir),
            "automatic_hard_gate": {
                "passed": gate.passed,
                "gate_ids": list(gate.gate_ids),
                "blocking_reasons": list(gate.blocking_reasons),
            },
            "object_only_boundary": True,
            "candidate_vlm_evidence": candidate_vlm_evidence,
            "interval_selection": (
                composition.provenance if composition is not None else {"status": "not_applicable"}
            ),
        }
    )
    write_json(result_dir / "generic_object_publication.json", publication_record)
    metrics = {
        "schema_version": 2,
        "stage": "stage4_contact_refine",
        "component": "generic_sequence_executor",
        "status": publication.status,
        "solve_attempt_id": solve_result.solve_attempt_id,
        "attempt_dir": str(attempt_dir),
        "candidate_path": publication.candidate_path,
        "candidate_sha256": publication.candidate_sha256,
        "accepted_path": publication.accepted_path,
        "accepted_sha256": publication.accepted_sha256,
        "hard_gate": publication_record["hard_gate"],
        "initial_squared_error": solve_result.initial_squared_error,
        "final_squared_error": solve_result.final_squared_error,
        "function_evaluations": solve_result.function_evaluations,
        "factor_ids": list(solve_result.factor_ids),
        "measurement_count": prepared.measurement_count,
        "contact_constraint_count": prepared.contact_constraint_count,
        "case_dispatch_used": False,
        "baseline_pose_read": False,
        "human_state_optimized": False,
        "accepted_outputs_written": publication.accepted_path is not None,
        "candidate_vlm_evidence": candidate_vlm_evidence,
        "stable_attempt_id": stable_result.solve_attempt_id,
        "challenger_attempt_id": (
            challenger_result.solve_attempt_id if challenger_result is not None else None
        ),
        "interval_selection": (
            composition.provenance if composition is not None else {"status": "not_applicable"}
        ),
    }
    write_json(stage_paths(profile)["stage4_metrics"], metrics)
    return metrics
