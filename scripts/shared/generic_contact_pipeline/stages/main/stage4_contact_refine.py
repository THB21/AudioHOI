from __future__ import annotations

from ...core.base.config import CaseProfile
from ...core.base.io import REPO, write_json
from ...core.base.schema import stage_paths
from ...core.factors import factor_arbitration_ledger_record
from ...core.solver import (
    AcceptedObjectOutputPublisher,
    GenericSequenceExecutor,
    ObjectPublicationGate,
    SequenceOptimizationParameters,
    capability_object_problem_preparation_record,
    evaluate_object_publication_gate,
    object_publication_record,
    prepare_capability_object_problem,
    update_isolated_attempt_evidence,
    write_isolated_sequence_attempt,
)


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
    )
    preparation_record = capability_object_problem_preparation_record(prepared)
    write_json(candidate_dir / "generic_problem_preparation.json", preparation_record)
    max_evaluations = int(config.get("max_function_evaluations", 200))
    solve_result = GenericSequenceExecutor().solve(
        prepared.preparation.problem,
        SequenceOptimizationParameters(max_function_evaluations=max_evaluations),
    )
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
    }
    write_json(stage_paths(profile)["stage4_metrics"], metrics)
    return metrics
