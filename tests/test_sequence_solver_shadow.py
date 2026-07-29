from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.shared.generic_contact_pipeline.core.base.config import load_case_profile
from scripts.shared.generic_contact_pipeline.core import solver as solver_api
from scripts.shared.generic_contact_pipeline.core.evaluation.legacy_ball_residual_inputs import (
    build_legacy_ball_residual_input_bundle,
)
from scripts.shared.generic_contact_pipeline.core.solver import (
    LINE_CONTACT_SANDBOX_ARTIFACTS,
    GENERIC_OBJECT_SANDBOX_ARTIFACTS,
    SANDBOX_MANIFEST_NAME,
    build_candidate_sandbox_manifest,
    build_canonical_candidate_sandbox_summary,
    build_canonical_sequence_problem_summary,
    build_canonical_sequence_solver_diagnostics_summary,
    build_sequence_problem_shadow,
    build_sequence_problem_contract,
    build_generic_executor_runtime_plan,
    GenericSequenceExecutor,
    build_generic_residual_boundary,
    build_generic_residual_dry_run,
    build_generic_residual_execution_plan,
    build_residual_input_bundle,
    build_state_regularization_residual_inputs,
    build_world_space_contact_residual_inputs,
    build_sequence_solver_shadow_diagnostics,
    validate_candidate_sandbox_manifest,
    validate_sequence_problem_shadow,
    verify_candidate_sandbox_summary,
    verify_sequence_problem_summary,
    verify_sequence_solver_diagnostics_summary,
    write_candidate_sandbox_manifest,
)
from scripts.shared.generic_contact_pipeline.core.state import SphereGeometryProvider


REPO = Path(__file__).resolve().parents[1]
CASE_DIRECTORIES = {
    "basketball": "01_basketball",
    "football": "10_football",
    "mug": "02_mug",
    "chair": "05_chair",
    "stick": "11_stick",
}


def test_sequence_problem_shadow_is_plan_only_and_never_consumes_legacy_pose() -> None:
    result_dir = REPO / "samples_known_object/01_basketball/results/benchmark_vlm_qwen"
    problem = build_sequence_problem_shadow(load_case_profile("basketball"), result_dir)
    assert problem["mode"] == "generic_sequence_solver_shadow"
    assert problem["solver_executed"] is False
    assert problem["accepted_outputs_written"] is False
    assert problem["baseline_pose_read"] is False
    assert problem["state_contract"]["baseline_pose_read"] is False
    assert problem["inputs"]["interaction_state_shadow"]["consumed_by_solver"] is False
    assert problem["inputs"]["interaction_state_shadow"]["metrics"]["final_pose_read"] is False
    assert problem["inputs"]["factor_activation_shadow"]["consumed_by_solver"] is False
    assert problem["inputs"]["factor_activation_shadow"]["record_count"] == problem["inputs"]["factor_shadow"]["factor_count"]
    assert problem["inputs"]["compiled_factor_shadow"]["consumed_by_solver"] is False
    assert problem["inputs"]["compiled_factor_shadow"]["count"] == problem["inputs"]["factor_shadow"]["factor_count"]
    assert problem["sequence_problem_contract"]["consumed_by_solver"] is False
    assert problem["sequence_problem_contract"]["compiled_factor_count"] == problem["inputs"]["compiled_factor_shadow"]["count"]
    assert problem["runtime_plan"]["executor_id"] == "generic_sequence_executor"
    assert problem["runtime_plan"]["case_dispatch_used"] is False
    assert problem["runtime_plan"]["solver_executed"] is False
    assert problem["executor_prepare"]["status"] == "prepared_not_executed"
    assert problem["executor_prepare"]["case_dispatch_used"] is False
    assert problem["executor_prepare"]["solver_executed"] is False
    assert problem["attempt_ledger"]["attempt_id"].startswith("generic-attempt-")
    assert problem["attempt_ledger"]["case_dispatch_used"] is False
    assert problem["attempt_ledger"]["residual_evaluation_status"] == "not_executed"
    assert problem["residual_boundary"]["case_dispatch_used"] is False
    assert problem["residual_boundary"]["residuals_executed"] is False
    assert problem["residual_boundary"]["supported_count"] + problem["residual_boundary"]["pending_count"] == problem["inputs"]["compiled_factor_shadow"]["count"]
    if problem["residual_boundary"]["pending_count"]:
        assert problem["residual_boundary"]["pending_gap_records"]
    assert problem["residual_execution_plan"]["case_dispatch_used"] is False
    assert problem["residual_execution_plan"]["residuals_executed"] is False
    assert problem["residual_execution_plan"]["record_count"] == problem["inputs"]["compiled_factor_shadow"]["count"]
    assert problem["residual_execution_plan"]["ready_count"] == problem["residual_boundary"]["supported_count"]
    assert problem["residual_execution_plan"]["blocked_count"] == problem["residual_boundary"]["pending_count"]
    assert problem["attempt_plan"]["attempt_id"] == problem["attempt_ledger"]["attempt_id"]
    assert problem["attempt_plan"]["writes"] == []
    assert problem["attempt_plan"]["initializer_status"] == "not_executed"
    assert validate_sequence_problem_shadow(problem) == []


def test_sequence_problem_contract_is_generic_executor_input_boundary() -> None:
    state_contract = {
        "spec_id": "rigid6:suitcase_mesh",
        "geometry_kind": "rigid_mesh",
        "required_dofs": ["root.translation", "root.rotation"],
    }
    measurement_shadow = {"measurements": {"count": 12, "canonical_sha256": "m" * 64}}
    contact_shadow = {"constraints": {"count": 4, "canonical_sha256": "c" * 64}}
    interaction_shadow = {"frame_count": 12, "canonical_sha256": "i" * 64}
    compiled_factor_shadow = {
        "count": 2,
        "canonical_sha256": "f" * 64,
        "records": [
            {"factor_id": "point_reprojection:center"},
            {"factor_id": "contact_distance:handle"},
        ],
    }

    contract = build_sequence_problem_contract(
        sample_id="heldout_suitcase",
        state_contract=state_contract,
        measurement_shadow=measurement_shadow,
        contact_shadow=contact_shadow,
        interaction_shadow=interaction_shadow,
        compiled_factor_shadow=compiled_factor_shadow,
    )

    assert contract.state_spec_id == "rigid6:suitcase_mesh"
    assert contract.geometry_kind == "rigid_mesh"
    assert contract.compiled_factor_ids == ("point_reprojection:center", "contact_distance:handle")
    assert contract.consumed_by_solver is False
    assert all(name not in contract.canonical_sha256 for name in ("basketball", "football", "mug", "chair", "stick"))


def test_generic_executor_runtime_plan_uses_contract_not_case_dispatch() -> None:
    state_contract = {
        "spec_id": "translation3:small_sphere",
        "geometry_kind": "sphere",
        "required_dofs": ["root.translation"],
    }
    measurement_shadow = {"measurements": {"count": 4, "canonical_sha256": "m" * 64}}
    contact_shadow = {"constraints": {"count": 1, "canonical_sha256": "c" * 64}}
    interaction_shadow = {"frame_count": 4, "canonical_sha256": "i" * 64}
    compiled_factor_shadow = {
        "count": 1,
        "canonical_sha256": "f" * 64,
        "records": [{"factor_id": "point_reprojection:center"}],
    }
    contract = build_sequence_problem_contract(
        sample_id="heldout_pingpong",
        state_contract=state_contract,
        measurement_shadow=measurement_shadow,
        contact_shadow=contact_shadow,
        interaction_shadow=interaction_shadow,
        compiled_factor_shadow=compiled_factor_shadow,
    )
    plan = build_generic_executor_runtime_plan(contract, compiled_factor_shadow)

    assert plan.executor_id == "generic_sequence_executor"
    assert plan.status == "not_executed"
    assert plan.case_dispatch_used is False
    assert plan.solver_executed is False
    assert plan.accepted_outputs_written is False
    assert plan.compiled_factor_ids == ("point_reprojection:center",)


def test_generic_sequence_executor_prepare_is_contract_only_and_non_executing() -> None:
    state_contract = {
        "spec_id": "rigid6:heldout_mesh",
        "geometry_kind": "rigid_mesh",
        "required_dofs": ["root.translation", "root.rotation"],
    }
    measurement_shadow = {"measurements": {"count": 8, "canonical_sha256": "m" * 64}}
    contact_shadow = {"constraints": {"count": 2, "canonical_sha256": "c" * 64}}
    interaction_shadow = {"frame_count": 8, "canonical_sha256": "i" * 64}
    compiled_factor_shadow = {
        "count": 2,
        "canonical_sha256": "f" * 64,
        "records": [
            {"factor_id": "point_reprojection:center"},
            {"factor_id": "support_and_penetration:floor"},
        ],
    }
    contract = build_sequence_problem_contract(
        sample_id="heldout_mesh_object",
        state_contract=state_contract,
        measurement_shadow=measurement_shadow,
        contact_shadow=contact_shadow,
        interaction_shadow=interaction_shadow,
        compiled_factor_shadow=compiled_factor_shadow,
    )
    runtime_plan = build_generic_executor_runtime_plan(contract, compiled_factor_shadow)
    executor = GenericSequenceExecutor()
    prepared = executor.prepare(contract, runtime_plan, compiled_factor_shadow)
    attempt = executor.plan_attempt(contract, runtime_plan, prepared)

    assert prepared.executor_id == "generic_sequence_executor"
    assert prepared.status == "prepared_not_executed"
    assert prepared.case_dispatch_used is False
    assert prepared.solver_executed is False
    assert prepared.accepted_outputs_written is False
    assert prepared.compiled_factor_count == 2
    assert attempt.attempt_id.startswith("generic-attempt-")
    assert attempt.status == "attempt_planned_not_executed"
    assert attempt.case_dispatch_used is False
    assert attempt.residual_evaluation_status == "not_executed"
    assert attempt.solver_executed is False
    assert attempt.accepted_outputs_written is False


def test_generic_residual_boundary_maps_compiled_factors_without_executing() -> None:
    state_contract = {
        "spec_id": "translation3:heldout_sphere",
        "geometry_kind": "sphere",
        "required_dofs": ["root.translation"],
    }
    measurement_shadow = {"measurements": {"count": 6, "canonical_sha256": "m" * 64}}
    contact_shadow = {"constraints": {"count": 1, "canonical_sha256": "c" * 64}}
    interaction_shadow = {"frame_count": 6, "canonical_sha256": "i" * 64}
    compiled_factor_shadow = {
        "count": 2,
        "canonical_sha256": "f" * 64,
        "records": [
            {"factor_id": "point_reprojection:center", "residual_fn_ref": "shadow_residual::point_reprojection"},
            {"factor_id": "audio_event_prior:impact", "residual_fn_ref": "shadow_residual::audio_event_prior"},
        ],
    }
    contract = build_sequence_problem_contract(
        sample_id="heldout_audio_sphere",
        state_contract=state_contract,
        measurement_shadow=measurement_shadow,
        contact_shadow=contact_shadow,
        interaction_shadow=interaction_shadow,
        compiled_factor_shadow=compiled_factor_shadow,
    )
    runtime_plan = build_generic_executor_runtime_plan(contract, compiled_factor_shadow)
    executor = GenericSequenceExecutor()
    prepared = executor.prepare(contract, runtime_plan, compiled_factor_shadow)
    attempt = executor.plan_attempt(contract, runtime_plan, prepared)
    boundary = build_generic_residual_boundary(attempt, compiled_factor_shadow)

    assert boundary.status == "planned_not_executed"
    assert boundary.supported_count == 2
    assert boundary.pending_count == 0
    assert boundary.pending_gap_records == ()
    assert boundary.case_dispatch_used is False
    assert boundary.residuals_executed is False
    by_factor = {record.factor_id: record for record in boundary.records}
    assert by_factor["point_reprojection:center"].evaluator_ref == "FactorResidualEvaluator.point_reprojection"
    assert by_factor["audio_event_prior:impact"].evaluator_ref == "FactorResidualEvaluator.audio_event_prior"
    assert by_factor["audio_event_prior:impact"].status == "supported_not_executed"


def test_generic_residual_execution_plan_records_inputs_without_executing() -> None:
    state_contract = {
        "spec_id": "translation3:heldout_sphere",
        "geometry_kind": "sphere",
        "required_dofs": ["root.translation"],
    }
    measurement_shadow = {"measurements": {"count": 6, "canonical_sha256": "m" * 64}}
    contact_shadow = {"constraints": {"count": 1, "canonical_sha256": "c" * 64}}
    interaction_shadow = {"frame_count": 6, "canonical_sha256": "i" * 64}
    compiled_factor_shadow = {
        "count": 1,
        "canonical_sha256": "f" * 64,
        "records": [
            {
                "factor_id": "audio_event_prior:impact",
                "kind": "audio_event_prior",
                "residual_fn_ref": "shadow_residual::audio_event_prior",
                "input_ids": [
                    "state:StateSpec:root",
                    "measurement:AudioEventIR:audio_events",
                    "constraint:ContactConstraintIR:audio_contact_phase",
                ],
                "gate_provenance": ["activation_policy:audio_event_aligned"],
            },
        ],
    }
    contract = build_sequence_problem_contract(
        sample_id="heldout_audio_sphere",
        state_contract=state_contract,
        measurement_shadow=measurement_shadow,
        contact_shadow=contact_shadow,
        interaction_shadow=interaction_shadow,
        compiled_factor_shadow=compiled_factor_shadow,
    )
    runtime_plan = build_generic_executor_runtime_plan(contract, compiled_factor_shadow)
    executor = GenericSequenceExecutor()
    prepared = executor.prepare(contract, runtime_plan, compiled_factor_shadow)
    attempt = executor.plan_attempt(contract, runtime_plan, prepared)
    boundary = build_generic_residual_boundary(attempt, compiled_factor_shadow)
    residual_plan = build_generic_residual_execution_plan(attempt, compiled_factor_shadow, boundary)

    assert residual_plan.status == "planned_not_executed"
    assert residual_plan.residuals_executed is False
    assert residual_plan.case_dispatch_used is False
    assert residual_plan.ready_count == 1
    assert residual_plan.blocked_count == 0
    record = residual_plan.records[0]
    assert record.factor_id == "audio_event_prior:impact"
    assert record.evaluator_ref == "FactorResidualEvaluator.audio_event_prior"
    assert record.input_ids == (
        "state:StateSpec:root",
        "measurement:AudioEventIR:audio_events",
        "constraint:ContactConstraintIR:audio_contact_phase",
    )
    assert record.gate_provenance == ("activation_policy:audio_event_aligned",)
    assert record.status == "ready_not_executed"


def test_generic_residual_dry_run_executes_factor_values_without_solving() -> None:
    state_contract = {
        "spec_id": "translation3:heldout_sphere",
        "geometry_kind": "sphere",
        "required_dofs": ["root.translation"],
    }
    measurement_shadow = {"measurements": {"count": 2, "canonical_sha256": "m" * 64}}
    contact_shadow = {"constraints": {"count": 1, "canonical_sha256": "c" * 64}}
    interaction_shadow = {"frame_count": 2, "canonical_sha256": "i" * 64}
    compiled_factor_shadow = {
        "count": 1,
        "canonical_sha256": "f" * 64,
        "records": [
            {
                "factor_id": "point_reprojection:center",
                "kind": "point_reprojection",
                "residual_fn_ref": "shadow_residual::point_reprojection",
                "input_ids": ["state:StateSpec:root", "measurement:MeasurementIR:visual_observation"],
                "gate_provenance": ["activation_policy:visible_free"],
            },
        ],
    }
    contract = build_sequence_problem_contract(
        sample_id="heldout_sphere",
        state_contract=state_contract,
        measurement_shadow=measurement_shadow,
        contact_shadow=contact_shadow,
        interaction_shadow=interaction_shadow,
        compiled_factor_shadow=compiled_factor_shadow,
    )
    runtime_plan = build_generic_executor_runtime_plan(contract, compiled_factor_shadow)
    executor = GenericSequenceExecutor()
    prepared = executor.prepare(contract, runtime_plan, compiled_factor_shadow)
    attempt = executor.plan_attempt(contract, runtime_plan, prepared)
    boundary = build_generic_residual_boundary(attempt, compiled_factor_shadow)
    residual_plan = build_generic_residual_execution_plan(attempt, compiled_factor_shadow, boundary)

    dry_run = build_generic_residual_dry_run(
        residual_plan,
        {
            "point_reprojection:center": {
                "predicted": [[12.0, 21.0], [29.0, 41.0]],
                "target": [[10.0, 20.0], [30.0, 40.0]],
                "weight": 2.0,
                "sigma_px": 5.0,
            }
        },
    )

    assert dry_run.status == "residuals_executed_dry_run"
    assert dry_run.residuals_executed is True
    assert dry_run.solver_executed is False
    assert dry_run.accepted_outputs_written is False
    assert dry_run.executed_count == 1
    assert dry_run.skipped_count == 0
    record = dry_run.records[0]
    assert record.factor_id == "point_reprojection:center"
    assert record.status == "executed"
    assert record.residual_count == 4
    assert record.residual_sha256
    assert record.rms > 0.0


def test_residual_input_bundle_resolves_providers_by_residual_capability() -> None:
    requests = []

    def point_provider(request):
        requests.append(request)
        return {
            "predicted": [[12.0, 21.0]],
            "target": [[10.0, 20.0]],
            "weight": 2.0,
            "sigma_px": 5.0,
        }

    bundle = build_residual_input_bundle(
        {
            "records": [
                {
                    "factor_id": "point_reprojection:center",
                    "residual_fn_ref": "shadow_residual::point_reprojection",
                    "evaluator_ref": "FactorResidualEvaluator.point_reprojection",
                    "input_ids": ["measurement:measurement_ir:center_track"],
                    "gate_provenance": ["gate_axis:visibility_state"],
                    "status": "ready_not_executed",
                },
                {
                    "factor_id": "contact_distance:hand",
                    "residual_fn_ref": "shadow_residual::contact_distance",
                    "evaluator_ref": "FactorResidualEvaluator.contact_distance",
                    "input_ids": ["constraint:contact_constraint_ir:hand"],
                    "gate_provenance": ["gate_axis:contact_state"],
                    "status": "ready_not_executed",
                },
            ]
        },
        {"shadow_residual::point_reprojection": point_provider},
    )

    assert bundle == {
        "point_reprojection:center": {
            "predicted": [[12.0, 21.0]],
            "target": [[10.0, 20.0]],
            "weight": 2.0,
            "sigma_px": 5.0,
        }
    }
    assert len(requests) == 1
    assert requests[0].factor_id == "point_reprojection:center"
    assert requests[0].residual_fn_ref == "shadow_residual::point_reprojection"
    assert requests[0].input_ids == ("measurement:measurement_ir:center_track",)
    assert requests[0].gate_provenance == ("gate_axis:visibility_state",)


def test_core_solver_api_does_not_expose_case_named_residual_builders() -> None:
    assert not hasattr(solver_api, "build_legacy_ball_residual_input_bundle")


def test_world_space_contact_inputs_use_geometry_feature_points() -> None:
    payload = build_world_space_contact_residual_inputs(
        factor_id="contact_distance:entity_edge",
        geometry_provider=SphereGeometryProvider(radius_m=1.0),
        object_states={1: (0.0, 0.0, 0.0), 2: (1.0, 0.0, 0.0)},
        source_sites={1: (2.0, 0.0, 0.0), 2: (3.0, 0.0, 0.0)},
        active_frames=(2,),
        object_feature_id="object:surface",
        weight=2.0,
        sigma_m=0.5,
    )

    assert payload == {
        "contact_distance:entity_edge": {
            "anchors": [[3.0, 0.0, 0.0]],
            "targets": [[2.0, 0.0, 0.0]],
            "weight": 2.0,
            "sigma_m": 0.5,
        }
    }


def test_legacy_ball_residual_input_bundle_executes_generic_depth_contact_temporal_blocks() -> None:
    for case_name, case_dir in (("basketball", "01_basketball"), ("football", "10_football")):
        result_dir = REPO / f"samples_known_object/{case_dir}/results/benchmark_vlm_qwen"
        problem = build_sequence_problem_shadow(load_case_profile(case_name), result_dir)
        bundle = build_legacy_ball_residual_input_bundle(result_dir, problem["residual_execution_plan"])
        dry_run = build_generic_residual_dry_run(problem["residual_execution_plan"], bundle)

        executed = {record.factor_id: record for record in dry_run.records if record.status == "executed"}
        assert "point_reprojection:measurement_ir" in executed
        assert "metric_depth:measurement_ir" in executed
        assert "contact_distance:interaction_state" in executed
        assert "temporal_acceleration:state_sequence" in executed
        assert any(factor_id.startswith("pose_prior:") for factor_id in executed)
        assert dry_run.solver_executed is False
        assert dry_run.accepted_outputs_written is False
        assert dry_run.case_dispatch_used is False
        assert all(record.residual_count > 0 for record in executed.values())
        assert all(record.rms >= 0.0 for record in executed.values())


def test_legacy_ball_contact_parity_adapter_emits_world_space_meter_sites() -> None:
    for case_name, case_dir in (("basketball", "01_basketball"), ("football", "10_football")):
        result_dir = REPO / f"samples_known_object/{case_dir}/results/benchmark_vlm_qwen"
        problem = build_sequence_problem_shadow(load_case_profile(case_name), result_dir)
        bundle = build_legacy_ball_residual_input_bundle(result_dir, problem["residual_execution_plan"])
        contact_payloads = [
            payload for factor_id, payload in bundle.items() if factor_id.startswith("contact_distance:")
        ]

        assert contact_payloads
        for payload in contact_payloads:
            assert payload["anchors"]
            assert payload["targets"]
            assert max(abs(value) for point in payload["anchors"] for value in point) < 20.0
            assert max(abs(value) for point in payload["targets"] for value in point) < 20.0


def test_state_regularization_inputs_are_generic_state_reference_contract() -> None:
    payload = build_state_regularization_residual_inputs(
        factor_id="regularization:root",
        values=((1.0, 2.0, 3.0), (1.5, 2.5, 3.5)),
        target=((0.5, 2.0, 2.0), (1.0, 2.0, 3.0)),
        scales=(1.0, 2.0, 0.5),
        weight=0.25,
    )

    assert payload == {
        "regularization:root": {
            "values": [[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]],
            "target": [[0.5, 2.0, 2.0], [1.0, 2.0, 3.0]],
            "weight": 0.25,
            "scales": [[1.0, 2.0, 0.5], [1.0, 2.0, 0.5]],
        }
    }


def test_legacy_ball_bundle_uses_generic_state_regularization_inputs() -> None:
    for case_name, case_dir in (("basketball", "01_basketball"), ("football", "10_football")):
        result_dir = REPO / f"samples_known_object/{case_dir}/results/benchmark_vlm_qwen"
        problem = build_sequence_problem_shadow(load_case_profile(case_name), result_dir)
        bundle = build_legacy_ball_residual_input_bundle(result_dir, problem["residual_execution_plan"])
        dry_run = build_generic_residual_dry_run(problem["residual_execution_plan"], bundle)

        regularization_records = [
            record
            for record in dry_run.records
            if record.factor_id.startswith("regularization:")
        ]
        assert regularization_records
        assert all(record.status == "executed" for record in regularization_records)
        assert all(record.residual_count > 0 for record in regularization_records)
        assert dry_run.skipped_count == 0


def test_sequence_problem_uses_profile_state_contract_not_object_pose_init() -> None:
    result_dir = REPO / "samples_known_object/05_chair/results/benchmark_vlm_qwen"
    problem = build_sequence_problem_shadow(load_case_profile("chair"), result_dir)
    payload = json.dumps(problem, sort_keys=True)
    assert "object_pose_init.csv" not in payload
    assert "physical6d_seed" not in payload
    assert "interaction_state_shadow" in payload
    assert problem["state_contract"]["state_model"] == "semantic_graph_6d"
    assert problem["state_contract"]["geometry_kind"] == "articulated_urdf"


def test_sequence_problem_hash_is_independent_of_absolute_worktree_prefix() -> None:
    relative = Path("samples_known_object/02_mug/results/benchmark_vlm_qwen")
    absolute = REPO / relative
    relative_problem = build_sequence_problem_shadow(load_case_profile("mug"), relative)
    absolute_problem = build_sequence_problem_shadow(load_case_profile("mug"), absolute)
    assert relative_problem["canonical_sha256"] == absolute_problem["canonical_sha256"]
    assert relative_problem["attempt_plan"]["attempt_id"] == absolute_problem["attempt_plan"]["attempt_id"]


def test_five_case_sequence_problem_matches_frozen_manifest() -> None:
    expected = json.loads((REPO / "tests/golden/sequence_problem_shadow_v1.json").read_text())
    actual = build_canonical_sequence_problem_summary()
    assert actual == expected
    assert verify_sequence_problem_summary() == []


def test_sequence_problem_validation_rejects_accepted_output_writes() -> None:
    result_dir = REPO / "samples_known_object/11_stick/results/benchmark_vlm_qwen"
    problem = build_sequence_problem_shadow(load_case_profile("stick"), result_dir)
    problem["accepted_outputs_written"] = True
    problem["attempt_plan"]["writes"] = ["object_pose_init.csv"]
    errors = validate_sequence_problem_shadow(problem)
    assert any("accepted outputs" in error for error in errors)
    assert any("accepted writes" in error for error in errors)


def test_sequence_problem_export_cli_writes_reviewable_manifest(tmp_path: Path) -> None:
    out = tmp_path / "mug_sequence_problem.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/shared/generic_contact_pipeline/tools/export_sequence_problem_shadow.py",
            "--case",
            "mug",
            "--result-dir",
            "samples_known_object/02_mug/results/benchmark_vlm_qwen",
            "--out",
            str(out),
        ],
        cwd=REPO,
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(out.read_text())
    assert json.loads(completed.stdout) == payload
    assert payload["mode"] == "generic_sequence_solver_shadow"
    assert payload["attempt_plan"]["attempt_id"].startswith("generic-attempt-")


def test_sequence_problem_verifier_cli_reports_all_cases() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/shared/generic_contact_pipeline/tools/verify_sequence_problem_shadow.py",
        ],
        cwd=REPO,
        check=True,
        text=True,
        capture_output=True,
    )
    lines = completed.stdout.strip().splitlines()
    assert len(lines) == len(CASE_DIRECTORIES)
    assert lines[0].startswith("basketball: state=translation3")
    assert any(line.startswith("chair: state=semantic_graph_6d") and "gaps=[]" in line for line in lines)


def test_sequence_solver_shadow_diagnostics_never_execute_or_write() -> None:
    result_dir = REPO / "samples_known_object/10_football/results/benchmark_vlm_qwen"
    diagnostics = build_sequence_solver_shadow_diagnostics(load_case_profile("football"), result_dir)
    assert diagnostics["status"] == "ready_for_future_shadow_solve"
    assert diagnostics["solver_executed"] is False
    assert diagnostics["accepted_outputs_written"] is False
    assert diagnostics["baseline_pose_read"] is False
    phase_statuses = {phase["phase_id"]: phase["status"] for phase in diagnostics["phases"]}
    assert phase_statuses == {
        "assemble_problem": "pass",
        "initialize_state": "not_executed",
        "solve_sequence": "not_executed",
        "evaluate_candidate": "not_executed",
    }


def test_sequence_solver_shadow_diagnostics_block_only_required_unmigrated_mechanisms() -> None:
    mug = build_sequence_solver_shadow_diagnostics(
        load_case_profile("mug"),
        REPO / "samples_known_object/02_mug/results/benchmark_vlm_qwen",
    )
    chair = build_sequence_solver_shadow_diagnostics(
        load_case_profile("chair"),
        REPO / "samples_known_object/05_chair/results/benchmark_vlm_qwen",
    )
    stick = build_sequence_solver_shadow_diagnostics(
        load_case_profile("stick"),
        REPO / "samples_known_object/11_stick/results/benchmark_vlm_qwen",
    )
    assert mug["blocking_gap_ids"] == []
    assert chair["blocking_gap_ids"] == []
    assert chair["nonblocking_gap_ids"] == []
    assert stick["blocking_gap_ids"] == []
    assert stick["nonblocking_gap_ids"] == []
    assert mug["status"] == "ready_for_future_shadow_solve"
    assert chair["status"] == "ready_for_future_shadow_solve"
    assert stick["status"] == "ready_for_future_shadow_solve"


def test_five_case_sequence_solver_diagnostics_matches_frozen_manifest() -> None:
    expected = json.loads((REPO / "tests/golden/sequence_solver_diagnostics_v1.json").read_text())
    actual = build_canonical_sequence_solver_diagnostics_summary()
    assert actual == expected
    assert verify_sequence_solver_diagnostics_summary() == []


def test_sequence_solver_diagnostics_export_cli_writes_reviewable_manifest(tmp_path: Path) -> None:
    out = tmp_path / "chair_sequence_solver_diagnostics.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/shared/generic_contact_pipeline/tools/export_sequence_solver_diagnostics.py",
            "--case",
            "chair",
            "--result-dir",
            "samples_known_object/05_chair/results/benchmark_vlm_qwen",
            "--out",
            str(out),
        ],
        cwd=REPO,
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(out.read_text())
    assert json.loads(completed.stdout) == payload
    assert payload["mode"] == "generic_sequence_solver_shadow_diagnostics"
    assert payload["blocking_gap_ids"] == []
    assert payload["nonblocking_gap_ids"] == []


def test_sequence_solver_diagnostics_verifier_cli_reports_all_cases() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/shared/generic_contact_pipeline/tools/verify_sequence_solver_diagnostics.py",
        ],
        cwd=REPO,
        check=True,
        text=True,
        capture_output=True,
    )
    lines = completed.stdout.strip().splitlines()
    assert len(lines) == len(CASE_DIRECTORIES)
    assert lines[0].startswith("basketball: status=ready_for_future_shadow_solve")
    assert any(
        line.startswith("chair: status=ready_for_future_shadow_solve")
        and "blocking_gaps=[] nonblocking_gaps=[]" in line
        for line in lines
    )


def test_candidate_sandbox_manifest_allows_ready_ball_and_chair_cases() -> None:
    basketball = build_candidate_sandbox_manifest(
        load_case_profile("basketball"),
        REPO / "samples_known_object/01_basketball/results/benchmark_vlm_qwen",
    )
    mug = build_candidate_sandbox_manifest(
        load_case_profile("mug"),
        REPO / "samples_known_object/02_mug/results/benchmark_vlm_qwen",
    )
    chair = build_candidate_sandbox_manifest(
        load_case_profile("chair"),
        REPO / "samples_known_object/05_chair/results/benchmark_vlm_qwen",
    )
    assert basketball["eligible_for_candidate_sandbox"] is True
    assert basketball["planned_artifacts"] == GENERIC_OBJECT_SANDBOX_ARTIFACTS
    assert basketball["accepted_outputs_written"] is False
    assert mug["eligible_for_candidate_sandbox"] is True
    assert mug["planned_artifacts"] == GENERIC_OBJECT_SANDBOX_ARTIFACTS
    assert chair["eligible_for_candidate_sandbox"] is True
    assert chair["planned_artifacts"] == GENERIC_OBJECT_SANDBOX_ARTIFACTS
    assert chair["blocking_gap_ids"] == []
    assert chair["nonblocking_gap_ids"] == []
    assert validate_candidate_sandbox_manifest(basketball) == []
    assert validate_candidate_sandbox_manifest(mug) == []
    assert validate_candidate_sandbox_manifest(chair) == []


def test_candidate_sandbox_manifest_allows_line_contact_as_nonblocking_compatibility() -> None:
    stick = build_candidate_sandbox_manifest(
        load_case_profile("stick"),
        REPO / "samples_known_object/11_stick/results/benchmark_vlm_qwen",
    )
    assert stick["eligible_for_candidate_sandbox"] is True
    assert stick["status"] == "sandbox_ready"
    assert stick["blocking_gap_ids"] == []
    assert stick["nonblocking_gap_ids"] == []
    assert stick["planned_artifacts"] == GENERIC_OBJECT_SANDBOX_ARTIFACTS


def test_candidate_sandbox_validation_rejects_accepted_output_names() -> None:
    manifest = build_candidate_sandbox_manifest(
        load_case_profile("football"),
        REPO / "samples_known_object/10_football/results/benchmark_vlm_qwen",
    )
    manifest["planned_artifacts"] = ["object_pose_init.csv"]
    manifest["accepted_outputs_written"] = True
    errors = validate_candidate_sandbox_manifest(manifest)
    assert any("accepted outputs" in error for error in errors)
    assert any("accepted output names" in error for error in errors)


def test_candidate_sandbox_validation_rejects_inconsistent_gap_eligibility() -> None:
    blocked = build_candidate_sandbox_manifest(
        load_case_profile("basketball"),
        REPO / "samples_known_object/01_basketball/results/benchmark_vlm_qwen",
    )
    blocked["eligible_for_candidate_sandbox"] = True
    blocked["status"] = "sandbox_ready"
    blocked["planned_artifacts"] = blocked["planned_artifacts"] or ["generic_sequence_solver_shadow_candidate.json"]
    blocked["blocking_gap_ids"] = ["synthetic_blocking_gap"]
    errors = validate_candidate_sandbox_manifest(blocked)
    assert any("eligible sandbox must not carry blocking gaps" in error for error in errors)

    ready = build_candidate_sandbox_manifest(
        load_case_profile("basketball"),
        REPO / "samples_known_object/01_basketball/results/benchmark_vlm_qwen",
    )
    ready["eligible_for_candidate_sandbox"] = False
    ready["status"] = "blocked_by_known_gaps"
    ready["planned_artifacts"] = []
    errors = validate_candidate_sandbox_manifest(ready)
    assert any("blocked sandbox must record at least one blocking gap" in error for error in errors)


@pytest.mark.parametrize("case_name", ["basketball", "football"])
def test_candidate_sandbox_materialize_writes_generic_sphere_candidate_artifacts(case_name: str, tmp_path: Path) -> None:
    candidate_dir = tmp_path / f"{case_name}_candidate"
    manifest = write_candidate_sandbox_manifest(
        load_case_profile(case_name),
        REPO / f"samples_known_object/{CASE_DIRECTORIES[case_name]}/results/benchmark_vlm_qwen",
        candidate_dir,
    )
    written = candidate_dir / SANDBOX_MANIFEST_NAME
    assert written.exists()
    assert json.loads(written.read_text()) == manifest
    assert {path.name for path in candidate_dir.iterdir()} == {
        SANDBOX_MANIFEST_NAME,
        "generic_problem_preparation.json",
        "generic_object_pose_candidate.csv",
        "generic_object_publication.json",
        "generic_sequence_solver_attempts",
    }
    publication = json.loads((candidate_dir / "generic_object_publication.json").read_text())
    assert publication["accepted_path"] is None
    assert publication["case_dispatch_used"] is False
    preparation = json.loads((candidate_dir / "generic_problem_preparation.json").read_text())
    assert preparation["case_dispatch_used"] is False
    assert preparation["baseline_pose_read"] is False
    assert not (candidate_dir / "object_pose_init.csv").exists()
    assert not (candidate_dir / "object_pose.csv").exists()


def test_candidate_sandbox_materialize_writes_chair_safe_candidate_artifacts(tmp_path: Path) -> None:
    candidate_dir = tmp_path / "chair_candidate"
    manifest = write_candidate_sandbox_manifest(
        load_case_profile("chair"),
        REPO / "samples_known_object/05_chair/results/benchmark_vlm_qwen",
        candidate_dir,
    )

    assert manifest["eligible_for_candidate_sandbox"] is True
    assert {path.name for path in candidate_dir.iterdir()} == set(GENERIC_OBJECT_SANDBOX_ARTIFACTS)
    publication = json.loads((candidate_dir / "generic_object_publication.json").read_text())
    assert publication["status"] == "candidate_blocked"
    assert publication["case_dispatch_used"] is False
    assert publication["human_state_optimized"] is False
    assert not (candidate_dir / "object_pose.csv").exists()
    assert not (candidate_dir / "object_contact_points.csv").exists()


def test_candidate_sandbox_materialize_writes_mug_safe_candidate_artifacts(tmp_path: Path) -> None:
    candidate_dir = tmp_path / "mug_candidate"
    manifest = write_candidate_sandbox_manifest(
        load_case_profile("mug"),
        REPO / "samples_known_object/02_mug/results/benchmark_vlm_qwen",
        candidate_dir,
    )

    assert manifest["eligible_for_candidate_sandbox"] is True
    assert {path.name for path in candidate_dir.iterdir()} == set(GENERIC_OBJECT_SANDBOX_ARTIFACTS)
    preparation = json.loads((candidate_dir / "generic_problem_preparation.json").read_text())
    assert preparation["initializer_kind"] == "observation_periodic_rigid"
    assert preparation["baseline_pose_read"] is False
    assert preparation["human_state_optimized"] is False
    assert not (candidate_dir / "object_pose.csv").exists()
    assert not (candidate_dir / "object_phase.csv").exists()


def test_candidate_sandbox_materialize_writes_stick_line_contact_safe_candidate_artifacts(tmp_path: Path) -> None:
    candidate_dir = tmp_path / "stick_candidate"
    manifest = write_candidate_sandbox_manifest(
        load_case_profile("stick"),
        REPO / "samples_known_object/11_stick/results/benchmark_vlm_qwen",
        candidate_dir,
    )

    assert manifest["eligible_for_candidate_sandbox"] is True
    assert manifest["nonblocking_gap_ids"] == []
    assert {path.name for path in candidate_dir.iterdir()} == set(GENERIC_OBJECT_SANDBOX_ARTIFACTS)
    preparation = json.loads((candidate_dir / "generic_problem_preparation.json").read_text())
    assert preparation["initializer_kind"] == "line_s_two_site"
    assert preparation["baseline_pose_read"] is False
    assert preparation["human_state_optimized"] is False
    assert not (candidate_dir / "object_pose.csv").exists()
    assert not (candidate_dir / "object_contact_points.csv").exists()


def test_five_case_candidate_sandbox_matches_frozen_manifest() -> None:
    expected = json.loads((REPO / "tests/golden/sequence_candidate_sandbox_v1.json").read_text())
    actual = build_canonical_candidate_sandbox_summary()
    assert actual == expected
    assert verify_candidate_sandbox_summary() == []


def test_candidate_sandbox_export_cli_writes_reviewable_manifest(tmp_path: Path) -> None:
    out = tmp_path / "football_candidate_sandbox.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/shared/generic_contact_pipeline/tools/export_candidate_sandbox.py",
            "--case",
            "football",
            "--result-dir",
            "samples_known_object/10_football/results/benchmark_vlm_qwen",
            "--out",
            str(out),
        ],
        cwd=REPO,
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(out.read_text())
    assert json.loads(completed.stdout) == payload
    assert payload["mode"] == "generic_sequence_solver_candidate_sandbox"
    assert payload["planned_artifacts"] == GENERIC_OBJECT_SANDBOX_ARTIFACTS


def test_candidate_sandbox_verifier_cli_reports_all_cases() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/shared/generic_contact_pipeline/tools/verify_candidate_sandbox.py",
        ],
        cwd=REPO,
        check=True,
        text=True,
        capture_output=True,
    )
    lines = completed.stdout.strip().splitlines()
    assert len(lines) == len(CASE_DIRECTORIES)
    assert lines[0].startswith("basketball: status=sandbox_ready eligible=True")
    assert any("chair: status=sandbox_ready eligible=True" in line for line in lines)


def test_candidate_sandbox_verifier_cli_materializes_and_verifies_chair_candidate(tmp_path: Path) -> None:
    candidate_root = tmp_path / "candidate_root"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/shared/generic_contact_pipeline/tools/verify_candidate_sandbox.py",
            "--materialize-chair-candidates",
            "--candidate-root",
            str(candidate_root),
        ],
        cwd=REPO,
        check=True,
        text=True,
        capture_output=True,
    )

    chair_dir = candidate_root / "benchmark_vlm_qwen_chair"
    assert "chair_materialized=True" in completed.stdout
    assert (chair_dir / "generic_object_pose_candidate.csv").exists()
    assert (chair_dir / "generic_object_publication.json").exists()
    assert not (chair_dir / "object_pose.csv").exists()


def test_candidate_sandbox_verifier_cli_materializes_and_verifies_mug_candidate(tmp_path: Path) -> None:
    candidate_root = tmp_path / "candidate_root"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/shared/generic_contact_pipeline/tools/verify_candidate_sandbox.py",
            "--materialize-mug-candidates",
            "--candidate-root",
            str(candidate_root),
        ],
        cwd=REPO,
        check=True,
        text=True,
        capture_output=True,
    )

    mug_dir = candidate_root / "benchmark_vlm_qwen_mug"
    assert "mug_materialized=True" in completed.stdout
    assert (mug_dir / "generic_object_pose_candidate.csv").exists()
    assert (mug_dir / "generic_object_publication.json").exists()
    assert not (mug_dir / "object_pose.csv").exists()
    assert not (mug_dir / "object_phase.csv").exists()


def test_candidate_sandbox_verifier_cli_materializes_and_verifies_sphere_candidates(tmp_path: Path) -> None:
    candidate_root = tmp_path / "candidate_root"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/shared/generic_contact_pipeline/tools/verify_candidate_sandbox.py",
            "--materialize-sphere-candidates",
            "--candidate-root",
            str(candidate_root),
        ],
        cwd=REPO,
        check=True,
        text=True,
        capture_output=True,
    )

    for case_name in ("basketball", "football"):
        candidate_dir = candidate_root / f"benchmark_vlm_qwen_{case_name}"
        assert f"{case_name}_materialized=True" in completed.stdout
        assert (candidate_dir / "generic_object_pose_candidate.csv").exists()
        assert (candidate_dir / "generic_object_publication.json").exists()
        assert not (candidate_dir / "object_pose.csv").exists()


def test_candidate_sandbox_verifier_cli_materializes_all_supported_candidates(tmp_path: Path) -> None:
    candidate_root = tmp_path / "candidate_root"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/shared/generic_contact_pipeline/tools/verify_candidate_sandbox.py",
            "--materialize-all-candidates",
            "--candidate-root",
            str(candidate_root),
        ],
        cwd=REPO,
        check=True,
        text=True,
        capture_output=True,
    )

    expected_artifacts = {
        "basketball": ("generic_object_pose_candidate.csv", "generic_object_publication.json"),
        "football": ("generic_object_pose_candidate.csv", "generic_object_publication.json"),
            "mug": ("generic_object_pose_candidate.csv", "generic_object_publication.json"),
            "chair": ("generic_object_pose_candidate.csv", "generic_object_publication.json"),
            "stick": ("generic_object_pose_candidate.csv", "generic_object_publication.json"),
    }
    for case_name, artifacts in expected_artifacts.items():
        candidate_dir = candidate_root / f"benchmark_vlm_qwen_{case_name}"
        assert f"{case_name}_materialized=True" in completed.stdout
        for artifact in artifacts:
            assert (candidate_dir / artifact).exists()
        assert not (candidate_dir / "object_pose.csv").exists()

    assert "stick_materialized=True" in completed.stdout


def test_candidate_sandbox_verifier_cli_checks_materialized_candidate_golden(tmp_path: Path) -> None:
    candidate_root = tmp_path / "candidate_root"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/shared/generic_contact_pipeline/tools/verify_candidate_sandbox.py",
            "--materialize-all-candidates",
            "--candidate-root",
            str(candidate_root),
            "--materialized-golden",
            "tests/golden/sequence_candidate_materialized_v1.json",
        ],
        cwd=REPO,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "materialized_golden_verified=True" in completed.stdout
