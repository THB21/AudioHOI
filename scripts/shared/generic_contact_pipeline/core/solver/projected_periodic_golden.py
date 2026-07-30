from __future__ import annotations

import hashlib
import json
import csv
from pathlib import Path

from ..base.io import REPO


DEFAULT_PROJECTED_PERIODIC_GOLDEN = REPO / "tests/golden/mug_projected_periodic_migration_v1.json"
BODY_CANDIDATE_NAME = "generic_periodic_body_candidate.csv"
PHASE_CANDIDATE_NAME = "generic_periodic_phase_candidate.csv"
PROJECTED_PERIODIC_ATTEMPT_NAME = "generic_projected_periodic_attempt.json"
PROJECTED_PERIODIC_SAFE_OUTPUTS = (
    BODY_CANDIDATE_NAME,
    PHASE_CANDIDATE_NAME,
    PROJECTED_PERIODIC_ATTEMPT_NAME,
)
ACCEPTED_OUTPUT_NAMES = {
    "object_pose_init.csv",
    "object_pose.csv",
    "object_phase.csv",
    "handle_phase.csv",
    "object_contact_points.csv",
}
SWITCHED_RESULT_NAME = "generic_mug_periodic_switched_v1"
SWITCHED_OUTPUTS = (
    "observation_seed/body_pose.csv",
    "observation_seed/axial_phase.csv",
    "object_observations.csv",
    "object_local_points.csv",
    "object_pose_init.csv",
    "object_pose_pre_smooth.csv",
    "object_pose.csv",
    "object_phase.csv",
    "object_contact_points.csv",
)
SWITCHED_RENDERS = (
    "object_only/overlay.mp4",
    "object_only/camera3d.mp4",
    "object_only/side_yz.mp4",
    "with_human/overlay.mp4",
    "with_human/camera3d.mp4",
    "with_human/side_yz.mp4",
)
SWITCHED_LOSS_ARTIFACTS = (
    "loss_analysis/loss_summary.json",
    "loss_analysis/loss_trace.csv",
    "loss_analysis/per_frame_residuals.csv",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    return payload if isinstance(payload, dict) else {}


def validate_projected_periodic_candidate_attempt(attempt: dict[str, object]) -> list[str]:
    errors: list[str] = []
    if attempt.get("mode") != "generic_projected_periodic_candidate":
        errors.append("projected-periodic candidate mode must be generic_projected_periodic_candidate")
    if attempt.get("solver_executed") is not True:
        errors.append("projected-periodic candidate must execute solver")
    if attempt.get("executor_scope") != "isolated_candidate_dir":
        errors.append("projected-periodic candidate must record isolated_candidate_dir scope")
    if attempt.get("accepted_outputs_written") is not False:
        errors.append("projected-periodic candidate must not write accepted outputs")
    if attempt.get("baseline_pose_read") is not False:
        errors.append("projected-periodic candidate must not read baseline pose")
    if attempt.get("historical_phase_read") is not False:
        errors.append("projected-periodic candidate must not read historical phase")
    if attempt.get("body_candidate_artifact") != BODY_CANDIDATE_NAME:
        errors.append("projected-periodic body candidate artifact name is not safe")
    if attempt.get("phase_candidate_artifact") != PHASE_CANDIDATE_NAME:
        errors.append("projected-periodic phase candidate artifact name is not safe")
    if int(attempt.get("frames", 0) or 0) <= 0:
        errors.append("projected-periodic candidate must record positive frame count")
    return errors


def verify_materialized_projected_periodic_candidate(candidate_dir: Path) -> list[str]:
    errors: list[str] = []
    for name in sorted(ACCEPTED_OUTPUT_NAMES):
        if (candidate_dir / name).exists():
            errors.append(f"accepted output artifact present in projected-periodic candidate dir: {name}")
    for name in PROJECTED_PERIODIC_SAFE_OUTPUTS:
        if not (candidate_dir / name).exists():
            errors.append(f"missing planned output {name}")

    attempt_path = candidate_dir / PROJECTED_PERIODIC_ATTEMPT_NAME
    attempt = _load_json(attempt_path)
    if not attempt:
        return errors + [f"missing or invalid attempt manifest {PROJECTED_PERIODIC_ATTEMPT_NAME}"]
    errors.extend(validate_projected_periodic_candidate_attempt(attempt))

    body_path = candidate_dir / BODY_CANDIDATE_NAME
    phase_path = candidate_dir / PHASE_CANDIDATE_NAME
    if body_path.exists():
        body_rows = _read_csv(body_path)
        if len(body_rows) != int(attempt.get("frames", 0) or 0):
            errors.append("projected-periodic body row count does not match attempt frames")
        if _sha256(body_path) != attempt.get("body_candidate_sha256"):
            errors.append("projected-periodic body candidate sha256 does not match attempt")
    if phase_path.exists():
        phase_rows = _read_csv(phase_path)
        if len(phase_rows) != int(attempt.get("frames", 0) or 0):
            errors.append("projected-periodic phase row count does not match attempt frames")
        if _sha256(phase_path) != attempt.get("phase_candidate_sha256"):
            errors.append("projected-periodic phase candidate sha256 does not match attempt")
    return errors


def _latest_stage_attempt(result_dir: Path, stage: str) -> dict[str, object]:
    active = json.loads((result_dir / "provenance/stages" / stage / "active_attempt.json").read_text())
    record_path = Path(str(active["record"]))
    if not record_path.is_absolute():
        record_path = REPO / record_path
    record = json.loads(record_path.read_text())
    return {
        "status": record["status"],
        "contract_status": record["contract_audit"]["status"],
        "artifact_count": len(record["artifacts_after"]),
        "stored_artifact_count": len(record["stored_artifacts"]),
    }


def _switched_pipeline_summary(results_dir: Path) -> dict[str, object]:
    result_dir = results_dir / SWITCHED_RESULT_NAME
    render_dir = results_dir / "renders" / SWITCHED_RESULT_NAME
    stage6 = json.loads((result_dir / "stage6_compare_report.json").read_text())["checks"]
    stage1_active = json.loads((result_dir / "provenance/stages/stage1/active_attempt.json").read_text())
    stage1_path = Path(str(stage1_active["record"]))
    if not stage1_path.is_absolute():
        stage1_path = REPO / stage1_path
    stage1 = json.loads(stage1_path.read_text())
    seed_suffixes = (
        "observation_seed/body_pose.csv",
        "observation_seed/axial_phase.csv",
        "observation_seed/observation_seed_report.json",
    )
    seed_store = {
        suffix: next(
            value["sha256"]
            for path, value in stage1["stored_artifacts"].items()
            if path.endswith(suffix)
        )
        for suffix in seed_suffixes
    }
    return {
        "result_name": SWITCHED_RESULT_NAME,
        "output_sha256": {relative: _sha256(result_dir / relative) for relative in SWITCHED_OUTPUTS},
        "render_sha256": {relative: _sha256(render_dir / relative) for relative in SWITCHED_RENDERS},
        "loss_sha256": {relative: _sha256(result_dir / relative) for relative in SWITCHED_LOSS_ARTIFACTS},
        "stage_attempts": {
            stage: _latest_stage_attempt(result_dir, stage)
            for stage in ("stage1", "stage2", "stage3", "stage4", "stage5", "stage6", "stage7")
        },
        "stage1_seed_artifact_store_sha256": seed_store,
        "stage6_semantic_gap": {
            "required_csvs_pass": stage6["required_csvs_pass"],
            "required_renders_pass": stage6["required_renders_pass"],
            "frame_count": stage6["frame_count"],
            "pose_delta": stage6["pose_delta"],
            "pose_delta_pass": stage6["pose_delta_pass"],
            "phase_delta": stage6["phase_delta"],
            "phase_delta_pass": stage6["phase_delta_pass"],
            "overall_pass": stage6["overall_pass"],
        },
    }


def build_projected_periodic_regression_summary(attempt_path: Path) -> dict[str, object]:
    attempt_path = attempt_path.resolve()
    attempt = json.loads(attempt_path.read_text())
    candidate_dir = attempt_path.parent
    result_dir = candidate_dir.parent
    body_path = candidate_dir / str(attempt["body_candidate_artifact"])
    phase_path = candidate_dir / str(attempt["phase_candidate_artifact"])
    reference_body = result_dir / "observation_seed/body_pose.csv"
    reference_phase = result_dir / "observation_seed/axial_phase.csv"
    return {
        "schema_version": 1,
        "attempt_id": attempt["attempt_id"],
        "solver_executed": attempt["solver_executed"],
        "accepted_outputs_written": attempt["accepted_outputs_written"],
        "baseline_pose_read": attempt["baseline_pose_read"],
        "historical_phase_read": attempt["historical_phase_read"],
        "state_spec": attempt["state_spec"],
        "geometry_provider": attempt["geometry_provider"],
        "kinematic_contract": attempt["kinematic_contract"],
        "parameters": attempt["parameters"],
        "input_sha256": {name: item["sha256"] for name, item in attempt["inputs"].items()},
        "body_candidate_sha256": _sha256(body_path),
        "phase_candidate_sha256": _sha256(phase_path),
        "body_reference_sha256": _sha256(reference_body),
        "phase_reference_sha256": _sha256(reference_phase),
        "body_equals_observation_seed": body_path.read_bytes() == reference_body.read_bytes(),
        "phase_equals_observation_seed": phase_path.read_bytes() == reference_phase.read_bytes(),
        "frames": attempt["frames"],
        "body_fit_success_frames": attempt["body_fit_success_frames"],
        "body_residual_rms_median": attempt["body_residual_rms_median"],
        "body_residual_rms_p90": attempt["body_residual_rms_p90"],
        "phase": attempt["phase"],
        "switched_pipeline": _switched_pipeline_summary(result_dir.parent),
    }


def verify_projected_periodic_regression(
    attempt_path: Path,
    expected_path: Path = DEFAULT_PROJECTED_PERIODIC_GOLDEN,
) -> list[str]:
    try:
        actual = build_projected_periodic_regression_summary(attempt_path)
        expected = json.loads(expected_path.read_text())
    except (FileNotFoundError, KeyError, ValueError) as exc:
        return [f"projected periodic regression artifacts are incomplete: {exc}"]
    if actual == expected:
        return []
    return [
        f"{key}: expected {expected.get(key)!r}, got {actual.get(key)!r}"
        for key in sorted(set(expected) | set(actual))
        if expected.get(key) != actual.get(key)
    ]
