from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from ..base.config import CaseProfile
from ..base.io import repo_relative_value, write_json
from ..base.runtime import runtime_python
from .diagnostics import build_sequence_solver_shadow_diagnostics
from .line_diagnostics import LINE_CONTACT_SANDBOX_ARTIFACTS
from .sphere_sequence import SPHERE_ATTEMPT_NAME, SPHERE_CANDIDATE_NAME, SPHERE_RESIDUAL_NAME


ACCEPTED_OUTPUT_NAMES = {
    "object_pose_init.csv",
    "object_pose.csv",
    "object_phase.csv",
    "handle_phase.csv",
    "object_contact_points.csv",
}
SANDBOX_MANIFEST_NAME = "generic_sequence_solver_shadow_candidate.json"
SPHERE_SANDBOX_ARTIFACTS = [
    SANDBOX_MANIFEST_NAME,
    SPHERE_CANDIDATE_NAME,
    SPHERE_RESIDUAL_NAME,
    SPHERE_ATTEMPT_NAME,
]
CHAIR_SANDBOX_ARTIFACTS = [
    SANDBOX_MANIFEST_NAME,
    "generic_chair_factor_candidate.csv",
    "generic_chair_factor_residuals.csv",
    "chair_generic_factor_executor_attempt.json",
    "chair_generic_factor_residuals.json",
    "generic_problem_preparation.json",
]
MUG_PERIODIC_SANDBOX_ARTIFACTS = [
    SANDBOX_MANIFEST_NAME,
    "generic_periodic_body_candidate.csv",
    "generic_periodic_phase_candidate.csv",
    "generic_projected_periodic_attempt.json",
]
GENERIC_OBJECT_SANDBOX_ARTIFACTS = [
    SANDBOX_MANIFEST_NAME,
    "generic_object_pose_candidate.csv",
    "generic_object_publication.json",
    "generic_problem_preparation.json",
    "generic_sequence_solver_attempts",
]


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def default_candidate_dir(result_dir: Path, case_name: str) -> Path:
    return result_dir.parents[0] / "generic_sequence_solver_shadow" / f"{result_dir.name}_{case_name}"


def _sphere_contact_events_path(result_dir: Path) -> Path:
    contact_events = result_dir / "contact_events.csv"
    if contact_events.exists():
        return contact_events
    return result_dir / "contact_candidates_internal/contact_candidates_labeled.csv"


def _run_isolated_sphere_sequence_executor(profile: CaseProfile, result_dir: Path, candidate_dir: Path) -> None:
    if profile.component("pose_model") != "translation3" or profile.component("geometry_model") != "sphere_proxy":
        raise ValueError("isolated sphere sequence executor requires translation3 + sphere_proxy")
    inputs = {
        "contact events": _sphere_contact_events_path(result_dir),
        "human sites": result_dir / "human_sites.csv",
        "support geometry": result_dir / "support_geometry.json",
    }
    for name, path in inputs.items():
        if not path.exists():
            raise FileNotFoundError(f"missing {name}: {path}")
    cmd = [
        runtime_python("audiohoi", override_env="AUDIOHOI_PYTHON"),
        str(Path(__file__).resolve().parents[2] / "tools/solve_sphere_sequence_candidate.py"),
        "--case",
        profile.case_name,
        "--result-dir",
        str(result_dir),
        "--contact-events-csv",
        str(inputs["contact events"]),
        "--human-sites-csv",
        str(inputs["human sites"]),
        "--support-geometry-json",
        str(inputs["support geometry"]),
        "--candidate-dir",
        str(candidate_dir),
    ]
    subprocess.run(cmd, cwd=Path(__file__).resolve().parents[5], check=True, text=True, capture_output=True)


def _run_generic_object_executor(profile: CaseProfile, result_dir: Path, candidate_dir: Path) -> None:
    """Run the one capability-driven object solver and hard-gated publisher."""
    cmd = [
        runtime_python("audiohoi", override_env="AUDIOHOI_PYTHON"),
        str(Path(__file__).resolve().parents[2] / "tools/prepare_generic_object_problem.py"),
        "--case",
        profile.case_name,
        "--result-name",
        result_dir.name,
        "--output",
        str(candidate_dir / "generic_problem_preparation.json"),
        "--solve",
        "--candidate-dir",
        str(candidate_dir),
        "--max-nfev",
        "2",
    ]
    subprocess.run(cmd, cwd=Path(__file__).resolve().parents[5], check=True, text=True, capture_output=True)


def build_candidate_sandbox_manifest(profile: CaseProfile, result_dir: Path, candidate_dir: Path | None = None) -> dict[str, object]:
    diagnostics = build_sequence_solver_shadow_diagnostics(profile, result_dir)
    target_dir = candidate_dir or default_candidate_dir(result_dir, profile.case_name)
    eligible = diagnostics["status"] == "ready_for_future_shadow_solve"
    status = "sandbox_ready" if eligible else "blocked_by_known_gaps"
    is_sphere = profile.component("pose_model") == "translation3" and profile.component("geometry_model") == "sphere_proxy"
    uses_generic_object_executor = isinstance(profile.data.get("generic_object_problem"), dict)
    if eligible and uses_generic_object_executor:
        planned_artifacts = GENERIC_OBJECT_SANDBOX_ARTIFACTS
    elif eligible and is_sphere:
        planned_artifacts = SPHERE_SANDBOX_ARTIFACTS
    elif eligible:
        planned_artifacts = [SANDBOX_MANIFEST_NAME]
    else:
        planned_artifacts = []
    canonical_payload = {
        "attempt_id": diagnostics["attempt_id"],
        "status": status,
        "geometry_kind": "sphere" if is_sphere else "other",
        "generic_object_executor": uses_generic_object_executor,
        "candidate_dir": str(repo_relative_value(target_dir)),
        "planned_artifacts": planned_artifacts,
        "blocking_gap_ids": diagnostics["blocking_gap_ids"],
        "nonblocking_gap_ids": diagnostics["nonblocking_gap_ids"],
    }
    return {
        "schema_version": 1,
        "mode": "generic_sequence_solver_candidate_sandbox",
        "sample_id": profile.case_name,
        "geometry_kind": "sphere" if is_sphere else "other",
        "generic_object_executor": uses_generic_object_executor,
        "status": status,
        "eligible_for_candidate_sandbox": eligible,
        "solver_executed": False,
        "accepted_outputs_written": False,
        "baseline_pose_read": False,
        "canonical_result_dir": str(repo_relative_value(result_dir)),
        "candidate_dir": str(repo_relative_value(target_dir)),
        "attempt_id": diagnostics["attempt_id"],
        "problem_sha256": diagnostics["problem_sha256"],
        "diagnostics_sha256": diagnostics["canonical_sha256"],
        "blocking_gap_ids": diagnostics["blocking_gap_ids"],
        "nonblocking_gap_ids": diagnostics["nonblocking_gap_ids"],
        "planned_artifacts": planned_artifacts,
        "forbidden_artifact_names": sorted(ACCEPTED_OUTPUT_NAMES),
        "write_policy": (
            "sphere_solver_writes_only_safe_candidate_attempt_and_residual_artifacts"
            if is_sphere
            else "generic_executor_writes_candidate_and_attempt_then_hard_gate_controls_single_publisher"
        ),
        "canonical_sha256": _canonical_hash(canonical_payload),
    }


def validate_candidate_sandbox_manifest(manifest: dict[str, object]) -> list[str]:
    errors: list[str] = []
    if manifest.get("mode") != "generic_sequence_solver_candidate_sandbox":
        errors.append("candidate sandbox mode must be generic_sequence_solver_candidate_sandbox")
    if manifest.get("solver_executed") is not False:
        errors.append("candidate sandbox must not execute solver in this branch")
    if manifest.get("accepted_outputs_written") is not False:
        errors.append("candidate sandbox must not write accepted outputs")
    if manifest.get("baseline_pose_read") is not False:
        errors.append("candidate sandbox must not read baseline pose")

    eligible = manifest.get("eligible_for_candidate_sandbox") is True
    status = str(manifest.get("status", ""))
    blocking_gap_ids = manifest.get("blocking_gap_ids", [])
    nonblocking_gap_ids = manifest.get("nonblocking_gap_ids", [])
    if not isinstance(blocking_gap_ids, list):
        errors.append("blocking_gap_ids must be a list")
        blocking_gap_ids = []
    if not isinstance(nonblocking_gap_ids, list):
        errors.append("nonblocking_gap_ids must be a list")
    if eligible:
        if status != "sandbox_ready":
            errors.append("eligible sandbox must have status=sandbox_ready")
        if blocking_gap_ids:
            errors.append("eligible sandbox must not carry blocking gaps")
    else:
        if status != "blocked_by_known_gaps":
            errors.append("blocked sandbox must have status=blocked_by_known_gaps")
        if not blocking_gap_ids:
            errors.append("blocked sandbox must record at least one blocking gap")

    candidate_dir = Path(str(manifest.get("candidate_dir", "")))
    canonical_result_dir = Path(str(manifest.get("canonical_result_dir", "")))
    if not str(candidate_dir):
        errors.append("candidate_dir must be recorded")
    if not str(canonical_result_dir):
        errors.append("canonical_result_dir must be recorded")
    if candidate_dir == canonical_result_dir:
        errors.append("candidate_dir must not equal canonical_result_dir")

    planned = manifest.get("planned_artifacts", [])
    if not isinstance(planned, list):
        errors.append("planned_artifacts must be a list")
    else:
        forbidden = set(ACCEPTED_OUTPUT_NAMES)
        overlap = sorted(str(item) for item in planned if Path(str(item)).name in forbidden)
        if overlap:
            errors.append(f"planned artifacts include accepted output names: {overlap}")
        if manifest.get("generic_object_executor") is True:
            expected = GENERIC_OBJECT_SANDBOX_ARTIFACTS
        elif manifest.get("geometry_kind") == "sphere":
            expected = SPHERE_SANDBOX_ARTIFACTS
        else:
            expected = [SANDBOX_MANIFEST_NAME]
        if eligible and planned != expected:
            errors.append(f"eligible sandbox must plan exactly the safe artifacts: {expected}")
        if not eligible and planned:
            errors.append("blocked sandbox must not plan artifacts")
    return errors


def write_candidate_sandbox_manifest(profile: CaseProfile, result_dir: Path, candidate_dir: Path | None = None) -> dict[str, object]:
    manifest = build_candidate_sandbox_manifest(profile, result_dir, candidate_dir)
    errors = validate_candidate_sandbox_manifest(manifest)
    if errors:
        raise ValueError("; ".join(errors))
    if manifest["eligible_for_candidate_sandbox"] is True:
        target_dir = Path(str(manifest["candidate_dir"]))
        if isinstance(profile.data.get("generic_object_problem"), dict):
            _run_generic_object_executor(profile, result_dir, target_dir)
        elif profile.component("pose_model") == "translation3" and profile.component("geometry_model") == "sphere_proxy":
            _run_isolated_sphere_sequence_executor(profile, result_dir, target_dir)
        write_json(target_dir / SANDBOX_MANIFEST_NAME, manifest)
    return manifest


def verify_materialized_generic_object_candidate(candidate_dir: Path) -> list[str]:
    errors: list[str] = []
    required = (
        "generic_object_pose_candidate.csv",
        "generic_object_publication.json",
        "generic_problem_preparation.json",
        "generic_sequence_solver_attempts",
    )
    for name in required:
        if not (candidate_dir / name).exists():
            errors.append(f"missing generic object candidate artifact: {name}")
    if errors:
        return errors
    publication = json.loads((candidate_dir / "generic_object_publication.json").read_text())
    if publication.get("status") not in {"candidate_blocked", "accepted"}:
        errors.append("generic object publication has invalid status")
    if publication.get("case_dispatch_used") is not False:
        errors.append("generic object publication used case dispatch")
    if publication.get("human_state_optimized") is not False:
        errors.append("generic object publication optimized human state")
    attempts = [path for path in (candidate_dir / "generic_sequence_solver_attempts").iterdir() if path.is_dir()]
    if len(attempts) != 1 or not (attempts[0] / "status.json").is_file():
        errors.append("generic object candidate must contain one isolated solve attempt")
    if (candidate_dir / "object_pose.csv").exists():
        errors.append("candidate directory must not contain canonical object_pose.csv")
    return errors
