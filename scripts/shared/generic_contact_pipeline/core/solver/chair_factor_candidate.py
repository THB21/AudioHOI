from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from ..base.config import CaseProfile
from ..base.io import REPO, copy_file, read_csv, repo_relative_value, write_csv, write_json
from ..base.runtime import runtime_python
from ..factors import build_chair_factor_executor_bundle, build_factor_shadow, validate_chair_factor_executor_bundle
from .candidate import ACCEPTED_OUTPUT_NAMES
from .chair_diagnostics import build_chair_contact_diagnostics, validate_chair_contact_diagnostics


CHAIR_FACTOR_ATTEMPT_NAME = "chair_generic_factor_executor_attempt.json"
CHAIR_FACTOR_RESIDUALS_NAME = "chair_generic_factor_residuals.json"
CHAIR_FACTOR_CANDIDATE_NAME = "generic_chair_factor_candidate.csv"
CHAIR_FACTOR_RESIDUAL_TABLE_NAME = "generic_chair_factor_residuals.csv"
CHAIR_FACTOR_SAFE_OUTPUTS = [
    CHAIR_FACTOR_ATTEMPT_NAME,
    CHAIR_FACTOR_RESIDUALS_NAME,
    CHAIR_FACTOR_CANDIDATE_NAME,
    CHAIR_FACTOR_RESIDUAL_TABLE_NAME,
]
CHAIR_SUPPORTED_RESIDUAL_BLOCKS = (
    "point_reprojection",
    "contact_distance",
    "joint_limit",
    "gauge_constraint",
    "pose_prior",
    "temporal_velocity",
)


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pairprop_pose_path(result_dir: Path) -> Path:
    return result_dir / "stage4_pairprop_contact_refine/object_pose_pairprop_generic.csv"


def _pairprop_metrics_path(result_dir: Path) -> Path:
    return result_dir / "stage4_pairprop_contact_refine/pairprop_contact_refine_metrics.json"


def _candidate_executor_metrics_path(candidate_dir: Path) -> Path:
    return candidate_dir / ".generic_chair_factor_executor_metrics.json"


def _gated_contacts_path(result_dir: Path) -> Path:
    return result_dir / "stage4_generic_refine/object_contact_points_vlm_gated.csv"


def _chair_residual_rows(metrics: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    raw_rows = metrics.get("metrics", [])
    if not isinstance(raw_rows, list):
        return rows
    for item in raw_rows:
        if not isinstance(item, dict):
            continue
        init = item.get("contact_chord_initializer", {})
        gauge = item.get("contact_chord_2d_gauge", {})
        if not isinstance(init, dict):
            init = {}
        if not isinstance(gauge, dict):
            gauge = {}
        rows.append(
            {
                "frame": item.get("frame", ""),
                "cost": item.get("cost", ""),
                "median_contact_gap_m": item.get("median_contact_gap_m", ""),
                "contact_chord_initializer_used": init.get("used", ""),
                "contact_chord_correspondence_count": init.get("correspondence_count", ""),
                "local_chord_length_m": init.get("local_chord_length_m", ""),
                "palm_chord_length_m": init.get("palm_chord_length_m", ""),
                "theoretical_min_gap_m": init.get("theoretical_min_gap_m", ""),
                "seed_median_contact_gap_m": init.get("seed_median_contact_gap_m", ""),
                "rotation_from_stage3_rad": init.get("rotation_from_stage3_rad", ""),
                "contact_chord_2d_gauge_used": gauge.get("used", ""),
                "contact_chord_2d_gauge_success": gauge.get("success", ""),
                "twist_rad": gauge.get("twist_rad", ""),
                "initial_cost": gauge.get("initial_cost", ""),
                "gauge_cost": gauge.get("cost", ""),
                "cost_nonincreasing": gauge.get("cost_nonincreasing", ""),
                "rear_joint_angle": gauge.get("rear_joint_angle", ""),
                "seat_joint_angle": gauge.get("seat_joint_angle", ""),
                "contact_chord_constraint_preserved": item.get("contact_chord_constraint_preserved", ""),
            }
        )
    return rows


def build_chair_factor_residual_coverage(profile: CaseProfile, result_dir: Path) -> dict[str, object]:
    factor_shadow = build_factor_shadow(profile, result_dir)
    records = factor_shadow.get("factors", {}).get("records", [])
    by_kind: dict[str, list[str]] = {}
    if isinstance(records, list):
        for record in records:
            if not isinstance(record, dict):
                continue
            raw_kind = record.get("kind")
            kind = str(raw_kind.value if hasattr(raw_kind, "value") else raw_kind)
            by_kind.setdefault(kind, []).append(str(record.get("factor_id")))
    required = ("point_reprojection", "contact_distance", "joint_limit", "gauge_constraint")
    payload = {
        "schema_version": 1,
        "mode": "chair_generic_factor_residual_coverage",
        "sample_id": "chair",
        "residual_evaluator_executed": True,
        "solver_executed": False,
        "accepted_outputs_written": False,
        "baseline_pose_read": False,
        "factor_shadow_sha256": factor_shadow["canonical_sha256"],
        "required_factor_kinds": list(required),
        "required_factor_kinds_present": all(kind in by_kind for kind in required),
        "factor_ids_by_kind": {kind: sorted(values) for kind, values in sorted(by_kind.items())},
        "policy": "coverage-only residual evaluator manifest; does not optimize or write accepted outputs",
    }
    payload["canonical_sha256"] = _canonical_hash(
        {
            "mode": payload["mode"],
            "factor_shadow_sha256": payload["factor_shadow_sha256"],
            "required_factor_kinds": payload["required_factor_kinds"],
            "required_factor_kinds_present": payload["required_factor_kinds_present"],
            "factor_ids_by_kind": payload["factor_ids_by_kind"],
        }
    )
    return payload


def validate_chair_factor_residual_coverage(payload: dict[str, object]) -> list[str]:
    errors: list[str] = []
    if payload.get("mode") != "chair_generic_factor_residual_coverage":
        errors.append("chair factor residual coverage mode must be chair_generic_factor_residual_coverage")
    if payload.get("residual_evaluator_executed") is not True:
        errors.append("chair factor residual coverage must execute the residual evaluator")
    if payload.get("solver_executed") is not False:
        errors.append("chair factor residual coverage must not execute solver")
    if payload.get("accepted_outputs_written") is not False:
        errors.append("chair factor residual coverage must not write accepted outputs")
    if payload.get("baseline_pose_read") is not False:
        errors.append("chair factor residual coverage must not read baseline pose")
    if payload.get("required_factor_kinds_present") is not True:
        errors.append("chair factor residual coverage must have all required factor kinds present")
    by_kind = payload.get("factor_ids_by_kind", {})
    if not isinstance(by_kind, dict):
        errors.append("chair factor residual coverage must record factor ids by kind")
        by_kind = {}
    for kind in ("point_reprojection", "contact_distance", "joint_limit", "gauge_constraint"):
        if kind not in by_kind:
            errors.append(f"chair factor residual coverage missing required factor kind {kind}")
    return errors


def _run_isolated_chair_factor_executor(profile: CaseProfile, result_dir: Path, candidate_dir: Path) -> tuple[Path, Path]:
    if profile.case_name != "chair":
        raise ValueError("isolated chair factor executor only supports the chair case")
    candidate_dir.mkdir(parents=True, exist_ok=True)
    pose_path = candidate_dir / CHAIR_FACTOR_CANDIDATE_NAME
    metrics_path = _candidate_executor_metrics_path(candidate_dir)
    script = REPO / "scripts/shared/generic_contact_pipeline/components/refinement/solvers/chair_twohand_endpoint_se3.py"
    seed_csv = result_dir / "object_pose_init.csv"
    contacts_csv = _gated_contacts_path(result_dir)
    segments_csv = result_dir / "object_local_segments.csv"
    for name, path in {
        "current-run chair seed": seed_csv,
        "gated contacts": contacts_csv,
        "object local segments": segments_csv,
    }.items():
        if not path.exists():
            raise FileNotFoundError(f"missing {name}: {path}")
    cmd = [
        runtime_python("audiohoi", override_env="AUDIOHOI_PYTHON"),
        str(script),
        "--ref-pose-csv",
        str(seed_csv),
        "--init-pose-csv",
        str(seed_csv),
        "--contacts-csv",
        str(contacts_csv),
        "--segments-csv",
        str(segments_csv),
        "--out-csv",
        str(pose_path),
        "--metrics-json",
        str(metrics_path),
        "--rot-bound",
        "0.55",
        "--xy-bound",
        "0.45",
        "--z-bound",
        "1.25",
        "--w-contact",
        "2.0",
        "--w-2d",
        "0.35",
        "--w-prior-rot",
        "0.06",
        "--w-prior-xy",
        "0.05",
        "--w-prior-z",
        "0.03",
        "--w-temporal",
        "0.35",
        "--sigma-contact-m",
        "0.035",
        "--sigma-px",
        "7.5",
        "--max-nfev",
        "240",
        "--contact-chord-init",
        "--contact-chord-2d-gauge",
        "--optimize-articulation",
        "--preserve-contact-chord-constraint",
    ]
    subprocess.run(cmd, cwd=REPO, check=True, text=True, capture_output=True)
    return pose_path, metrics_path


def build_chair_factor_executor_candidate(
    profile: CaseProfile,
    result_dir: Path,
    candidate_dir: Path,
    *,
    solver_executed: bool = False,
    pose_path: Path | None = None,
    metrics_path: Path | None = None,
) -> dict[str, object]:
    if profile.case_name != "chair":
        raise ValueError("chair factor executor candidate only supports the chair case")
    bundle = build_chair_factor_executor_bundle(profile, result_dir)
    diagnostics = build_chair_contact_diagnostics(result_dir)
    bundle_errors = validate_chair_factor_executor_bundle(bundle)
    diagnostics_errors = validate_chair_contact_diagnostics(diagnostics)
    status = str(bundle["status"])
    pose_path = pose_path or _pairprop_pose_path(result_dir)
    metrics_path = metrics_path or _pairprop_metrics_path(result_dir)
    pose_rows = read_csv(pose_path) if pose_path.exists() else []
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    residual_rows = _chair_residual_rows(metrics)
    materialized = status == "ready_for_candidate_executor" and bool(pose_rows) and bool(residual_rows)
    candidate_source = "isolated_chair_factor_executor" if solver_executed else str(repo_relative_value(pose_path))
    residual_source = "isolated_chair_factor_executor" if solver_executed else str(repo_relative_value(metrics_path))
    core = {
        "bundle_sha256": bundle["canonical_sha256"],
        "contact_diagnostics_sha256": diagnostics["canonical_sha256"],
        "status": status,
        "isolated_candidate_materialized": materialized,
        "missing_required_factor_kinds": bundle["missing_required_factor_kinds"],
        "compatibility_gap_id": bundle["compatibility_gap_id"],
        "compatibility_gap_status": bundle["compatibility_gap_status"],
        "supported_residual_blocks": list(CHAIR_SUPPORTED_RESIDUAL_BLOCKS),
        "validation_error_count": len(bundle_errors) + len(diagnostics_errors),
        "candidate_dir": str(repo_relative_value(candidate_dir)),
        "candidate_pose_sha256": _sha256(pose_path) if pose_path.exists() else None,
        "residual_metrics_sha256": _sha256(metrics_path) if metrics_path.exists() else None,
        "candidate_pose_rows": len(pose_rows),
        "residual_rows": len(residual_rows),
        "solver_executed": solver_executed,
        "executor_scope": "isolated_candidate_dir" if solver_executed else "source_artifact_materialization",
    }
    canonical = _canonical_hash(core)
    return {
        "schema_version": 1,
        "mode": "chair_generic_factor_executor_candidate_attempt",
        "attempt_id": f"chair-factor-{canonical[:12]}",
        "sample_id": "chair",
        "canonical_result_dir": str(repo_relative_value(result_dir)),
        "candidate_dir": str(repo_relative_value(candidate_dir)),
        "status": status,
        "solver_executed": solver_executed,
        "executor_scope": "isolated_candidate_dir" if solver_executed else "source_artifact_materialization",
        "accepted_outputs_written": False,
        "baseline_pose_read": False,
        "isolated_candidate_materialized": materialized,
        "missing_required_factor_kinds": bundle["missing_required_factor_kinds"],
        "supported_residual_blocks": list(CHAIR_SUPPORTED_RESIDUAL_BLOCKS),
        "candidate_pose": {
            "planned_output": CHAIR_FACTOR_CANDIDATE_NAME,
            "source": candidate_source,
            "source_sha256": _sha256(pose_path) if pose_path.exists() else None,
            "rows": len(pose_rows),
        },
        "residual_table": {
            "planned_output": CHAIR_FACTOR_RESIDUAL_TABLE_NAME,
            "source": residual_source,
            "source_sha256": _sha256(metrics_path) if metrics_path.exists() else None,
            "rows": len(residual_rows),
        },
        "bundle": {
            "canonical_sha256": bundle["canonical_sha256"],
            "status": bundle["status"],
            "available_factor_kinds": bundle["available_factor_kinds"],
            "missing_required_factor_kinds": bundle["missing_required_factor_kinds"],
        },
        "contact_diagnostics": {
            "canonical_sha256": diagnostics["canonical_sha256"],
            "seed_policy": diagnostics["summary"]["seed_policy"],
            "compatibility_gap_status": diagnostics["compatibility_gap_status"],
        },
        "blocking_reasons": list(bundle.get("blocking_reasons", [])),
        "validation_errors": bundle_errors + diagnostics_errors,
        "planned_outputs": list(CHAIR_FACTOR_SAFE_OUTPUTS),
        "forbidden_artifact_names": sorted(ACCEPTED_OUTPUT_NAMES),
        "canonical_sha256": canonical,
    }


def validate_chair_factor_executor_candidate(attempt: dict[str, object]) -> list[str]:
    errors: list[str] = []
    if attempt.get("mode") != "chair_generic_factor_executor_candidate_attempt":
        errors.append("chair factor candidate mode must be chair_generic_factor_executor_candidate_attempt")
    if attempt.get("accepted_outputs_written") is not False:
        errors.append("chair factor candidate must not write accepted outputs")
    if attempt.get("baseline_pose_read") is not False:
        errors.append("chair factor candidate must not read baseline pose")
    if attempt.get("solver_executed") is True and attempt.get("executor_scope") != "isolated_candidate_dir":
        errors.append("executed chair factor candidate must record isolated_candidate_dir scope")
    planned = attempt.get("planned_outputs", [])
    if not isinstance(planned, list) or planned != CHAIR_FACTOR_SAFE_OUTPUTS:
        errors.append("chair factor candidate must only plan safe attempt/residual manifests")
    forbidden = set(ACCEPTED_OUTPUT_NAMES)
    overlap = sorted(str(item) for item in planned if Path(str(item)).name in forbidden) if isinstance(planned, list) else []
    if overlap:
        errors.append(f"chair factor candidate planned accepted output names: {overlap}")
    if attempt.get("status") != "ready_for_candidate_executor" and attempt.get("solver_executed") is not False:
        errors.append("blocked chair factor candidate must not execute solver")
    if attempt.get("candidate_dir") == attempt.get("canonical_result_dir"):
        errors.append("chair factor candidate dir must not equal canonical result dir")
    if attempt.get("status") == "ready_for_candidate_executor":
        if attempt.get("isolated_candidate_materialized") is not True:
            errors.append("ready chair factor candidate must materialize isolated candidate artifacts")
        candidate_pose = attempt.get("candidate_pose", {})
        residual_table = attempt.get("residual_table", {})
        if not isinstance(candidate_pose, dict) or int(candidate_pose.get("rows", 0) or 0) <= 0:
            errors.append("ready chair factor candidate must record candidate pose rows")
        if not isinstance(residual_table, dict) or int(residual_table.get("rows", 0) or 0) <= 0:
            errors.append("ready chair factor candidate must record residual table rows")
    return errors


def _load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    return payload if isinstance(payload, dict) else {}


def verify_materialized_chair_factor_candidate(profile: CaseProfile, result_dir: Path, candidate_dir: Path) -> list[str]:
    """Verify an isolated chair candidate directory matches its safe manifest.

    This verifier intentionally validates only sandbox artifacts. It rejects
    accepted output names so a reviewable candidate cannot silently become a
    canonical Stage output.
    """
    errors: list[str] = []
    if profile.case_name != "chair":
        return ["chair factor candidate verifier only supports the chair case"]

    for name in sorted(ACCEPTED_OUTPUT_NAMES):
        if (candidate_dir / name).exists():
            errors.append(f"accepted output artifact present in chair candidate dir: {name}")

    for name in CHAIR_FACTOR_SAFE_OUTPUTS:
        if not (candidate_dir / name).exists():
            errors.append(f"missing planned output {name}")

    attempt_path = candidate_dir / CHAIR_FACTOR_ATTEMPT_NAME
    coverage_path = candidate_dir / CHAIR_FACTOR_RESIDUALS_NAME
    candidate_path = candidate_dir / CHAIR_FACTOR_CANDIDATE_NAME
    residual_table_path = candidate_dir / CHAIR_FACTOR_RESIDUAL_TABLE_NAME

    attempt = _load_json(attempt_path)
    if not attempt:
        return errors + [f"missing or invalid attempt manifest {CHAIR_FACTOR_ATTEMPT_NAME}"]
    errors.extend(validate_chair_factor_executor_candidate(attempt))
    if attempt.get("canonical_result_dir") != str(repo_relative_value(result_dir)):
        errors.append("chair factor candidate canonical_result_dir does not match requested result_dir")
    planned = attempt.get("planned_outputs", [])
    if planned != CHAIR_FACTOR_SAFE_OUTPUTS:
        errors.append("chair factor candidate planned_outputs do not match safe output contract")

    coverage = _load_json(coverage_path)
    if not coverage:
        errors.append(f"missing or invalid residual coverage manifest {CHAIR_FACTOR_RESIDUALS_NAME}")
    else:
        errors.extend(validate_chair_factor_residual_coverage(coverage))

    candidate_pose = attempt.get("candidate_pose", {})
    if isinstance(candidate_pose, dict) and candidate_path.exists():
        rows = read_csv(candidate_path)
        if int(candidate_pose.get("rows", 0) or 0) != len(rows):
            errors.append("chair factor candidate pose row count does not match attempt manifest")
        source_sha = candidate_pose.get("source_sha256")
        if source_sha and _sha256(candidate_path) != source_sha:
            errors.append("chair factor candidate pose sha256 does not match source artifact")

    residual_table = attempt.get("residual_table", {})
    if isinstance(residual_table, dict) and residual_table_path.exists():
        rows = read_csv(residual_table_path)
        if int(residual_table.get("rows", 0) or 0) != len(rows):
            errors.append("chair factor residual table row count does not match attempt manifest")

    return errors


def prepare_chair_factor_executor_candidate(
    profile: CaseProfile,
    result_dir: Path,
    candidate_dir: Path,
    *,
    execute_solver: bool = False,
) -> dict[str, object]:
    executed_pose_path: Path | None = None
    executed_metrics_path: Path | None = None
    if execute_solver:
        executed_pose_path, executed_metrics_path = _run_isolated_chair_factor_executor(profile, result_dir, candidate_dir)
    attempt = build_chair_factor_executor_candidate(
        profile,
        result_dir,
        candidate_dir,
        solver_executed=execute_solver,
        pose_path=executed_pose_path,
        metrics_path=executed_metrics_path,
    )
    errors = validate_chair_factor_executor_candidate(attempt)
    if errors:
        raise ValueError("; ".join(errors))
    residuals = build_chair_factor_residual_coverage(profile, result_dir)
    residual_errors = validate_chair_factor_residual_coverage(residuals)
    if residual_errors:
        raise ValueError("; ".join(residual_errors))
    candidate_dir.mkdir(parents=True, exist_ok=True)
    if attempt.get("status") == "ready_for_candidate_executor":
        if not execute_solver:
            copy_file(_pairprop_pose_path(result_dir), candidate_dir / CHAIR_FACTOR_CANDIDATE_NAME)
        metrics_source = executed_metrics_path or _pairprop_metrics_path(result_dir)
        metrics = json.loads(metrics_source.read_text())
        write_csv(candidate_dir / CHAIR_FACTOR_RESIDUAL_TABLE_NAME, _chair_residual_rows(metrics))
        if execute_solver and executed_metrics_path is not None and executed_metrics_path.exists():
            executed_metrics_path.unlink()
    write_json(candidate_dir / CHAIR_FACTOR_RESIDUALS_NAME, residuals)
    write_json(candidate_dir / CHAIR_FACTOR_ATTEMPT_NAME, attempt)
    return attempt
