from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from ..base.config import CaseProfile
from ..base.io import REPO, repo_path, repo_relative_value


SOLVED_SOURCE_CLASSES = {
    "cached_derivative_of_historical_solved_phase",
    "cached_derivative_of_historical_solved_pose",
    "historical_solved_output",
    "historical_solved_seed",
    "preserved_solved_snapshot",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_record(path: Path) -> dict[str, object]:
    exists = path.is_file()
    record: dict[str, object] = {
        "path": str(repo_relative_value(path)),
        "exists": exists,
    }
    if exists:
        record["size_bytes"] = path.stat().st_size
        record["sha256"] = _sha256(path)
    return record


def _candidate(
    candidate_id: str,
    path: Path,
    source_class: str,
    *,
    selection: str = "if_file_exists",
    dependencies: list[Path] | None = None,
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "source_class": source_class,
        "selection": selection,
        **_path_record(path),
        "dependencies": [_path_record(item) for item in dependencies or []],
    }


def _dependency(
    dependency_id: str,
    *,
    consumer_stage: str,
    consumer: str,
    candidates: list[dict[str, object]],
) -> dict[str, object]:
    selected: dict[str, object] | None = None
    for candidate in candidates:
        selection = candidate["selection"]
        if candidate["exists"] or selection in {"generated_if_observations_exist", "unconditional"}:
            selected = candidate
            break
    if selected is None:
        resolution = "unresolved"
        readiness = "blocked_missing_input"
    elif selected["exists"]:
        resolution = "selected_existing_file"
        readiness = "ready"
    elif selected["selection"] == "generated_if_observations_exist":
        observation_exists = any(item["exists"] for item in selected["dependencies"])
        resolution = "generated_fallback" if observation_exists else "generated_empty_fallback"
        readiness = "degraded_output_change"
    else:
        resolution = "selected_missing_path"
        readiness = "blocked_missing_input"
    source_class = str(selected["source_class"]) if selected else "unresolved"
    return {
        "dependency_id": dependency_id,
        "consumer_stage": consumer_stage,
        "consumer": consumer,
        "selected_candidate_id": selected["candidate_id"] if selected else None,
        "selected_source_class": source_class,
        "selected_path": selected["path"] if selected else None,
        "selected_exists": bool(selected and selected["exists"]),
        "resolution": resolution,
        "rerun_readiness": readiness,
        "solved_seed_dependency": source_class in SOLVED_SOURCE_CLASSES,
        "candidates": candidates,
    }


def _baseline_path(profile: CaseProfile, key: str) -> Path:
    raw = profile.baseline.get(key, "")
    return repo_path(raw) if raw else REPO / "__missing_baseline__" / key


def _mug_dependencies(profile: CaseProfile) -> list[dict[str, object]]:
    rebuild_dir = profile.result_dir / "m17_rebuild_from_recovered_m15"
    observation_csv = profile.sample_dir / "results/object_observations/object_observations.csv"
    phase_candidates = [
        _candidate(
            "rebuilt_m17_phase",
            rebuild_dir / "corrected_handle_phase.csv",
            "cached_derivative_of_historical_solved_phase",
            dependencies=[_baseline_path(profile, "m15_recovered_phase_csv")],
        ),
        _candidate(
            "preserved_m17_snapshot",
            profile.result_dir / "provenance_snapshots/mug_m17_handle_phase.csv",
            "preserved_solved_snapshot",
        ),
        _candidate(
            "historical_m17_phase",
            _baseline_path(profile, "m17_phase_csv"),
            "historical_solved_seed",
        ),
        _candidate(
            "historical_final_phase",
            _baseline_path(profile, "final_phase_csv"),
            "historical_solved_output",
        ),
        _candidate(
            "identity_phase_fallback",
            rebuild_dir / "identity_handle_phase.csv",
            "synthetic_identity_fallback",
            selection="generated_if_observations_exist",
            dependencies=[observation_csv],
        ),
    ]
    pose_seed_raw = profile.baseline.get("m18_pose_csv") or profile.baseline.get("final_pose_csv", "")
    pose_seed = repo_path(pose_seed_raw) if pose_seed_raw else profile.sample_dir / "results/pipe/anchored_pose_obs.csv"
    return [
        _dependency(
            "mug_handle_phase_source",
            consumer_stage="stage1,stage3,stage4",
            consumer="rigid_body_parts,rigid6_plus_phase,stable_grasp_anchor",
            candidates=phase_candidates,
        ),
        _dependency(
            "mug_body_pose_seed",
            consumer_stage="stage3",
            consumer="rigid6_plus_phase/mug_opening_2d_pose_correction",
            candidates=[
                _candidate(
                    "historical_m18_pose",
                    pose_seed,
                    "historical_solved_seed",
                    selection="unconditional",
                )
            ],
        ),
    ]


def _chair_dependencies(profile: CaseProfile) -> list[dict[str, object]]:
    mainline_pose = _baseline_path(profile, "final_pose_csv")
    mainline_segments = profile.sample_dir / "results/mainline_0425/semantic_local_points/chair_semantic_local_segments.csv"
    mainline_observations = profile.sample_dir / "results/mainline_0425/inputs_2d/chair_semantic_observations.csv"
    target_segments = profile.result_dir / "object_local_segments.csv"
    candidates = [
        _candidate(
            "rebuilt_physical6d_seed",
            profile.result_dir / "physical6d_rebuild_from_mainline_saved2d/physical6d_pose.csv",
            "cached_derivative_of_historical_solved_pose",
            dependencies=[mainline_pose, mainline_segments, mainline_observations, target_segments],
        ),
        _candidate(
            "preserved_physical6d_snapshot",
            profile.result_dir / "provenance_snapshots/chair_physical6d_seed.csv",
            "preserved_solved_snapshot",
        ),
        _candidate(
            "historical_physical6d_seed",
            _baseline_path(profile, "physical6d_seed_csv"),
            "historical_solved_seed",
            selection="unconditional",
        ),
    ]
    return [
        _dependency(
            "chair_stage4_pairprop_seed",
            consumer_stage="stage4",
            consumer="small_se3/chair_twohand_endpoint_se3",
            candidates=candidates,
        )
    ]


def audit_seed_dependencies(profile: CaseProfile) -> dict[str, Any]:
    """Inspect solved-seed selection without running a solver or writing data."""

    if profile.case_name == "mug":
        dependencies = _mug_dependencies(profile)
    elif profile.case_name == "chair":
        dependencies = _chair_dependencies(profile)
    else:
        raise ValueError(f"Seed dependency audit is only defined for mug/chair, got {profile.case_name!r}")
    readiness = "ready"
    if any(item["rerun_readiness"] == "blocked_missing_input" for item in dependencies):
        readiness = "blocked_missing_input"
    elif any(item["rerun_readiness"] == "degraded_output_change" for item in dependencies):
        readiness = "degraded_output_change"
    return {
        "schema_version": 1,
        "case_name": profile.case_name,
        "result_name": profile.result_name,
        "result_dir": str(repo_relative_value(profile.result_dir)),
        "rerun_readiness": readiness,
        "has_solved_seed_dependency": any(item["solved_seed_dependency"] for item in dependencies),
        "dependencies": dependencies,
    }
