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
        if candidate["exists"] or selection in {
            "generated_by_stage1",
            "generated_by_stage3",
            "generated_if_observations_exist",
            "unconditional",
        }:
            selected = candidate
            break
    if selected is None:
        resolution = "unresolved"
        readiness = "blocked_missing_input"
    elif selected["exists"]:
        resolution = "selected_existing_file"
        readiness = "ready"
    elif selected["selection"] in {"generated_by_stage1", "generated_by_stage3"}:
        inputs_exist = bool(selected["dependencies"]) and all(item["exists"] for item in selected["dependencies"])
        resolution = "generated_stage_output" if inputs_exist else "missing_generation_inputs"
        readiness = "ready" if inputs_exist else "blocked_missing_input"
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
    observation_csv = profile.sample_dir / "results/object_observations/object_observations.csv"
    proxy_csv = profile.sample_dir / "results/object_proxy_observations/object_proxy_observations.csv"
    body_pose = profile.result_dir / "observation_seed/body_pose.csv"
    phase = profile.result_dir / "observation_seed/axial_phase.csv"
    generation_inputs = [
        observation_csv,
        proxy_csv,
        profile.sample_dir / "results/gvhmr/result.pkl",
        profile.sample_dir / "articraft/materialized_mug_mesh/assets/meshes/body_shell.obj",
        profile.sample_dir / "articraft/materialized_mug_mesh/assets/meshes/handle_loop.obj",
    ]
    return [
        _dependency(
            "mug_handle_phase_source",
            consumer_stage="stage1,stage3,stage4",
            consumer="rigid_body_parts,rigid6_plus_phase,stable_grasp_anchor",
            candidates=[
                _candidate(
                    "observation_derived_axial_phase",
                    phase,
                    "observation_derived_stage_output",
                    selection="generated_by_stage1",
                    dependencies=generation_inputs,
                )
            ],
        ),
        _dependency(
            "mug_contact_export_pose",
            consumer_stage="stage1",
            consumer="rigid_body_parts/export_mug_articraft_contact_points",
            candidates=[
                _candidate(
                    "observation_derived_body_pose",
                    body_pose,
                    "observation_derived_stage_output",
                    selection="generated_by_stage1",
                    dependencies=generation_inputs,
                )
            ],
        ),
        _dependency(
            "mug_body_pose_seed",
            consumer_stage="stage3",
            consumer="rigid6_plus_phase/mug_opening_2d_pose_correction",
            candidates=[
                _candidate(
                    "observation_derived_body_pose",
                    body_pose,
                    "observation_derived_stage_output",
                    selection="generated_by_stage1",
                    dependencies=generation_inputs,
                )
            ],
        ),
    ]


def _chair_dependencies(profile: CaseProfile) -> list[dict[str, object]]:
    candidates = [
        _candidate(
            "current_stage3_observation_fit",
            profile.result_dir / "object_pose_init.csv",
            "observation_derived_stage_output",
            selection="generated_by_stage3",
            dependencies=[
                profile.sample_dir / "results/tracking/object_mesh_tracks.csv",
                profile.sample_dir / "results/da3/scene_depth/00001.npy",
                profile.sample_dir / "results/gvhmr/result.pkl",
                repo_path(profile.data["articraft_model_py"]),
            ],
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
