from __future__ import annotations

from pathlib import Path

from ..base.config import CaseProfile
from ..base.io import REPO, read_csv_header, repo_path, write_csv
from ..base.schema import stage_paths


BASELINE_READER_COMPONENTS = {}

VERIFIED_INTERMEDIATE_COMPONENTS = {
    "rigid_body_plus_parts": "derives mug body pose and axial phase from current 2D/depth observations, then exports Articraft contact points from those result-owned artifacts",
    "rigid6_plus_phase": "refines the current observation-derived body pose with generic CSV-only 2D/contact correction; no historical M18/M17 input is read",
    "stable_grasp_anchor": "uses the current Stage 3 pose and observation-derived axial phase, then rebuilds final pose/phase and grasp-anchor state in generic code",
    "small_se3": "rebuilds chair physical6d seed from retained mainline saved-2D inputs, then runs generic two-hand endpoint SE(3) contact refinement with semantic-2D/contact/freeze quality gate",
}

PROVENANCE_SNAPSHOT_COMPONENTS = {}

EXTERNAL_RUNNER_COMPONENTS = {
}

GENERIC_SOLVER_EXTERNAL_ENV_COMPONENTS = {
    "anchor_depth": "runs generic_contact_pipeline/components/refinement/anchor_depth_solver.py in the audiohoi env because solver deps are not in base Python",
    "proxy_sphere": "runs generic_contact_pipeline/components/render/proxy_sphere_scene.py in the audiohoi env because render deps are not in base Python",
    "articraft_mesh": "runs generic_contact_pipeline/components/render/articraft_mug_scene.py in the audiohoi env because render deps are not in base Python",
    "urdf_solid_mesh": "runs generic_contact_pipeline/components/render/urdf_solid_scene.py in the audiohoi env because render deps are not in base Python",
    "two_hand_toprail_endpoint": "runs generic_contact_pipeline/components/contact/chair_grasp_anchor_state.py in the audiohoi env, then builds two-hand endpoint contact candidates from generic object_observations",
    "semantic_graph_6d": "runs generic_contact_pipeline/components/render/fit_chair_metric_6d_pose.py in the audiohoi env on generic semantic observations/local points/contact state",
    "hand_floor": "runs generic_contact_pipeline/components/contact/solvers/ball_proxy_depth_builder.py and ball_contact_candidate_builder.py in the audiohoi env; builds hand/floor gates from object mesh tracks + GVHMR + DA3 + audio",
    "foot_floor": "runs generic_contact_pipeline/components/contact/solvers/ball_proxy_depth_builder.py and ball_contact_candidate_builder.py in the audiohoi env; builds foot/floor gates from object mesh tracks + GVHMR + DA3 + audio",
}


def component_status(profile: CaseProfile) -> dict[str, dict[str, str]]:
    comps = {
        "geometry_model": profile.component("geometry_model"),
        "observation_model": profile.component("observation_model"),
        "contact_policy": profile.component("contact_policy"),
        "pose_model": profile.component("pose_model"),
        "render_backend": profile.component("render_backend"),
    }
    for idx, name in enumerate(profile.refinement_policies()):
        comps[f"refinement_policy_{idx}"] = name
    out: dict[str, dict[str, str]] = {}
    for role, comp in comps.items():
        if comp in EXTERNAL_RUNNER_COMPONENTS:
            status = "external_runner_on_generic_inputs"
            note = EXTERNAL_RUNNER_COMPONENTS[comp]
        elif comp in GENERIC_SOLVER_EXTERNAL_ENV_COMPONENTS:
            status = "generic_solver_external_env"
            note = GENERIC_SOLVER_EXTERNAL_ENV_COMPONENTS[comp]
        elif comp in VERIFIED_INTERMEDIATE_COMPONENTS:
            status = "clean_from_verified_intermediate"
            note = VERIFIED_INTERMEDIATE_COMPONENTS[comp]
        elif comp in PROVENANCE_SNAPSHOT_COMPONENTS:
            status = "preserved_provenance_snapshot"
            note = PROVENANCE_SNAPSHOT_COMPONENTS[comp]
        elif comp in BASELINE_READER_COMPONENTS:
            status = "baseline_reader"
            note = BASELINE_READER_COMPONENTS[comp]
        else:
            status = "clean_from_raw_or_stage_inputs"
            note = "no solved artifact dependency declared"
        priority = {
            "external_runner_on_generic_inputs": "migrate_runtime_code",
            "generic_solver_external_env": "keep_env_boundary_or_package_deps",
            "baseline_reader": "replace_solved_artifact_input",
            "clean_from_verified_intermediate": "move_left_to_raw_preprocess_when_safe",
            "preserved_provenance_snapshot": "keep_snapshot_until_missing_upstream_is_restored",
            "clean_from_raw_or_stage_inputs": "keep",
        }.get(status, "review")
        out[role] = {"component": comp, "migration_status": status, "migration_priority": priority, "note": note}
    return out


def schema_status(profile: CaseProfile) -> dict[str, object]:
    paths = stage_paths(profile)
    csv_paths = {
        "object_observations": paths["object_observations"],
        "object_local_points": paths["object_local_points"],
        "object_local_segments": paths["object_local_segments"],
        "contact_candidates": paths["contact_candidates"],
        "object_pose_init": paths["object_pose_init"],
        "object_pose": paths["object_pose"],
        "object_contact_points": paths["object_contact_points"],
        "object_phase": paths["object_phase"],
    }
    out = {}
    for name, path in csv_paths.items():
        exists = path.exists()
        out[name] = {
            "path": str(path),
            "exists": exists,
            "columns": read_csv_header(path) if exists else [],
        }
    return out


def baseline_path_status(profile: CaseProfile) -> dict[str, object]:
    out = {}
    for key, path in profile.baseline.items():
        p = repo_path(path)
        out[key] = {"path": str(p), "exists": p.exists(), "is_dir": p.is_dir()}
    return out


def old_script_reference_scan() -> dict[str, object]:
    root = REPO / "scripts/shared/generic_contact_pipeline"
    patterns = [
        "scripts.shared." + "radius_free_proxy",
        "scripts/shared/" + "radius_free_proxy",
        "stage4_anchor_refinement/" + "run_anchor_refinement.py",
        "stage5_render/" + "render_pose6d_scene.py",
        "stage5_render/" + "render_mug_articraft_fast_scene.py",
        "stage5_render/" + "render_chair_articraft_solid_3d_scene.py",
    ]
    hits: list[dict[str, object]] = []
    for path in sorted([*root.rglob("*.py"), *root.rglob("*.yaml")]):
        if "__pycache__" in path.parts:
            continue
        try:
            text = path.read_text(errors="ignore")
        except Exception:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            for pattern in patterns:
                if pattern in line:
                    hits.append(
                        {
                            "path": str(path.relative_to(REPO)),
                            "line": lineno,
                            "pattern": pattern,
                            "text": line.strip()[:240],
                        }
                    )
    return {
        "root": str(root),
        "patterns": patterns,
        "hit_count": len(hits),
        "hits": hits[:50],
        "pass": len(hits) == 0,
    }


def build_audit(profile: CaseProfile) -> dict[str, object]:
    statuses = component_status(profile)
    counts: dict[str, int] = {}
    for item in statuses.values():
        status = item["migration_status"]
        counts[status] = counts.get(status, 0) + 1
    return {
        "case_name": profile.case_name,
        "result_name": profile.result_name,
        "component_status": statuses,
        "migration_status_counts": counts,
        "runtime_coupling_components": {
            role: item
            for role, item in statuses.items()
            if item["migration_status"] in {"external_runner_on_generic_inputs", "baseline_reader"}
        },
        "old_script_reference_scan": old_script_reference_scan(),
        "schema_status": schema_status(profile),
        "baseline_path_status": baseline_path_status(profile),
        "migration_note": (
            "v1 keeps solved baselines only for regression comparison and explicit provenance snapshots. "
            "baseline_reader components would copy solved artifacts, but current four case profiles should not use any. "
            "external_runner_on_generic_inputs components would run old stable scripts on generic stage files, but current four case profiles should not use any. "
            "generic_solver_external_env components are generic-owned code but intentionally run in a non-base conda env for dependency isolation. "
            "clean_from_verified_intermediate components rebuild outputs from trusted intermediate artifacts. "
            "preserved_provenance_snapshot components use explicit generic_pipeline_v1/provenance_snapshots files where the historical upstream chain is no longer fully present. "
            "clean_from_raw_or_stage_inputs components do not declare solved-artifact dependencies. "
            "Each step should be migrated leftward and validated with stage6_compare before replacing the old path."
        ),
    }


def write_audit_csv(profile: CaseProfile, audit: dict[str, object]) -> str:
    rows = []
    for role, item in dict(audit.get("component_status", {})).items():
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "case_name": profile.case_name,
                "result_name": profile.result_name,
                "role": role,
                "component": item.get("component", ""),
                "migration_status": item.get("migration_status", ""),
                "migration_priority": item.get("migration_priority", ""),
                "note": item.get("note", ""),
            }
        )
    out = write_csv(
        stage_paths(profile)["migration_audit_csv"],
        rows,
        fields=["case_name", "result_name", "role", "component", "migration_status", "migration_priority", "note"],
    )
    return str(out)
