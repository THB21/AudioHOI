from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from ..base.config import CaseProfile
from ..base.io import repo_relative_value
from ..contact_constraints.shadow import build_contact_constraint_shadow
from ..factors.shadow import build_factor_shadow
from ..interaction import build_interaction_timeline, frame_record, interaction_intervals
from ..measurements.shadow import build_measurement_shadow


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _profile_state_contract(profile: CaseProfile) -> dict[str, object]:
    pose_model = str(profile.data.get("pose_model", ""))
    geometry_model = str(profile.data.get("geometry_model", ""))
    object_family = str(profile.data.get("object_family", ""))
    if pose_model == "semantic_graph_6d":
        state_model = "semantic_graph_6d"
        spec_id = "semantic_graph_6d:articulated"
        geometry_kind = "articulated_urdf"
        required_dofs = ("root.translation", "root.rotation", "joint.front_to_rear", "joint.front_to_seat")
    elif pose_model == "rigid6_plus_phase":
        state_model = "rigid6_plus_phase"
        spec_id = "rigid6_plus_phase:rigid_mesh"
        geometry_kind = "rigid_mesh"
        required_dofs = ("root.translation", "root.rotation", "scale", "handle.phase")
    elif "line_object" in profile.data or object_family == "rigid_staff":
        state_model = "translation3_with_line_orientation_prior"
        spec_id = "translation3:line_capsule"
        geometry_kind = "line_capsule"
        required_dofs = ("root.translation", "root.rotation")
    else:
        state_model = "translation3"
        spec_id = "translation3:sphere"
        geometry_kind = "sphere"
        required_dofs = ("root.translation", "root.rotation")
    return {
        "source": "case_profile_contract",
        "source_fields": sorted(
            field
            for field in (
                "pose_model",
                "geometry_model",
                "object_family",
                "line_object",
                "articraft_urdf",
                "articraft_model_py",
            )
            if field in profile.data
        ),
        "baseline_pose_read": False,
        "state_model": state_model,
        "spec_id": spec_id,
        "geometry_model": geometry_model,
        "geometry_kind": geometry_kind,
        "required_dofs": list(required_dofs),
    }


def _factor_requirements(factor_shadow: dict[str, Any]) -> list[dict[str, object]]:
    records = factor_shadow.get("factors", {}).get("records", [])
    if not isinstance(records, list):
        return []
    requirements: list[dict[str, object]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        requirements.append(
            {
                "factor_id": record.get("factor_id"),
                "kind": record.get("kind"),
                "input_ref_count": len(record.get("input_refs", [])) if isinstance(record.get("input_refs"), list) else 0,
                "gate": record.get("gate"),
                "weight_source": record.get("weight_source"),
                "residual_source": record.get("residual_source"),
                "consumed_by_solver": False,
            }
        )
    return requirements


def _interaction_state_shadow(profile: CaseProfile, result_dir: Path) -> dict[str, object]:
    timeline = build_interaction_timeline(profile.case_name, result_dir)
    frames = [frame_record(state) for state in timeline.frames]
    frames = repo_relative_value(frames)
    intervals = interaction_intervals(timeline)
    contact_states = Counter(str(record["contact_state"]) for record in frames)
    contact_modes = Counter(str(record["contact_mode"]) for record in frames)
    motion_modes = Counter(str(record["motion_mode"]) for record in frames)
    visibility_states = Counter(str(record["visibility_state"]) for record in frames)
    core = {
        "schema_version": timeline.schema_version,
        "sample_id": timeline.sample_id,
        "target_entity_id": timeline.target_entity_id,
        "frame_count": len(frames),
        "frames": frames,
        "intervals": intervals,
        "metrics": timeline.metrics,
    }
    return {
        "source": {
            "producer": "interaction_state_timeline_shadow",
            "inputs": [
                str(repo_relative_value(result_dir / "object_observations.csv")),
                str(repo_relative_value(result_dir / "contact_state_frames.csv")),
                str(repo_relative_value(result_dir / "contact_candidates_internal/audio_events.csv")),
                str(repo_relative_value(result_dir / "events/audio_events.csv")),
            ],
        },
        "frame_count": len(frames),
        "interval_count": len(intervals),
        "by_visibility_state": dict(sorted(visibility_states.items())),
        "by_contact_state": dict(sorted(contact_states.items())),
        "by_contact_mode": dict(sorted(contact_modes.items())),
        "by_motion_mode": dict(sorted(motion_modes.items())),
        "metrics": timeline.metrics,
        "canonical_sha256": _canonical_hash(core),
        "consumed_by_solver": False,
    }


def build_sequence_problem_shadow(profile: CaseProfile, result_dir: Path) -> dict[str, object]:
    """Build a deterministic generic-solver problem manifest without solving it.

    The manifest is allowed to read typed observation/contact/factor traces that
    were already frozen for regression. It deliberately does not read
    ``object_pose_init.csv`` or profile baseline pose paths, because those would
    turn legacy solved poses into an initializer.
    """
    observation_csv = result_dir / "object_observations.csv"
    contact_csv = result_dir / "object_contact_points.csv"
    measurement_shadow = build_measurement_shadow(profile.case_name, observation_csv, _read_csv(observation_csv))
    measurement_shadow["source"]["path"] = str(repo_relative_value(observation_csv))
    contact_shadow = build_contact_constraint_shadow(profile.case_name, contact_csv, _read_csv(contact_csv))
    contact_shadow["source"]["path"] = str(repo_relative_value(contact_csv))
    interaction_shadow = _interaction_state_shadow(profile, result_dir)
    factor_shadow = build_factor_shadow(profile, result_dir)
    state_contract = _profile_state_contract(profile)
    factor_requirements = _factor_requirements(factor_shadow)
    factor_kinds = Counter(str(item["kind"]) for item in factor_requirements)
    problem_core = {
        "state_contract": state_contract,
        "measurements": measurement_shadow["measurements"],
        "constraints": contact_shadow["constraints"],
        "interactions": {
            "frame_count": interaction_shadow["frame_count"],
            "interval_count": interaction_shadow["interval_count"],
            "by_visibility_state": interaction_shadow["by_visibility_state"],
            "by_contact_state": interaction_shadow["by_contact_state"],
            "by_contact_mode": interaction_shadow["by_contact_mode"],
            "by_motion_mode": interaction_shadow["by_motion_mode"],
            "canonical_sha256": interaction_shadow["canonical_sha256"],
        },
        "factor_kinds": dict(sorted(factor_kinds.items())),
        "factor_requirements": factor_requirements,
        "gaps": factor_shadow["gaps"],
    }
    canonical_sha256 = _canonical_hash(problem_core)
    attempt_id = f"shadow-{canonical_sha256[:12]}"
    return {
        "schema_version": 1,
        "mode": "generic_sequence_solver_shadow",
        "sample_id": profile.case_name,
        "result_dir": str(repo_relative_value(result_dir)),
        "solver_executed": False,
        "accepted_outputs_written": False,
        "baseline_pose_read": False,
        "state_contract": state_contract,
        "inputs": {
            "measurement_shadow": {
                "source": measurement_shadow["source"],
                "count": measurement_shadow["measurements"]["count"],
                "frames": measurement_shadow["measurements"]["frames"],
                "canonical_sha256": measurement_shadow["measurements"]["canonical_sha256"],
                "consumed_by_solver": False,
            },
            "contact_constraint_shadow": {
                "source": contact_shadow["source"],
                "count": contact_shadow["constraints"]["count"],
                "canonical_sha256": contact_shadow["constraints"]["canonical_sha256"],
                "consumed_by_solver": False,
            },
            "interaction_state_shadow": interaction_shadow,
            "factor_shadow": {
                "canonical_sha256": factor_shadow["canonical_sha256"],
                "factor_count": factor_shadow["factors"]["count"],
                "factor_kinds": factor_shadow["factors"]["by_kind"],
                "gap_ids": [gap["gap_id"] for gap in factor_shadow["gaps"]],
                "consumed_by_solver": False,
            },
        },
        "problem": {
            "factor_requirements": factor_requirements,
            "factor_kinds": dict(sorted(factor_kinds.items())),
            "gaps": factor_shadow["gaps"],
        },
        "attempt_plan": {
            "attempt_id": attempt_id,
            "mode": "shadow_plan_only",
            "initializer_status": "not_executed",
            "initializer_policy": f"{state_contract['state_model']}:observation_derived_candidate",
            "writes": [],
            "accepted_output_policy": "never_overwrite_accepted_outputs_in_shadow_mode",
            "failure_policy": "record_attempt_diagnostics_only",
        },
        "canonical_sha256": canonical_sha256,
    }
