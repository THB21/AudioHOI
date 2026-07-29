"""Legacy artifact adapter for the generic object sequence problem factory."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from ..base.config import CaseProfile
from ..contact_constraints import ContactConstraint, ContactState, LocalXYZ, adapt_legacy_contact_rows
from ..human_sites import GVHMRSiteExtractionResult, HumanSiteMeasurement, extract_gvhmr_site_measurements
from ..measurements import Line2DMeasurement, adapt_legacy_observation_rows
from ..state import (
    AssetGeometryBuildResult,
    StateAdaptationResult,
    adapt_legacy_state_rows,
    build_articulated_geometry_from_asset_descriptor,
)
from .problem import build_sequence_problem_shadow
from .problem_factory import (
    SequenceFactorInputs,
    SequenceProblemFactory,
    SequenceProblemPreparation,
    sequence_problem_preparation_record,
)
from .residual_inputs import ContactFactorInput, LineReprojectionFactorInput, WorldSpaceContactSample
from ..state import PinholeCamera


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"typed object problem input is empty: {path}")
    return rows


def _state_vectors(
    adaptation: StateAdaptationResult,
    rows: Sequence[Mapping[str, str]],
) -> dict[int, tuple[float, ...]]:
    states: dict[int, tuple[float, ...]] = {}
    for row in rows:
        values: list[float] = []
        for dof in adaptation.state_spec.dofs:
            values.extend(float(row[field]) for field in dof.source_fields)
        frame = int(row["frame"])
        if frame in states:
            raise ValueError(f"duplicate initial object state frame: {frame}")
        states[frame] = tuple(values)
    return states


def _normalized_site(body_part: str, side: str) -> tuple[str, str]:
    normalized_part = "hand" if body_part == "palm" else body_part
    return normalized_part, side


def _contact_samples(
    constraints: Sequence[ContactConstraint],
    human_sites: Sequence[HumanSiteMeasurement],
) -> tuple[WorldSpaceContactSample, ...]:
    sites = {
        (measurement.frame, *_normalized_site(measurement.site.body_part, measurement.site.side)): measurement
        for measurement in human_sites
    }
    samples: list[WorldSpaceContactSample] = []
    for constraint in constraints:
        if constraint.state not in {ContactState.ACTIVE, ContactState.OCCLUDED_HOLD}:
            continue
        if not isinstance(constraint.object_coordinate, LocalXYZ):
            continue
        for frame in range(constraint.interval.start_frame, constraint.interval.end_frame + 1):
            key = (frame, *_normalized_site(constraint.human_site.body_part, constraint.human_site.side))
            measurement = sites.get(key)
            if measurement is None:
                continue
            samples.append(
                WorldSpaceContactSample(
                    frame=frame,
                    source_xyz_m=measurement.xyz_m,
                    object_feature_id=constraint.object_feature.geometry_feature_id,
                )
            )
    return tuple(samples)


def _configured_records(execution_plan: Mapping[str, object]) -> tuple[dict[str, object], ...]:
    return tuple(
        dict(record)
        for record in execution_plan.get("records", ())
        if isinstance(record, Mapping)
        and record.get("status") == "ready_not_executed"
        and isinstance(record.get("runtime_config"), Mapping)
    )


def _state_scales(records: Sequence[Mapping[str, object]], width: int) -> tuple[float, ...]:
    configured = {
        tuple(float(value) for value in record["runtime_config"]["state_scales"])
        for record in records
        if isinstance(record.get("runtime_config"), Mapping)
        and record["runtime_config"].get("state_scales") is not None
    }
    if not configured:
        return (1.0,) * width
    if len(configured) != 1 or len(next(iter(configured))) != width:
        raise ValueError("configured factor state scales are inconsistent with StateSpec")
    return next(iter(configured))


@dataclass(frozen=True)
class LegacyObjectProblemPreparation:
    preparation: SequenceProblemPreparation
    state_adaptation: StateAdaptationResult
    geometry: AssetGeometryBuildResult
    gvhmr_sites: GVHMRSiteExtractionResult
    measurement_count: int
    contact_constraint_count: int
    case_dispatch_used: bool = False
    human_state_optimized: bool = False
    accepted_outputs_written: bool = False

    def __post_init__(self) -> None:
        if self.case_dispatch_used or self.human_state_optimized or self.accepted_outputs_written:
            raise ValueError("legacy object problem preparation must remain object-only")


def legacy_object_problem_preparation_record(
    prepared: LegacyObjectProblemPreparation,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "mode": "generic_object_sequence_problem_preparation",
        "problem": sequence_problem_preparation_record(prepared.preparation),
        "state_spec_id": prepared.state_adaptation.state_spec.spec_id,
        "geometry": {
            "descriptor_path": prepared.geometry.descriptor_path,
            "descriptor_sha256": prepared.geometry.descriptor_sha256,
            "resource_path": prepared.geometry.resource_path,
            "resource_sha256": prepared.geometry.resource_sha256,
            "semantic_segments_path": prepared.geometry.semantic_segments_path,
            "semantic_segments_sha256": prepared.geometry.semantic_segments_sha256,
            "feature_ids": list(prepared.geometry.feature_ids),
            "contact_feature_ids": list(prepared.geometry.contact_feature_ids),
            "canonical_sha256": prepared.geometry.canonical_sha256,
        },
        "read_only_human_sites": {
            "schema": prepared.gvhmr_sites.schema,
            "measurement_count": len(prepared.gvhmr_sites.measurements),
            "source_artifact": prepared.gvhmr_sites.source_artifact,
            "source_sha256": prepared.gvhmr_sites.source_sha256,
            "body_model_artifact": prepared.gvhmr_sites.body_model_artifact,
            "body_model_sha256": prepared.gvhmr_sites.body_model_sha256,
            "read_only": prepared.gvhmr_sites.read_only,
        },
        "measurement_count": prepared.measurement_count,
        "contact_constraint_count": prepared.contact_constraint_count,
        "case_dispatch_used": prepared.case_dispatch_used,
        "human_state_optimized": prepared.human_state_optimized,
        "accepted_outputs_written": prepared.accepted_outputs_written,
    }


def prepare_legacy_articulated_object_problem(
    *,
    profile: CaseProfile,
    result_dir: Path,
    repository_root: Path,
    body_models_root: Path,
) -> LegacyObjectProblemPreparation:
    """Adapt current-run artifacts and delegate executable assembly to the generic factory."""

    pose_path = result_dir / "object_pose_init.csv"
    observation_path = result_dir / "object_observations.csv"
    contact_path = result_dir / "object_contact_points.csv"
    pose_rows = _rows(pose_path)
    observation_rows = _rows(observation_path)
    contact_rows = _rows(contact_path)
    state_adaptation = adapt_legacy_state_rows(profile, pose_rows, str(pose_path))
    initial_states = _state_vectors(state_adaptation, pose_rows)
    measurements = adapt_legacy_observation_rows(profile.case_name, observation_rows, str(observation_path)).measurements
    constraints = adapt_legacy_contact_rows(profile.case_name, contact_rows, str(contact_path)).constraints
    frame_times = {int(row["frame"]): float(row["time"]) for row in pose_rows}
    gvhmr_sites = extract_gvhmr_site_measurements(
        sample_id=profile.case_name,
        result_pkl=profile.sample_dir / "results/gvhmr/result.pkl",
        body_models_root=body_models_root,
        frame_times=frame_times,
    )
    descriptor_path = repository_root / str(profile.data["geometry_asset_descriptor"])
    geometry = build_articulated_geometry_from_asset_descriptor(
        descriptor_path=descriptor_path,
        repository_root=repository_root,
        result_dir=result_dir,
        state_spec=state_adaptation.state_spec,
        contact_constraints=constraints,
    )
    shadow = build_sequence_problem_shadow(profile, result_dir)
    execution_plan = shadow["residual_execution_plan"]
    records = _configured_records(execution_plan)
    camera = PinholeCamera(**profile.camera)
    cameras = {frame: camera for frame in initial_states}
    lines = tuple(measurement for measurement in measurements if isinstance(measurement, Line2DMeasurement))
    samples = _contact_samples(constraints, gvhmr_sites.measurements)
    contact_factors: dict[str, ContactFactorInput] = {}
    line_factors: dict[str, LineReprojectionFactorInput] = {}
    for record in records:
        factor_id = str(record["factor_id"])
        residual_ref = str(record["residual_fn_ref"])
        if residual_ref == "shadow_residual::contact_distance":
            contact_factors[factor_id] = ContactFactorInput(geometry.provider, samples, None)
        elif residual_ref == "shadow_residual::line_reprojection":
            line_factors[factor_id] = LineReprojectionFactorInput(
                geometry.provider,
                lines,
                cameras,
                allow_endpoint_swap=True,
            )
    factor_inputs = SequenceFactorInputs(
        state_scales=_state_scales(records, sum(dof.dimension for dof in state_adaptation.state_spec.dofs)),
        contact_factors=contact_factors,
        line_reprojection_factors=line_factors,
    )
    preparation = SequenceProblemFactory().prepare(
        attempt_id=str(shadow["attempt_ledger"]["attempt_id"]),
        sequence_contract_sha256=str(shadow["sequence_problem_contract"]["canonical_sha256"]),
        state_spec=state_adaptation.state_spec,
        initial_states=initial_states,
        residual_execution_plan=execution_plan,
        factor_inputs=factor_inputs,
        contact_constraints=constraints,
        human_sites=gvhmr_sites.measurements,
    )
    return LegacyObjectProblemPreparation(
        preparation=preparation,
        state_adaptation=state_adaptation,
        geometry=geometry,
        gvhmr_sites=gvhmr_sites,
        measurement_count=len(measurements),
        contact_constraint_count=len(constraints),
    )
