"""Compile typed contact constraints into rigid StateSpec hypotheses."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np

from ..contact_constraints import ContactConstraint, ContactState, LocalXYZ
from ..human_sites import HumanSiteMeasurement
from ..state import StateSpec
from .rigid_correspondence import StateSpecRigidCorrespondenceInitializer


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _normalized_body_part(body_part: str) -> str:
    # GVHMR exposes a hand-center skeleton site; legacy contact tables may call
    # the same coarse observation a palm. This is a schema alias, not a case.
    return "hand" if body_part == "palm" else body_part


def _site_key(frame: int, body_part: str, side: str) -> tuple[int, str, str]:
    return frame, _normalized_body_part(body_part), side


@dataclass(frozen=True)
class RigidContactHypothesis:
    frame: int
    state: tuple[float, ...]
    constraint_ids: tuple[str, str]
    human_site_measurement_ids: tuple[str, str]
    target_feature_ids: tuple[str, str]
    correspondence_metrics: dict[str, object]

    def __post_init__(self) -> None:
        if self.frame < 1 or not self.state:
            raise ValueError("rigid contact hypothesis requires a positive frame and state")
        if any(len(values) != 2 for values in (
            self.constraint_ids,
            self.human_site_measurement_ids,
            self.target_feature_ids,
        )):
            raise ValueError("rigid contact hypothesis requires exactly two typed correspondences")


@dataclass(frozen=True)
class RigidContactHypothesisLedger:
    schema_version: int
    state_spec_id: str
    input_state_frame_count: int
    eligible_contact_frame_count: int
    seeded_frame_count: int
    skipped_by_reason: dict[str, int]
    hypotheses: tuple[RigidContactHypothesis, ...]
    site_match_policy: str
    case_dispatch_used: bool
    baseline_pose_read: bool
    human_state_optimized: bool
    accepted_outputs_written: bool
    canonical_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != 1 or not self.state_spec_id:
            raise ValueError("invalid rigid contact hypothesis ledger")
        if self.seeded_frame_count != len(self.hypotheses):
            raise ValueError("rigid contact hypothesis count mismatch")
        if self.seeded_frame_count + sum(self.skipped_by_reason.values()) != self.eligible_contact_frame_count:
            raise ValueError("rigid contact hypothesis coverage does not match eligible frames")
        if self.case_dispatch_used or self.baseline_pose_read or self.human_state_optimized:
            raise ValueError("rigid contact hypotheses must remain generic and object-only")
        if self.accepted_outputs_written:
            raise ValueError("rigid contact hypothesis preparation cannot publish accepted output")


def build_typed_rigid_contact_hypotheses(
    *,
    state_spec: StateSpec,
    initial_states: Mapping[int, Sequence[float]],
    contact_constraints: Sequence[ContactConstraint],
    human_sites: Sequence[HumanSiteMeasurement],
) -> RigidContactHypothesisLedger:
    """Seed frames with exactly two resolved LocalXYZ contact correspondences.

    Geometry families with a different observation rank are deliberately left
    to their own capability initializer: one LocalXYZ does not observe rigid
    rotation, LineS belongs to a line initializer, and surface-only contact
    belongs to a sphere/mesh surface initializer.
    """

    if not initial_states:
        raise ValueError("typed rigid contact hypotheses require initial states")
    sample_ids = {
        *(constraint.sample_id for constraint in contact_constraints),
        *(measurement.sample_id for measurement in human_sites),
    }
    if len(sample_ids) != 1:
        raise ValueError("typed rigid contact hypotheses require one sample-aligned input set")

    sites_by_key: dict[tuple[int, str, str], HumanSiteMeasurement] = {}
    for measurement in human_sites:
        key = _site_key(measurement.frame, measurement.site.body_part, measurement.site.side)
        if key in sites_by_key:
            raise ValueError(f"duplicate human-site measurement for {key}")
        sites_by_key[key] = measurement

    constraints_by_frame: dict[int, list[ContactConstraint]] = {}
    for constraint in contact_constraints:
        if constraint.state not in {ContactState.ACTIVE, ContactState.OCCLUDED_HOLD}:
            continue
        if not isinstance(constraint.object_coordinate, LocalXYZ):
            continue
        for frame in range(constraint.interval.start_frame, constraint.interval.end_frame + 1):
            if frame in initial_states:
                constraints_by_frame.setdefault(frame, []).append(constraint)

    initializer = StateSpecRigidCorrespondenceInitializer(state_spec)
    hypotheses: list[RigidContactHypothesis] = []
    skipped = Counter()
    for frame, constraints in sorted(constraints_by_frame.items()):
        if len(constraints) != 2:
            skipped["requires_exactly_two_local_xyz_contacts"] += 1
            continue
        ordered = tuple(sorted(constraints, key=lambda item: item.constraint_id))
        measurements: list[HumanSiteMeasurement] = []
        for constraint in ordered:
            measurement = sites_by_key.get(
                _site_key(frame, constraint.human_site.body_part, constraint.human_site.side)
            )
            if measurement is None:
                break
            measurements.append(measurement)
        if len(measurements) != 2:
            skipped["missing_frame_aligned_human_site"] += 1
            continue

        local = np.asarray(
            [
                (
                    constraint.object_coordinate.x_m,
                    constraint.object_coordinate.y_m,
                    constraint.object_coordinate.z_m,
                )
                for constraint in ordered
            ],
            dtype=float,
        )
        target = np.asarray([measurement.xyz_m for measurement in measurements], dtype=float)
        state, metrics = initializer.align_two_points(initial_states[frame], local, target)
        if not bool(metrics.get("used")):
            skipped["degenerate_two_point_correspondence"] += 1
            continue
        hypotheses.append(
            RigidContactHypothesis(
                frame=frame,
                state=state,
                constraint_ids=(ordered[0].constraint_id, ordered[1].constraint_id),
                human_site_measurement_ids=(
                    measurements[0].measurement_id,
                    measurements[1].measurement_id,
                ),
                target_feature_ids=(
                    ordered[0].object_feature.geometry_feature_id,
                    ordered[1].object_feature.geometry_feature_id,
                ),
                correspondence_metrics=metrics,
            )
        )

    payload = {
        "schema_version": 1,
        "state_spec_id": state_spec.spec_id,
        "input_state_frame_count": len(initial_states),
        "eligible_contact_frame_count": len(constraints_by_frame),
        "seeded_frame_count": len(hypotheses),
        "skipped_by_reason": dict(sorted(skipped.items())),
        "hypotheses": [asdict(hypothesis) for hypothesis in hypotheses],
        "site_match_policy": "exact_frame_part_side_with_palm_to_hand_schema_alias",
        "case_dispatch_used": False,
        "baseline_pose_read": False,
        "human_state_optimized": False,
        "accepted_outputs_written": False,
    }
    return RigidContactHypothesisLedger(
        **{key: value for key, value in payload.items() if key != "hypotheses"},
        hypotheses=tuple(hypotheses),
        canonical_sha256=_canonical_hash(payload),
    )


def apply_rigid_contact_hypotheses(
    initial_states: Mapping[int, Sequence[float]],
    ledger: RigidContactHypothesisLedger,
) -> dict[int, tuple[float, ...]]:
    states = {
        int(frame): tuple(float(value) for value in state)
        for frame, state in initial_states.items()
    }
    for hypothesis in ledger.hypotheses:
        if hypothesis.frame not in states:
            raise ValueError(f"rigid contact hypothesis references missing frame {hypothesis.frame}")
        states[hypothesis.frame] = hypothesis.state
    return states


def rigid_contact_hypothesis_ledger_record(
    ledger: RigidContactHypothesisLedger,
) -> dict[str, object]:
    return asdict(ledger)
