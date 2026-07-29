"""Prepare executable object sequence problems from typed solver inputs."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Mapping, Sequence

from ..contact_constraints import ContactConstraint
from ..human_sites import HumanSiteMeasurement
from ..state import StateSpec
from .contact_correspondence import (
    RigidContactHypothesisLedger,
    apply_rigid_contact_hypotheses,
    build_typed_rigid_contact_hypotheses,
)
from .optimization import SequenceOptimizationProblem
from .parameterization import StateSpecParameterization
from .residual_inputs import (
    AudioAlignmentFactorInput,
    ContactFactorInput,
    GaugeFactorInput,
    JointLimitFactorInput,
    LineReprojectionFactorInput,
    PeriodicPhaseFactorInput,
    PosePriorFactorInput,
    SupportPlaneFactorInput,
    build_geometry_sequence_residual_dependencies,
    build_geometry_sequence_residual_input_bundle,
)


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _plan_records(plan: dict[str, object] | object) -> tuple[dict[str, object], ...]:
    if isinstance(plan, dict):
        return tuple(
            dict(record)
            for record in plan.get("records", ())
            if isinstance(record, Mapping)
        )
    return tuple(
        {
            "factor_id": record.factor_id,
            "residual_fn_ref": record.residual_fn_ref,
            "runtime_config": record.runtime_config,
            "status": record.status,
        }
        for record in getattr(plan, "records", ())
    )


@dataclass(frozen=True)
class SequenceFactorInputs:
    """Factor-id keyed typed inputs; no object identity or case dispatch."""

    state_scales: tuple[float, ...]
    reference_states: Mapping[int, Sequence[float]] | None = None
    contact_factors: Mapping[str, ContactFactorInput] = field(default_factory=dict)
    pose_prior_factors: Mapping[str, PosePriorFactorInput] = field(default_factory=dict)
    periodic_phase_factors: Mapping[str, PeriodicPhaseFactorInput] = field(default_factory=dict)
    joint_limit_factors: Mapping[str, JointLimitFactorInput] = field(default_factory=dict)
    gauge_factors: Mapping[str, GaugeFactorInput] = field(default_factory=dict)
    audio_alignment_factors: Mapping[str, AudioAlignmentFactorInput] = field(default_factory=dict)
    line_reprojection_factors: Mapping[str, LineReprojectionFactorInput] = field(default_factory=dict)
    support_plane_factors: Mapping[str, SupportPlaneFactorInput] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.state_scales or any(float(value) <= 0.0 for value in self.state_scales):
            raise ValueError("sequence factor inputs require positive state scales")


@dataclass(frozen=True)
class SequenceProblemPreparation:
    problem: SequenceOptimizationProblem
    selected_factor_ids: tuple[str, ...]
    residual_dependency_count: int
    initialization_ledger: RigidContactHypothesisLedger | None
    initial_residual_inputs_sha256: str
    case_dispatch_used: bool
    human_state_optimized: bool
    accepted_outputs_written: bool
    canonical_sha256: str

    def __post_init__(self) -> None:
        if self.selected_factor_ids != self.problem.factor_ids:
            raise ValueError("problem preparation factor ids must match the executable problem")
        if self.residual_dependency_count != len(self.problem.residual_dependencies):
            raise ValueError("problem preparation dependency count mismatch")
        if self.case_dispatch_used or self.human_state_optimized or self.accepted_outputs_written:
            raise ValueError("sequence problem preparation must remain generic and object-only")


class SequenceProblemFactory:
    """Build one executable generic problem from typed object-side contracts."""

    def prepare(
        self,
        *,
        attempt_id: str,
        sequence_contract_sha256: str,
        state_spec: StateSpec,
        initial_states: Mapping[int, Sequence[float]],
        residual_execution_plan: dict[str, object] | object,
        factor_inputs: SequenceFactorInputs,
        contact_constraints: Sequence[ContactConstraint] = (),
        human_sites: Sequence[HumanSiteMeasurement] = (),
        parent_solve_attempt_id: str | None = None,
    ) -> SequenceProblemPreparation:
        if not initial_states:
            raise ValueError("sequence problem factory requires object initial states")
        states = {
            int(frame): tuple(float(value) for value in state)
            for frame, state in initial_states.items()
        }
        frames = tuple(sorted(states))
        if tuple(states) != frames:
            states = {frame: states[frame] for frame in frames}
        parameterization = StateSpecParameterization.from_state_spec(state_spec)
        if len(factor_inputs.state_scales) != parameterization.state_width:
            raise ValueError("factor state scales must match StateSpec width")
        if any(len(state) != parameterization.state_width for state in states.values()):
            raise ValueError("initial object states must match StateSpec width")

        initialization_ledger: RigidContactHypothesisLedger | None = None
        if contact_constraints or human_sites:
            if not contact_constraints or not human_sites:
                raise ValueError("typed contact initialization requires constraints and read-only human sites")
            initialization_ledger = build_typed_rigid_contact_hypotheses(
                state_spec=state_spec,
                initial_states=states,
                contact_constraints=contact_constraints,
                human_sites=human_sites,
            )
            states = apply_rigid_contact_hypotheses(states, initialization_ledger)

        def residual_input_builder(
            object_states: Mapping[int, Sequence[float]],
        ) -> dict[str, dict[str, object]]:
            return build_geometry_sequence_residual_input_bundle(
                residual_execution_plan,
                object_states=object_states,
                state_scales=factor_inputs.state_scales,
                reference_states=factor_inputs.reference_states,
                contact_factors=factor_inputs.contact_factors,
                pose_prior_factors=factor_inputs.pose_prior_factors,
                periodic_phase_factors=factor_inputs.periodic_phase_factors,
                joint_limit_factors=factor_inputs.joint_limit_factors,
                gauge_factors=factor_inputs.gauge_factors,
                audio_alignment_factors=factor_inputs.audio_alignment_factors,
                line_reprojection_factors=factor_inputs.line_reprojection_factors,
                support_plane_factors=factor_inputs.support_plane_factors,
            )

        initial_bundle = residual_input_builder(states)
        configured_factor_ids = tuple(
            str(record.get("factor_id", ""))
            for record in _plan_records(residual_execution_plan)
            if record.get("status", "ready_not_executed") == "ready_not_executed"
            and isinstance(record.get("runtime_config"), Mapping)
        )
        if not configured_factor_ids:
            raise ValueError("sequence problem factory requires production-configured factors")
        unresolved = tuple(factor_id for factor_id in configured_factor_ids if factor_id not in initial_bundle)
        if unresolved:
            raise ValueError("configured factors have no typed runtime inputs: " + ",".join(unresolved))

        dependencies = build_geometry_sequence_residual_dependencies(
            residual_execution_plan,
            object_states=states,
            factor_ids=configured_factor_ids,
            reference_states=factor_inputs.reference_states,
            contact_factors=factor_inputs.contact_factors,
            periodic_phase_factors=factor_inputs.periodic_phase_factors,
            line_reprojection_factors=factor_inputs.line_reprojection_factors,
            support_plane_factors=factor_inputs.support_plane_factors,
        )
        seeded = initialization_ledger is not None and initialization_ledger.seeded_frame_count > 0
        problem = SequenceOptimizationProblem(
            attempt_id=attempt_id,
            sequence_contract_sha256=sequence_contract_sha256,
            frames=frames,
            initial_states=tuple(states[frame] for frame in frames),
            factor_ids=configured_factor_ids,
            residual_execution_plan=residual_execution_plan,
            residual_input_builder=residual_input_builder,
            state_parameterization=parameterization,
            residual_dependencies=dependencies,
            parent_solve_attempt_id=parent_solve_attempt_id,
            initialization_kind="typed_rigid_contact_correspondence" if seeded else None,
            initialization_ledger_sha256=(
                initialization_ledger.canonical_sha256 if seeded else None
            ),
        )
        input_hash = _canonical_hash(initial_bundle)
        payload = {
            "attempt_id": attempt_id,
            "sequence_contract_sha256": sequence_contract_sha256,
            "state_spec_id": state_spec.spec_id,
            "frames": frames,
            "selected_factor_ids": configured_factor_ids,
            "residual_dependency_count": len(dependencies),
            "initialization_ledger_sha256": (
                initialization_ledger.canonical_sha256 if initialization_ledger is not None else None
            ),
            "initial_residual_inputs_sha256": input_hash,
            "case_dispatch_used": False,
            "human_state_optimized": False,
            "accepted_outputs_written": False,
        }
        return SequenceProblemPreparation(
            problem=problem,
            selected_factor_ids=configured_factor_ids,
            residual_dependency_count=len(dependencies),
            initialization_ledger=initialization_ledger,
            initial_residual_inputs_sha256=input_hash,
            case_dispatch_used=False,
            human_state_optimized=False,
            accepted_outputs_written=False,
            canonical_sha256=_canonical_hash(payload),
        )


def sequence_problem_preparation_record(
    preparation: SequenceProblemPreparation,
) -> dict[str, object]:
    ledger = preparation.initialization_ledger
    return {
        "schema_version": 1,
        "attempt_id": preparation.problem.attempt_id,
        "sequence_contract_sha256": preparation.problem.sequence_contract_sha256,
        "state_spec_id": (
            preparation.problem.state_parameterization.state_spec.spec_id
            if preparation.problem.state_parameterization is not None
            else None
        ),
        "selected_factor_ids": list(preparation.selected_factor_ids),
        "residual_dependency_count": preparation.residual_dependency_count,
        "initialization_ledger": asdict(ledger) if ledger is not None else None,
        "initial_residual_inputs_sha256": preparation.initial_residual_inputs_sha256,
        "case_dispatch_used": preparation.case_dispatch_used,
        "human_state_optimized": preparation.human_state_optimized,
        "accepted_outputs_written": preparation.accepted_outputs_written,
        "canonical_sha256": preparation.canonical_sha256,
    }
