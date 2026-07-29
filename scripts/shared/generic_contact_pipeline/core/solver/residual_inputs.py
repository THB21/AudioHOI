from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..state.geometry_provider import GeometryProvider


@dataclass(frozen=True)
class ResidualInputRequest:
    """Case-independent request for one compiled factor's runtime inputs."""

    factor_id: str
    residual_fn_ref: str
    input_ids: tuple[str, ...]
    gate_provenance: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.factor_id or not self.residual_fn_ref:
            raise ValueError("residual input requests require factor_id and residual_fn_ref")
        if not self.input_ids:
            raise ValueError("residual input requests require input_ids")


ResidualInputProvider = Callable[[ResidualInputRequest], dict[str, Any] | None]


@dataclass(frozen=True)
class WorldSpaceContactSample:
    frame: int
    source_xyz_m: tuple[float, float, float]


@dataclass(frozen=True)
class ContactFactorInput:
    geometry_provider: GeometryProvider
    samples: tuple[WorldSpaceContactSample, ...]
    object_feature_id: str
    weight: float = 1.0
    sigma_m: float = 1.0


@dataclass(frozen=True)
class PosePriorFactorInput:
    state: tuple[float, ...]
    reference: tuple[float, ...]
    initial: tuple[float, ...]
    rot_bound: float = 1.0
    xy_bound: float = 1.0
    z_bound: float = 1.0
    rotation_weight: float = 1.0
    xy_weight: float = 1.0
    z_weight: float = 1.0


@dataclass(frozen=True)
class PeriodicPhaseFactorInput:
    values: tuple[float, ...]
    target: tuple[float, ...]
    weight: float = 1.0
    sigma_rad: float = 1.0


@dataclass(frozen=True)
class JointLimitFactorInput:
    values: tuple[float, ...]
    lower: float | None
    upper: float | None
    weight: float = 1.0
    sigma_rad: float = 1.0


@dataclass(frozen=True)
class GaugeFactorInput:
    values: tuple[float, ...]
    target: float = 0.0
    weight: float = 1.0
    sigma: float = 1.0


def _numeric_vector(values: Sequence[float], label: str) -> list[float]:
    vector = [float(value) for value in values]
    if not vector:
        raise ValueError(f"{label} must not be empty")
    return vector


def build_residual_input_bundle(
    residual_execution_plan: dict[str, object] | object,
    providers_by_residual_ref: Mapping[str, ResidualInputProvider],
) -> dict[str, dict[str, Any]]:
    """Resolve explicit inputs for compiled factors by residual capability.

    Providers are selected by ``residual_fn_ref`` rather than case or object
    identity. Missing providers and providers returning ``None`` leave the
    factor unresolved for the dry-run ledger to report.
    """

    if isinstance(residual_execution_plan, dict):
        records = [record for record in residual_execution_plan.get("records", []) if isinstance(record, dict)]
    else:
        records = [
            {
                "factor_id": record.factor_id,
                "residual_fn_ref": record.residual_fn_ref,
                "input_ids": record.input_ids,
                "gate_provenance": record.gate_provenance,
                "status": record.status,
            }
            for record in getattr(residual_execution_plan, "records", ())
        ]

    bundle: dict[str, dict[str, Any]] = {}
    for record in records:
        if str(record.get("status", "ready_not_executed")) != "ready_not_executed":
            continue
        residual_fn_ref = str(record.get("residual_fn_ref", ""))
        provider = providers_by_residual_ref.get(residual_fn_ref)
        if provider is None:
            continue
        request = ResidualInputRequest(
            factor_id=str(record.get("factor_id", "")),
            residual_fn_ref=residual_fn_ref,
            input_ids=tuple(str(item) for item in record.get("input_ids", ()) if item),
            gate_provenance=tuple(str(item) for item in record.get("gate_provenance", ()) if item),
        )
        payload = provider(request)
        if payload is not None:
            bundle[request.factor_id] = payload
    return bundle


def build_state_regularization_residual_inputs(
    *,
    factor_id: str,
    values: Sequence[Sequence[float]],
    target: Sequence[Sequence[float]],
    scales: tuple[float, ...],
    weight: float,
) -> dict[str, dict[str, Any]]:
    """Build regularization inputs from explicit numeric state vectors.

    CSV fields, case names, and geometry families belong to upstream adapters;
    this solver boundary only receives state/reference vectors and scales.
    """

    numeric_values = [[float(item) for item in row] for row in values]
    numeric_target = [[float(item) for item in row] for row in target]
    if len(numeric_values) != len(numeric_target):
        raise ValueError("regularization values and target must have the same row count")
    if any(len(row) != len(scales) for row in (*numeric_values, *numeric_target)):
        raise ValueError("regularization vectors and scales must have the same width")
    if not numeric_values:
        return {}
    return {
        factor_id: {
            "values": numeric_values,
            "target": numeric_target,
            "weight": float(weight),
            "scales": [[float(scale) for scale in scales] for _ in numeric_values],
        }
    }


def build_metric_depth_residual_inputs(
    *,
    factor_id: str,
    predicted_depth_by_frame: Mapping[int, float],
    target_depth_by_frame: Mapping[int, float],
    weight: float,
    sigma_m: float,
) -> dict[str, dict[str, Any]]:
    """Align predicted and measured metric depth by frame."""

    frames = sorted(set(predicted_depth_by_frame) & set(target_depth_by_frame))
    if not frames:
        return {}
    return {
        factor_id: {
            "predicted_depth_m": [float(predicted_depth_by_frame[frame]) for frame in frames],
            "target_depth_m": [float(target_depth_by_frame[frame]) for frame in frames],
            "weight": float(weight),
            "sigma_m": float(sigma_m),
        }
    }


def build_sequence_temporal_residual_inputs(
    *,
    factor_id: str,
    states_by_frame: Mapping[int, Sequence[float]],
    order: int,
    scales: tuple[float, ...],
    weight: float,
) -> dict[str, dict[str, Any]]:
    """Build full-trajectory first- or second-order temporal pairs."""

    if order not in {1, 2}:
        raise ValueError("temporal residual order must be 1 or 2")
    states = [_numeric_vector(states_by_frame[frame], "temporal state") for frame in sorted(states_by_frame)]
    if any(len(state) != len(scales) for state in states):
        raise ValueError("temporal states and scales must have the same width")
    if len(states) <= order:
        return {}
    if order == 1:
        current = states[1:]
        previous = states[:-1]
    else:
        deltas = [
            [current_value - previous_value for current_value, previous_value in zip(states[index], states[index - 1])]
            for index in range(1, len(states))
        ]
        current = deltas[1:]
        previous = deltas[:-1]
    return {
        factor_id: {
            "x": current,
            "prev": previous,
            "weight": float(weight),
            "scales": [float(scale) for scale in scales],
        }
    }


def build_pose_prior_residual_inputs(
    *,
    factor_id: str,
    state: Sequence[float],
    reference: Sequence[float],
    initial: Sequence[float],
    rot_bound: float,
    xy_bound: float,
    z_bound: float,
    rotation_weight: float,
    xy_weight: float,
    z_weight: float,
) -> dict[str, dict[str, Any]]:
    """Build an explicit six-dimensional SE(3)-tangent pose prior input."""

    numeric_state = _numeric_vector(state, "pose-prior state")
    numeric_reference = _numeric_vector(reference, "pose-prior reference")
    numeric_initial = _numeric_vector(initial, "pose-prior initial state")
    if len(numeric_state) != 6 or len(numeric_reference) != 6 or len(numeric_initial) != 6:
        raise ValueError("pose-prior state, reference, and initial vectors must have width 6")
    return {
        factor_id: {
            "x": numeric_state,
            "ref": numeric_reference,
            "init": numeric_initial,
            "rot_bound": float(rot_bound),
            "xy_bound": float(xy_bound),
            "z_bound": float(z_bound),
            "w_prior_rot": float(rotation_weight),
            "w_prior_xy": float(xy_weight),
            "w_prior_z": float(z_weight),
        }
    }


def build_world_space_contact_residual_inputs(
    *,
    factor_id: str,
    geometry_provider: GeometryProvider,
    object_states: Mapping[int, Sequence[float]],
    source_sites: Mapping[int, Sequence[float]],
    active_frames: Iterable[int],
    object_feature_id: str,
    weight: float,
    sigma_m: float,
) -> dict[str, dict[str, Any]]:
    """Resolve world-space entity-site pairs for a contact factor."""

    samples = [
        WorldSpaceContactSample(frame, tuple(float(value) for value in source_sites[frame]))
        for frame in sorted(set(int(value) for value in active_frames))
        if frame in source_sites
    ]
    return build_world_space_contact_sample_residual_inputs(
        factor_id=factor_id,
        geometry_provider=geometry_provider,
        object_states=object_states,
        samples=samples,
        object_feature_id=object_feature_id,
        weight=weight,
        sigma_m=sigma_m,
    )


def build_world_space_contact_sample_residual_inputs(
    *,
    factor_id: str,
    geometry_provider: GeometryProvider,
    object_states: Mapping[int, Sequence[float]],
    samples: Iterable[WorldSpaceContactSample],
    object_feature_id: str,
    weight: float,
    sigma_m: float,
) -> dict[str, dict[str, Any]]:
    """Resolve repeated world-space contact samples against object geometry."""

    anchors: list[list[float]] = []
    targets: list[list[float]] = []
    for sample in samples:
        state = object_states.get(int(sample.frame))
        if state is None:
            continue
        source_xyz = [float(value) for value in sample.source_xyz_m]
        if len(source_xyz) != 3:
            raise ValueError("world-space source sites must have exactly three coordinates")
        target_xyz = geometry_provider.contact_point_world(state, object_feature_id, source_xyz)
        anchors.append(source_xyz)
        targets.append([float(value) for value in target_xyz])
    if not anchors:
        return {}
    return {
        factor_id: {
            "anchors": anchors,
            "targets": targets,
            "weight": float(weight),
            "sigma_m": float(sigma_m),
        }
    }


def build_periodic_phase_prior_residual_inputs(
    *,
    factor_id: str,
    values: Sequence[float],
    target: Sequence[float],
    weight: float,
    sigma_rad: float,
) -> dict[str, dict[str, Any]]:
    numeric_values = [float(value) for value in values]
    numeric_target = [float(value) for value in target]
    if len(numeric_values) != len(numeric_target):
        raise ValueError("periodic phase values and targets must have the same length")
    if not numeric_values:
        return {}
    return {
        factor_id: {
            "values": numeric_values,
            "target": numeric_target,
            "weight": float(weight),
            "sigma_rad": float(sigma_rad),
        }
    }


def build_joint_limit_residual_inputs(
    *,
    factor_id: str,
    values: Sequence[float],
    lower: float | None,
    upper: float | None,
    weight: float,
    sigma_rad: float,
) -> dict[str, dict[str, Any]]:
    numeric_values = [float(value) for value in values]
    if not numeric_values:
        return {}
    return {
        factor_id: {
            "values": numeric_values,
            "lower": None if lower is None else float(lower),
            "upper": None if upper is None else float(upper),
            "weight": float(weight),
            "sigma_rad": float(sigma_rad),
        }
    }


def build_gauge_constraint_residual_inputs(
    *,
    factor_id: str,
    values: Sequence[float],
    target: float,
    weight: float,
    sigma: float,
) -> dict[str, dict[str, Any]]:
    numeric_values = [float(value) for value in values]
    if not numeric_values:
        return {}
    return {
        factor_id: {
            "values": numeric_values,
            "target": float(target),
            "weight": float(weight),
            "sigma": float(sigma),
        }
    }


def build_geometry_sequence_residual_input_bundle(
    residual_execution_plan: dict[str, object] | object,
    *,
    object_states: Mapping[int, Sequence[float]],
    state_scales: tuple[float, ...],
    reference_states: Mapping[int, Sequence[float]] | None = None,
    contact_factors: Mapping[str, ContactFactorInput] | None = None,
    pose_prior_factors: Mapping[str, PosePriorFactorInput] | None = None,
    periodic_phase_factors: Mapping[str, PeriodicPhaseFactorInput] | None = None,
    joint_limit_factors: Mapping[str, JointLimitFactorInput] | None = None,
    gauge_factors: Mapping[str, GaugeFactorInput] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build common sequence residuals from explicit state and factor inputs.

    Factor-specific data is keyed by factor id. Geometry family and object
    identity never select an executor or residual implementation here.
    """

    def temporal(request: ResidualInputRequest) -> dict[str, Any] | None:
        order = 1 if request.residual_fn_ref == "shadow_residual::temporal_velocity" else 2
        payload = build_sequence_temporal_residual_inputs(
            factor_id=request.factor_id,
            states_by_frame=object_states,
            order=order,
            scales=state_scales,
            weight=1.0,
        )
        return payload.get(request.factor_id)

    def regularization(request: ResidualInputRequest) -> dict[str, Any] | None:
        if reference_states is None:
            return None
        frames = sorted(set(object_states) & set(reference_states))
        payload = build_state_regularization_residual_inputs(
            factor_id=request.factor_id,
            values=[object_states[frame] for frame in frames],
            target=[reference_states[frame] for frame in frames],
            scales=state_scales,
            weight=1.0,
        )
        return payload.get(request.factor_id)

    def contact(request: ResidualInputRequest) -> dict[str, Any] | None:
        factor = (contact_factors or {}).get(request.factor_id)
        if factor is None:
            return None
        payload = build_world_space_contact_sample_residual_inputs(
            factor_id=request.factor_id,
            geometry_provider=factor.geometry_provider,
            object_states=object_states,
            samples=factor.samples,
            object_feature_id=factor.object_feature_id,
            weight=factor.weight,
            sigma_m=factor.sigma_m,
        )
        return payload.get(request.factor_id)

    def pose_prior(request: ResidualInputRequest) -> dict[str, Any] | None:
        factor = (pose_prior_factors or {}).get(request.factor_id)
        if factor is None:
            return None
        payload = build_pose_prior_residual_inputs(
            factor_id=request.factor_id,
            state=factor.state,
            reference=factor.reference,
            initial=factor.initial,
            rot_bound=factor.rot_bound,
            xy_bound=factor.xy_bound,
            z_bound=factor.z_bound,
            rotation_weight=factor.rotation_weight,
            xy_weight=factor.xy_weight,
            z_weight=factor.z_weight,
        )
        return payload.get(request.factor_id)

    def periodic_phase(request: ResidualInputRequest) -> dict[str, Any] | None:
        factor = (periodic_phase_factors or {}).get(request.factor_id)
        if factor is None:
            return None
        payload = build_periodic_phase_prior_residual_inputs(
            factor_id=request.factor_id,
            values=factor.values,
            target=factor.target,
            weight=factor.weight,
            sigma_rad=factor.sigma_rad,
        )
        return payload.get(request.factor_id)

    def joint_limit(request: ResidualInputRequest) -> dict[str, Any] | None:
        factor = (joint_limit_factors or {}).get(request.factor_id)
        if factor is None:
            return None
        payload = build_joint_limit_residual_inputs(
            factor_id=request.factor_id,
            values=factor.values,
            lower=factor.lower,
            upper=factor.upper,
            weight=factor.weight,
            sigma_rad=factor.sigma_rad,
        )
        return payload.get(request.factor_id)

    def gauge(request: ResidualInputRequest) -> dict[str, Any] | None:
        factor = (gauge_factors or {}).get(request.factor_id)
        if factor is None:
            return None
        payload = build_gauge_constraint_residual_inputs(
            factor_id=request.factor_id,
            values=factor.values,
            target=factor.target,
            weight=factor.weight,
            sigma=factor.sigma,
        )
        return payload.get(request.factor_id)

    return build_residual_input_bundle(
        residual_execution_plan,
        {
            "shadow_residual::temporal_velocity": temporal,
            "shadow_residual::temporal_acceleration": temporal,
            "shadow_residual::regularization": regularization,
            "shadow_residual::contact_distance": contact,
            "shadow_residual::pose_prior": pose_prior,
            "shadow_residual::periodic_phase_prior": periodic_phase,
            "shadow_residual::joint_limit": joint_limit,
            "shadow_residual::gauge_constraint": gauge,
        },
    )
