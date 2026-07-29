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

    anchors: list[list[float]] = []
    targets: list[list[float]] = []
    for frame in sorted(set(int(value) for value in active_frames)):
        state = object_states.get(frame)
        source = source_sites.get(frame)
        if state is None or source is None:
            continue
        source_xyz = [float(value) for value in source]
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
