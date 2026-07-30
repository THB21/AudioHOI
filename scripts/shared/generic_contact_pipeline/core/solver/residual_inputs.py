from __future__ import annotations

from dataclasses import dataclass
from math import atan2, isfinite
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from ..audio_events import AudioEvent
from ..interaction import ContactStateAxis, InteractionContactMode, InteractionTimeline
from ..measurements import Line2DMeasurement, Mask2DMeasurement, MetricDepthMeasurement, Point2DMeasurement
from ..state.geometry_provider import FeaturePointGeometryProvider, GeometryProvider, PinholeCamera, PlaneSurface
from .sparsity import ResidualRowDependency


@dataclass(frozen=True)
class ResidualInputRequest:
    """Case-independent request for one compiled factor's runtime inputs."""

    factor_id: str
    residual_fn_ref: str
    input_ids: tuple[str, ...]
    gate_provenance: tuple[str, ...]
    runtime_config: Mapping[str, object] | None = None
    activation_intervals: tuple[Mapping[str, object], ...] = ()

    def __post_init__(self) -> None:
        if not self.factor_id or not self.residual_fn_ref:
            raise ValueError("residual input requests require factor_id and residual_fn_ref")
        if not self.input_ids:
            raise ValueError("residual input requests require input_ids")
        previous_end = 0
        for interval in self.activation_intervals:
            if not {"start_frame", "end_frame", "status"}.issubset(interval):
                raise ValueError("residual activation intervals require start_frame, end_frame, and status")
            start = int(interval["start_frame"])
            end = int(interval["end_frame"])
            if start < 1 or end < start or start <= previous_end:
                raise ValueError("residual activation intervals must be ordered, non-overlapping positive ranges")
            if str(interval["status"]) not in {"active", "downweighted", "inactive"}:
                raise ValueError("residual activation interval has invalid status")
            previous_end = end


ResidualInputProvider = Callable[[ResidualInputRequest], dict[str, Any] | None]


def _optimization_safe_project(camera: PinholeCamera, points: Any) -> Any:
    """Keep reprojection residuals defined for rejected behind-camera trials."""

    if bool((points[:, 2] <= 1e-4).any()):
        points = points.copy()
        points[:, 2] = points[:, 2].clip(min=1e-4)
    return camera.project(points)


@dataclass(frozen=True)
class WorldSpaceContactSample:
    frame: int
    source_xyz_m: tuple[float, float, float]
    object_feature_id: str | None = None
    line_s: float | None = None
    confidence: float | None = None
    source_offset_xyz_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    contact_track_id: str | None = None

    def __post_init__(self) -> None:
        if self.frame < 1 or len(self.source_xyz_m) != 3 or not all(isfinite(value) for value in self.source_xyz_m):
            raise ValueError("world-space contact samples require a positive frame and xyz source")
        if self.object_feature_id is not None and not self.object_feature_id:
            raise ValueError("world-space contact sample feature id must be nonempty when present")
        if self.line_s is not None and (not isfinite(self.line_s) or not 0.0 <= self.line_s <= 1.0):
            raise ValueError("world-space contact sample line_s must be within [0, 1]")
        if self.confidence is not None and (
            not isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0
        ):
            raise ValueError("world-space contact sample confidence must be within [0, 1]")
        if len(self.source_offset_xyz_m) != 3 or not all(isfinite(value) for value in self.source_offset_xyz_m):
            raise ValueError("world-space contact sample offset must contain three finite coordinates")
        if self.contact_track_id is not None and not self.contact_track_id:
            raise ValueError("world-space contact track id must be nonempty when present")


@dataclass(frozen=True)
class ContactFactorInput:
    geometry_provider: GeometryProvider
    samples: tuple[WorldSpaceContactSample, ...]
    object_feature_id: str | None
    weight: float = 1.0
    sigma_m: float = 1.0
    residual_axes: tuple[int, ...] = (0, 1, 2)

    def __post_init__(self) -> None:
        if not self.residual_axes or any(axis not in {0, 1, 2} for axis in self.residual_axes):
            raise ValueError("contact residual axes must be a nonempty subset of x/y/z")


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
    state_index: int | None = None
    target_by_frame: Mapping[int, float] | None = None


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


@dataclass(frozen=True)
class AudioAlignmentFactorInput:
    events: tuple[AudioEvent, ...]
    timeline: InteractionTimeline
    weight: float = 1.0
    sigma_s: float = 1.0


@dataclass(frozen=True)
class LineReprojectionFactorInput:
    geometry_provider: FeaturePointGeometryProvider
    measurements: tuple[Line2DMeasurement, ...]
    cameras_by_frame: Mapping[int, PinholeCamera]
    weight: float = 1.0
    sigma_px: float = 1.0
    allow_endpoint_swap: bool = False
    constraint_mode: str = "endpoints"

    def __post_init__(self) -> None:
        if self.constraint_mode not in {"endpoints", "axis_line"}:
            raise ValueError("line reprojection constraint mode must be endpoints or axis_line")


@dataclass(frozen=True)
class PointReprojectionFactorInput:
    geometry_provider: FeaturePointGeometryProvider
    measurements: tuple[Point2DMeasurement, ...]
    cameras_by_frame: Mapping[int, PinholeCamera]
    weight: float = 1.0
    sigma_px: float = 1.0


@dataclass(frozen=True)
class MaskSilhouetteFactorInput:
    geometry_provider: FeaturePointGeometryProvider
    measurements: tuple[Mask2DMeasurement, ...]
    cameras_by_frame: Mapping[int, PinholeCamera]
    principal_axis_sigma_rad: float | None = None
    weight: float = 1.0
    sigma_px: float = 1.0

    def __post_init__(self) -> None:
        if self.principal_axis_sigma_rad is not None and self.principal_axis_sigma_rad <= 0.0:
            raise ValueError("mask principal-axis sigma must be positive")


@dataclass(frozen=True)
class MetricDepthFactorInput:
    measurements: tuple[MetricDepthMeasurement, ...]
    state_index: int = 2
    weight: float = 1.0
    sigma_m: float = 1.0
    target_by_frame: Mapping[int, float] | None = None


@dataclass(frozen=True)
class SupportPlaneFactorInput:
    geometry_provider: FeaturePointGeometryProvider
    support_feature_ids: tuple[str, ...]
    active_frames: tuple[int, ...]
    plane: PlaneSurface
    support_weight: float = 1.0
    penetration_weight: float = 1.0
    sigma_m: float = 1.0
    activation_status_by_frame: Mapping[int, str] | None = None
    activation_weight_by_frame: Mapping[int, float] | None = None
    proximity_gate_m: float | None = None
    tangent_gauge_weight: float = 0.0
    tangent_gauge_sigma_rad: float = 1.0


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
                "runtime_config": record.runtime_config,
                "activation_intervals": record.activation_intervals,
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
            runtime_config=(
                dict(record["runtime_config"])
                if isinstance(record.get("runtime_config"), Mapping)
                else None
            ),
            activation_intervals=tuple(
                dict(item)
                for item in record.get("activation_intervals", ())
                if isinstance(item, Mapping)
            ),
        )
        payload = provider(request)
        if payload is not None:
            bundle[request.factor_id] = payload
    return bundle


def _runtime_weight(request: ResidualInputRequest, fallback: float) -> float:
    if request.runtime_config is None:
        return float(fallback)
    return float(request.runtime_config["weight"])


def _runtime_sigma(request: ResidualInputRequest, fallback: float, expected_unit: str) -> float:
    if request.runtime_config is None:
        return float(fallback)
    sigma = request.runtime_config.get("sigma")
    sigma_unit = request.runtime_config.get("sigma_unit")
    if sigma is None or sigma_unit != expected_unit:
        raise ValueError(
            f"factor {request.factor_id} requires sigma_unit={expected_unit}, got {sigma_unit}"
        )
    return float(sigma)


def _runtime_state_scales(
    request: ResidualInputRequest,
    fallback: tuple[float, ...],
) -> tuple[float, ...]:
    if request.runtime_config is None or request.runtime_config.get("state_scales") is None:
        return fallback
    scales = tuple(float(value) for value in request.runtime_config["state_scales"])
    if len(scales) != len(fallback):
        raise ValueError(
            f"factor {request.factor_id} runtime state_scales width {len(scales)} does not match state width {len(fallback)}"
        )
    return scales


def _runtime_activation_tiers(request: ResidualInputRequest) -> dict[str, float]:
    if request.runtime_config is None:
        return {"active": 1.0, "downweighted": 1.0, "inactive": 0.0}
    raw = request.runtime_config.get("activation_weight_tiers")
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"factor {request.factor_id} requires compiled activation_weight_tiers")
    tiers = {str(item[0]): float(item[1]) for item in raw if isinstance(item, (list, tuple)) and len(item) == 2}
    if set(tiers) != {"active", "downweighted", "inactive"}:
        raise ValueError(f"factor {request.factor_id} has invalid activation_weight_tiers")
    return tiers


def _runtime_frame_weight(
    request: ResidualInputRequest,
    frame: int,
    base_weight: float,
) -> float:
    if not request.activation_intervals:
        return float(base_weight)
    status: str | None = None
    for interval in request.activation_intervals:
        start = int(interval["start_frame"])
        end = int(interval["end_frame"])
        if start <= frame <= end:
            status = str(interval["status"])
            break
    if status is None:
        raise ValueError(f"factor {request.factor_id} has no activation status for frame {frame}")
    return float(base_weight) * _runtime_activation_tiers(request)[status]


def _runtime_weights_by_frame(
    request: ResidualInputRequest,
    frames: Iterable[int],
    base_weight: float,
) -> dict[int, float] | None:
    if not request.activation_intervals:
        return None
    return {
        int(frame): _runtime_frame_weight(request, int(frame), base_weight)
        for frame in sorted(set(int(value) for value in frames))
    }


def _scalar_or_row_weights(
    row_weights: Sequence[float],
    fallback: float,
) -> float | list[float]:
    if not row_weights:
        return float(fallback)
    first = float(row_weights[0])
    if all(float(value) == first for value in row_weights[1:]):
        return first
    return [float(value) for value in row_weights]


def build_state_regularization_residual_inputs(
    *,
    factor_id: str,
    values: Sequence[Sequence[float]],
    target: Sequence[Sequence[float]],
    scales: tuple[float, ...],
    weight: float | Sequence[float],
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
    numeric_weight: float | list[float]
    if isinstance(weight, (list, tuple)):
        numeric_weight = [float(value) for value in weight]
        if len(numeric_weight) != len(numeric_values):
            raise ValueError("regularization row weights must match row count")
    else:
        numeric_weight = float(weight)
    return {
        factor_id: {
            "values": numeric_values,
            "target": numeric_target,
            "weight": numeric_weight,
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


def build_line_reprojection_residual_inputs(
    *,
    factor_id: str,
    geometry_provider: FeaturePointGeometryProvider,
    object_states: Mapping[int, Sequence[float]],
    measurements: Sequence[Line2DMeasurement],
    cameras_by_frame: Mapping[int, PinholeCamera],
    weight: float,
    sigma_px: float,
    allow_endpoint_swap: bool,
    constraint_mode: str = "endpoints",
    weight_by_frame: Mapping[int, float] | None = None,
) -> dict[str, dict[str, Any]]:
    predicted: list[list[list[float]]] = []
    target: list[list[list[float]]] = []
    row_weights: list[float] = []
    for measurement in measurements:
        frame = measurement.meta.frame
        state = object_states.get(frame)
        camera = cameras_by_frame.get(frame)
        if state is None or camera is None:
            continue
        points = geometry_provider.feature_points_world(
            state,
            measurement.meta.feature.geometry_feature_id,
        )
        if points.shape != (2, 3):
            raise ValueError("line geometry features must resolve to exactly two 3D endpoints")
        projected = _optimization_safe_project(camera, points)
        predicted.append(projected.astype(float).tolist())
        target.append(
            [
                [float(value) for value in measurement.start_uv],
                [float(value) for value in measurement.end_uv],
            ]
        )
        frame_weight = float((weight_by_frame or {}).get(frame, weight))
        confidence = measurement.meta.confidence
        row_weights.append(frame_weight * (confidence if confidence is not None else 1.0))
    if not predicted:
        return {}
    return {
        factor_id: {
            "predicted": predicted,
            "target": target,
            "weight": _scalar_or_row_weights(row_weights, weight),
            "sigma_px": float(sigma_px),
            "allow_endpoint_swap": bool(allow_endpoint_swap),
            "constraint_mode": constraint_mode,
        }
    }


def build_point_reprojection_residual_inputs(
    *, factor_id: str, geometry_provider: FeaturePointGeometryProvider,
    object_states: Mapping[int, Sequence[float]], measurements: Sequence[Point2DMeasurement],
    cameras_by_frame: Mapping[int, PinholeCamera], weight: float, sigma_px: float,
    weight_by_frame: Mapping[int, float] | None = None,
) -> dict[str, dict[str, Any]]:
    predicted: list[list[float]] = []
    target: list[list[float]] = []
    row_weights: list[float] = []
    for measurement in measurements:
        frame = measurement.meta.frame
        state = object_states.get(frame)
        camera = cameras_by_frame.get(frame)
        if state is None or camera is None:
            continue
        points = geometry_provider.feature_points_world(state, measurement.meta.feature.geometry_feature_id)
        if points.shape != (1, 3):
            raise ValueError("point geometry features must resolve to exactly one 3D point")
        predicted.append(_optimization_safe_project(camera, points)[0].astype(float).tolist())
        target.append([float(measurement.u), float(measurement.v)])
        frame_weight = float((weight_by_frame or {}).get(frame, weight))
        row_weights.append(frame_weight * (measurement.meta.confidence if measurement.meta.confidence is not None else 1.0))
    if not predicted:
        return {}
    return {factor_id: {"predicted": predicted, "target": target, "weight": _scalar_or_row_weights(row_weights, weight), "sigma_px": float(sigma_px)}}


def build_mask_silhouette_residual_inputs(
    *, factor_id: str, geometry_provider: FeaturePointGeometryProvider,
    object_states: Mapping[int, Sequence[float]], measurements: Sequence[Mask2DMeasurement],
    cameras_by_frame: Mapping[int, PinholeCamera], weight: float, sigma_px: float,
    principal_axis_sigma_rad: float | None = None,
    weight_by_frame: Mapping[int, float] | None = None,
) -> dict[str, dict[str, Any]]:
    predicted: list[list[float]] = []
    target: list[list[float]] = []
    predicted_axes: list[list[float]] = []
    target_axes: list[list[float]] = []
    axis_weights: list[float] = []
    row_weights: list[float] = []
    for measurement in measurements:
        frame = measurement.meta.frame
        state = object_states.get(frame)
        camera = cameras_by_frame.get(frame)
        if state is None or camera is None:
            continue
        points = geometry_provider.feature_points_world(state, measurement.meta.feature.geometry_feature_id)
        if points.shape[0] < 4:
            raise ValueError("mask silhouette geometry features must resolve to a nondegenerate point cloud")
        projected = _optimization_safe_project(camera, points)
        predicted.append([
            float(np.min(projected[:, 0])), float(np.min(projected[:, 1])),
            float(np.max(projected[:, 0])), float(np.max(projected[:, 1])),
        ])
        target.append([float(value) for value in measurement.bbox_xyxy])
        frame_weight = float((weight_by_frame or {}).get(frame, weight))
        confidence = measurement.meta.confidence if measurement.meta.confidence is not None else 1.0
        row_weights.append(frame_weight * confidence)
        if principal_axis_sigma_rad is not None:
            if measurement.principal_axis_uv is None or measurement.principal_variances_px2 is None:
                raise ValueError("mask principal-axis factor requires enriched typed mask measurements")
            centered = projected - np.mean(projected, axis=0, keepdims=True)
            covariance = centered.T @ centered / max(1, len(projected))
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            predicted_axis = eigenvectors[:, int(np.argmax(eigenvalues))]
            predicted_axes.append(predicted_axis.astype(float).tolist())
            target_axes.append([float(value) for value in measurement.principal_axis_uv])
            minor, major = measurement.principal_variances_px2
            anisotropy = max(0.0, (major - minor) / max(major + minor, 1e-9))
            axis_weights.append(frame_weight * confidence * anisotropy)
    if not predicted:
        return {}
    return {factor_id: {
        "predicted_bbox": predicted,
        "target_bbox": target,
        "weight": _scalar_or_row_weights(row_weights, weight),
        "sigma_px": float(sigma_px),
        "predicted_principal_axis": predicted_axes if principal_axis_sigma_rad is not None else None,
        "target_principal_axis": target_axes if principal_axis_sigma_rad is not None else None,
        "principal_axis_weight": axis_weights if principal_axis_sigma_rad is not None else None,
        "principal_axis_sigma_rad": principal_axis_sigma_rad,
    }}


def build_metric_depth_measurement_residual_inputs(
    *, factor_id: str, object_states: Mapping[int, Sequence[float]],
    measurements: Sequence[MetricDepthMeasurement], state_index: int,
    weight: float, sigma_m: float, weight_by_frame: Mapping[int, float] | None = None,
    target_by_frame: Mapping[int, float] | None = None,
) -> dict[str, dict[str, Any]]:
    predicted: list[float] = []
    target: list[float] = []
    row_weights: list[float] = []
    for measurement in measurements:
        state = object_states.get(measurement.meta.frame)
        if state is None:
            continue
        predicted.append(float(state[state_index]))
        target.append(float((target_by_frame or {}).get(measurement.meta.frame, measurement.depth_m)))
        frame_weight = float((weight_by_frame or {}).get(measurement.meta.frame, weight))
        row_weights.append(frame_weight * (measurement.meta.confidence if measurement.meta.confidence is not None else 1.0))
    if not predicted:
        return {}
    return {factor_id: {"predicted_depth_m": predicted, "target_depth_m": target, "weight": _scalar_or_row_weights(row_weights, weight), "sigma_m": float(sigma_m)}}


def build_support_plane_residual_inputs(
    *,
    factor_id: str,
    geometry_provider: FeaturePointGeometryProvider,
    object_states: Mapping[int, Sequence[float]],
    support_feature_ids: Sequence[str],
    active_frames: Iterable[int],
    plane: PlaneSurface,
    support_weight: float,
    penetration_weight: float,
    sigma_m: float,
    activation_status_by_frame: Mapping[int, str] | None = None,
    activation_weight_by_frame: Mapping[int, float] | None = None,
    proximity_gate_m: float | None = None,
    tangent_gauge_weight: float = 0.0,
    tangent_gauge_sigma_rad: float = 1.0,
) -> dict[str, dict[str, Any]]:
    distances: list[float] = []
    row_weights: list[float] = []
    for frame in sorted(set(int(value) for value in active_frames)):
        state = object_states.get(frame)
        if state is None:
            continue
        frame_distances: list[float] = []
        for feature_id in support_feature_ids:
            points = geometry_provider.feature_points_world(state, feature_id)
            frame_distances.extend(float(value) for value in plane.signed_distance(points))
        frame_weight = float((activation_weight_by_frame or {}).get(frame, 1.0))
        if (
            proximity_gate_m is not None
            and (activation_status_by_frame or {}).get(frame) != "active"
        ):
            nearest = min(abs(value) for value in frame_distances)
            frame_weight *= max(0.0, min(1.0, 1.0 - nearest / float(proximity_gate_m)))
        distances.extend(frame_distances)
        row_weights.extend([frame_weight] * len(frame_distances))
    if not distances:
        return {}
    payload: dict[str, Any] = {
            "signed_distance_m": distances,
            "support_weight": [float(support_weight) * value for value in row_weights],
            "penetration_weight": [float(penetration_weight) * value for value in row_weights],
            "sigma_m": float(sigma_m),
    }
    tangent_rows: list[float] = []
    tangent_weights: list[float] = []
    if tangent_gauge_weight > 0.0:
        for frame in sorted(set(int(value) for value in active_frames)):
            if (activation_status_by_frame or {}).get(frame) != "active":
                continue
            if frame - 1 not in object_states or frame not in object_states:
                continue
            rotvec = _quaternion_relative_rotvec_world(object_states[frame - 1], object_states[frame])
            tangent_rows.append(float(np.dot(rotvec, np.asarray(plane.normal, dtype=float))))
            tangent_weights.append(
                float(tangent_gauge_weight)
                * float((activation_weight_by_frame or {}).get(frame, 1.0))
            )
    if tangent_rows:
        payload["tangent_twist_rad"] = tangent_rows
        payload["tangent_weight"] = tangent_weights
        payload["tangent_sigma_rad"] = float(tangent_gauge_sigma_rad)
    return {factor_id: payload}


def build_sequence_temporal_residual_inputs(
    *,
    factor_id: str,
    states_by_frame: Mapping[int, Sequence[float]],
    order: int,
    scales: tuple[float, ...],
    weight: float,
    weight_by_frame: Mapping[int, float] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build full-trajectory first- or second-order temporal pairs."""

    if order not in {1, 2}:
        raise ValueError("temporal residual order must be 1 or 2")
    frames = sorted(states_by_frame)
    states = [_numeric_vector(states_by_frame[frame], "temporal state") for frame in frames]
    if any(len(state) != len(scales) for state in states):
        raise ValueError("temporal states and scales must have the same width")
    if len(states) <= order:
        return {}
    if order == 1:
        current = states[1:]
        previous = states[:-1]
        residual_frames = frames[1:]
    else:
        deltas = [
            [current_value - previous_value for current_value, previous_value in zip(states[index], states[index - 1])]
            for index in range(1, len(states))
        ]
        current = deltas[1:]
        previous = deltas[:-1]
        residual_frames = frames[2:]
    return {
        factor_id: {
            "x": current,
            "prev": previous,
            "weight": _scalar_or_row_weights(
                [float(weight_by_frame.get(frame, weight)) for frame in residual_frames],
                weight,
            ) if weight_by_frame is not None else float(weight),
            "scales": [float(scale) for scale in scales],
        }
    }


def build_sequence_static_freeze_residual_inputs(
    *,
    factor_id: str,
    states_by_frame: Mapping[int, Sequence[float]],
    activation_intervals: Sequence[Mapping[str, object]],
    scales: tuple[float, ...],
    weight: float,
    weight_by_frame: Mapping[int, float] | None = None,
) -> dict[str, dict[str, Any]]:
    """Reference every frame to its interaction-state interval anchor.

    Pairwise velocity penalties can accumulate slow drift. A static interval
    instead shares one anchor pose, while inactive intervals retain zero rows
    so residual ordering remains deterministic.
    """

    frames = sorted(int(frame) for frame in states_by_frame)
    if len(frames) <= 1:
        return {}
    interval_anchor: dict[int, int] = {}
    for interval in activation_intervals:
        start, end = int(interval["start_frame"]), int(interval["end_frame"])
        anchor = start
        for frame in range(start, end + 1):
            if frame in states_by_frame:
                interval_anchor[frame] = anchor
    current: list[list[float]] = []
    anchors: list[list[float]] = []
    residual_frames: list[int] = []
    for frame in frames[1:]:
        anchor = interval_anchor.get(frame)
        if anchor is None or anchor not in states_by_frame:
            raise ValueError(f"static-freeze factor {factor_id} has no interval anchor for frame {frame}")
        current.append(_numeric_vector(states_by_frame[frame], "static-freeze current state"))
        anchors.append(_numeric_vector(states_by_frame[anchor], "static-freeze anchor state"))
        residual_frames.append(frame)
    return {
        factor_id: {
            "x": current,
            "prev": anchors,
            "weight": _scalar_or_row_weights(
                [float(weight_by_frame.get(frame, weight)) for frame in residual_frames],
                weight,
            ) if weight_by_frame is not None else float(weight),
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
    object_feature_id: str | None,
    weight: float,
    sigma_m: float,
    weight_by_frame: Mapping[int, float] | None = None,
    residual_axes: tuple[int, ...] = (0, 1, 2),
) -> dict[str, dict[str, Any]]:
    """Resolve repeated world-space contact samples against object geometry."""

    anchors: list[list[float]] = []
    targets: list[list[float]] = []
    row_weights: list[float] = []
    sample_confidences: list[float] = []
    has_explicit_confidence = False
    for sample in samples:
        state = object_states.get(int(sample.frame))
        if state is None:
            continue
        source_xyz = [
            float(value) + float(offset)
            for value, offset in zip(sample.source_xyz_m, sample.source_offset_xyz_m)
        ]
        if len(source_xyz) != 3:
            raise ValueError("world-space source sites must have exactly three coordinates")
        feature_id = sample.object_feature_id or object_feature_id
        if feature_id is None:
            raise ValueError(
                f"world-space contact sample at frame {sample.frame} requires a target feature id"
            )
        if sample.line_s is None:
            target_xyz = geometry_provider.contact_point_world(state, feature_id, source_xyz)
        else:
            line_point_world = getattr(geometry_provider, "line_point_world", None)
            if line_point_world is None:
                raise ValueError("LineS contact requires line-parameter geometry capability")
            target_xyz = line_point_world(state, sample.line_s)
        target_xyz = [float(value) for value in target_xyz]
        for axis in {0, 1, 2} - set(residual_axes):
            source_xyz[axis] = target_xyz[axis]
        anchors.append(source_xyz)
        targets.append(target_xyz)
        frame_weight = float((weight_by_frame or {}).get(int(sample.frame), weight))
        row_weights.append(frame_weight * (sample.confidence if sample.confidence is not None else 1.0))
        sample_confidences.append(sample.confidence if sample.confidence is not None else 1.0)
        has_explicit_confidence = has_explicit_confidence or sample.confidence is not None
    if not anchors:
        return {}
    payload = {
        "anchors": anchors,
        "targets": targets,
        "weight": _scalar_or_row_weights(row_weights, weight),
        "sigma_m": float(sigma_m),
    }
    if has_explicit_confidence:
        payload["sample_confidence"] = sample_confidences
    return {factor_id: payload}


def _world_space_contact_velocity_pairs(
    samples: Iterable[WorldSpaceContactSample],
    object_states: Mapping[int, Sequence[float]],
) -> tuple[tuple[WorldSpaceContactSample, WorldSpaceContactSample], ...]:
    """Pair consecutive observations of the same semantic grasp edge."""

    selected: dict[tuple[str, str, float | None, int], WorldSpaceContactSample] = {}
    for sample in samples:
        feature_id = sample.object_feature_id or ""
        track_id = sample.contact_track_id or feature_id
        if not track_id or not feature_id or sample.frame not in object_states:
            continue
        key = (track_id, feature_id, sample.line_s, int(sample.frame))
        previous = selected.get(key)
        previous_confidence = -1.0 if previous is None or previous.confidence is None else previous.confidence
        confidence = 1.0 if sample.confidence is None else sample.confidence
        if previous is None or confidence > previous_confidence:
            selected[key] = sample
    grouped: dict[tuple[str, str, float | None], dict[int, WorldSpaceContactSample]] = {}
    for (track_id, feature_id, line_s, frame), sample in selected.items():
        grouped.setdefault((track_id, feature_id, line_s), {})[frame] = sample
    pairs: list[tuple[WorldSpaceContactSample, WorldSpaceContactSample]] = []
    for rows in grouped.values():
        for frame in sorted(rows):
            if frame - 1 in rows:
                pairs.append((rows[frame - 1], rows[frame]))
    return tuple(sorted(pairs, key=lambda pair: (pair[1].frame, pair[1].contact_track_id or "", pair[1].object_feature_id or "")))


def build_world_space_contact_relative_velocity_residual_inputs(
    *,
    factor_id: str,
    geometry_provider: GeometryProvider,
    object_states: Mapping[int, Sequence[float]],
    samples: Iterable[WorldSpaceContactSample],
    object_feature_id: str | None,
    weight: float,
    sigma_m_per_frame: float,
    weight_by_frame: Mapping[int, float] | None = None,
) -> dict[str, dict[str, Any]]:
    """Match object-anchor and read-only human-site displacement during a persistent grasp."""

    source_displacements: list[list[float]] = []
    target_displacements: list[list[float]] = []
    row_weights: list[float] = []
    for previous, current in _world_space_contact_velocity_pairs(samples, object_states):
        feature_id = current.object_feature_id or object_feature_id
        if feature_id is None:
            raise ValueError("contact relative velocity requires a target feature id")
        previous_source = np.asarray(previous.source_xyz_m, dtype=float) + np.asarray(previous.source_offset_xyz_m, dtype=float)
        current_source = np.asarray(current.source_xyz_m, dtype=float) + np.asarray(current.source_offset_xyz_m, dtype=float)
        if current.line_s is None:
            previous_target = np.asarray(
                geometry_provider.contact_point_world(object_states[previous.frame], feature_id, previous_source),
                dtype=float,
            )
            current_target = np.asarray(
                geometry_provider.contact_point_world(object_states[current.frame], feature_id, current_source),
                dtype=float,
            )
        else:
            line_point_world = getattr(geometry_provider, "line_point_world", None)
            if line_point_world is None:
                raise ValueError("LineS relative velocity requires line-parameter geometry capability")
            previous_target = np.asarray(line_point_world(object_states[previous.frame], previous.line_s), dtype=float)
            current_target = np.asarray(line_point_world(object_states[current.frame], current.line_s), dtype=float)
        source_displacements.append((current_source - previous_source).tolist())
        target_displacements.append((current_target - previous_target).tolist())
        confidence = min(
            1.0 if previous.confidence is None else previous.confidence,
            1.0 if current.confidence is None else current.confidence,
        )
        row_weights.append(float((weight_by_frame or {}).get(current.frame, weight)) * confidence)
    if not source_displacements:
        return {}
    return {
        factor_id: {
            "source_displacement_m": source_displacements,
            "target_displacement_m": target_displacements,
            "weight": _scalar_or_row_weights(row_weights, weight),
            "sigma_m_per_frame": float(sigma_m_per_frame),
        }
    }


def _quaternion_relative_rotvec_world(
    previous_state: Sequence[float],
    current_state: Sequence[float],
) -> np.ndarray:
    if len(previous_state) < 7 or len(current_state) < 7:
        raise ValueError("contact twist gauge requires a root quaternion at state indices 3:7")
    previous = np.asarray(previous_state[3:7], dtype=float)
    current = np.asarray(current_state[3:7], dtype=float)
    previous /= np.linalg.norm(previous)
    current /= np.linalg.norm(current)
    previous_inverse = previous * np.asarray((1.0, -1.0, -1.0, -1.0), dtype=float)
    aw, ax, ay, az = current
    bw, bx, by, bz = previous_inverse
    relative = np.asarray(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dtype=float,
    )
    relative /= np.linalg.norm(relative)
    if relative[0] < 0.0:
        relative *= -1.0
    vector_norm = float(np.linalg.norm(relative[1:]))
    if vector_norm <= 1e-12:
        return np.zeros(3, dtype=float)
    angle = 2.0 * atan2(vector_norm, float(relative[0]))
    return angle * relative[1:] / vector_norm


def _contact_chord_twist_rows(
    samples: Iterable[WorldSpaceContactSample],
    object_states: Mapping[int, Sequence[float]],
    geometry_provider: GeometryProvider,
    object_feature_id: str | None,
) -> tuple[tuple[int, float], ...]:
    anchors_by_frame: dict[int, dict[str, np.ndarray]] = {}
    for sample in samples:
        if sample.frame not in object_states:
            continue
        feature_id = sample.object_feature_id or object_feature_id
        track_id = sample.contact_track_id or feature_id
        if feature_id is None or track_id is None:
            continue
        source = np.asarray(sample.source_xyz_m, dtype=float) + np.asarray(sample.source_offset_xyz_m, dtype=float)
        if sample.line_s is None:
            target = geometry_provider.contact_point_world(object_states[sample.frame], feature_id, source)
        else:
            line_point_world = getattr(geometry_provider, "line_point_world", None)
            if line_point_world is None:
                raise ValueError("LineS twist gauge requires line-parameter geometry capability")
            target = line_point_world(object_states[sample.frame], sample.line_s)
        anchors_by_frame.setdefault(sample.frame, {})[track_id] = np.asarray(target, dtype=float)
    rows: list[tuple[int, float]] = []
    for frame in sorted(anchors_by_frame):
        if frame - 1 not in anchors_by_frame or frame - 1 not in object_states:
            continue
        anchors = tuple(anchors_by_frame[frame].values())
        if len(anchors) < 2:
            continue
        best_axis: np.ndarray | None = None
        best_length = 0.0
        for left_index, left in enumerate(anchors[:-1]):
            for right in anchors[left_index + 1 :]:
                axis = right - left
                length = float(np.linalg.norm(axis))
                if length > best_length:
                    best_axis, best_length = axis, length
        if best_axis is None or best_length <= 1e-6:
            continue
        axis_world = best_axis / best_length
        rotvec_world = _quaternion_relative_rotvec_world(object_states[frame - 1], object_states[frame])
        rows.append((frame, float(np.dot(rotvec_world, axis_world))))
    return tuple(rows)


def build_world_space_contact_twist_gauge_residual_inputs(
    *,
    factor_id: str,
    geometry_provider: GeometryProvider,
    object_states: Mapping[int, Sequence[float]],
    samples: Iterable[WorldSpaceContactSample],
    object_feature_id: str | None,
    weight: float,
    sigma_rad: float,
    weight_by_frame: Mapping[int, float] | None = None,
) -> dict[str, dict[str, Any]]:
    """Suppress only the unobservable spin around a persistent two-site contact chord."""

    rows = _contact_chord_twist_rows(samples, object_states, geometry_provider, object_feature_id)
    if not rows:
        return {}
    return {
        factor_id: {
            "twist_rad": [value for _frame, value in rows],
            "weight": _scalar_or_row_weights(
                [float((weight_by_frame or {}).get(frame, weight)) for frame, _value in rows],
                weight,
            ),
            "sigma_rad": float(sigma_rad),
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


def build_audio_event_alignment_residual_inputs(
    *,
    factor_id: str,
    events: Sequence[AudioEvent],
    timeline: InteractionTimeline,
    weight: float,
    sigma_s: float,
) -> dict[str, dict[str, Any]]:
    """Align typed audio peaks only with inferred interaction transitions."""

    events_by_id = {event.event_id: event for event in events}
    predicted: list[float] = []
    observed: list[float] = []
    for state in timeline.frames:
        is_transition = state.contact_state in {ContactStateAxis.ACTIVE, ContactStateAxis.RELEASE} or state.contact_mode in {
            InteractionContactMode.IMPACT,
            InteractionContactMode.RELEASE,
        }
        if not is_transition:
            continue
        for event_id in state.audio_event_ids:
            event = events_by_id.get(event_id)
            if event is None:
                continue
            predicted.append(float(state.time))
            observed.append(float(event.peak_time_s))
    if not predicted:
        return {}
    return {
        factor_id: {
            "predicted_event_time_s": predicted,
            "observed_event_time_s": observed,
            "weight": float(weight),
            "sigma_s": float(sigma_s),
        }
    }


def build_geometry_sequence_residual_input_bundle(
    residual_execution_plan: dict[str, object] | object,
    *,
    object_states: Mapping[int, Sequence[float]],
    state_scales: tuple[float, ...],
    reference_states: Mapping[int, Sequence[float]] | None = None,
    contact_factors: Mapping[str, ContactFactorInput] | None = None,
    contact_relative_velocity_factors: Mapping[str, ContactFactorInput] | None = None,
    contact_twist_gauge_factors: Mapping[str, ContactFactorInput] | None = None,
    pose_prior_factors: Mapping[str, PosePriorFactorInput] | None = None,
    periodic_phase_factors: Mapping[str, PeriodicPhaseFactorInput] | None = None,
    joint_limit_factors: Mapping[str, JointLimitFactorInput] | None = None,
    gauge_factors: Mapping[str, GaugeFactorInput] | None = None,
    audio_alignment_factors: Mapping[str, AudioAlignmentFactorInput] | None = None,
    line_reprojection_factors: Mapping[str, LineReprojectionFactorInput] | None = None,
    point_reprojection_factors: Mapping[str, PointReprojectionFactorInput] | None = None,
    mask_silhouette_factors: Mapping[str, MaskSilhouetteFactorInput] | None = None,
    metric_depth_factors: Mapping[str, MetricDepthFactorInput] | None = None,
    support_plane_factors: Mapping[str, SupportPlaneFactorInput] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build common sequence residuals from explicit state and factor inputs.

    Factor-specific data is keyed by factor id. Geometry family and object
    identity never select an executor or residual implementation here.
    """

    def temporal(request: ResidualInputRequest) -> dict[str, Any] | None:
        order = 1 if request.residual_fn_ref in {"shadow_residual::temporal_velocity", "shadow_residual::static_freeze"} else 2
        base_weight = _runtime_weight(request, 1.0)
        common = {
            "factor_id": request.factor_id,
            "states_by_frame": object_states,
            "scales": _runtime_state_scales(request, state_scales),
            "weight": base_weight,
            "weight_by_frame": _runtime_weights_by_frame(request, object_states, base_weight),
        }
        if request.residual_fn_ref == "shadow_residual::static_freeze":
            payload = build_sequence_static_freeze_residual_inputs(
                **common,
                activation_intervals=request.activation_intervals,
            )
        else:
            payload = build_sequence_temporal_residual_inputs(**common, order=order)
        return payload.get(request.factor_id)

    def regularization(request: ResidualInputRequest) -> dict[str, Any] | None:
        if reference_states is None:
            return None
        frames = sorted(set(object_states) & set(reference_states))
        base_weight = _runtime_weight(request, 1.0)
        weights_by_frame = _runtime_weights_by_frame(request, frames, base_weight)
        payload = build_state_regularization_residual_inputs(
            factor_id=request.factor_id,
            values=[object_states[frame] for frame in frames],
            target=[reference_states[frame] for frame in frames],
            scales=_runtime_state_scales(request, state_scales),
            weight=(
                _scalar_or_row_weights([weights_by_frame[frame] for frame in frames], base_weight)
                if weights_by_frame is not None
                else base_weight
            ),
        )
        return payload.get(request.factor_id)

    def contact(request: ResidualInputRequest) -> dict[str, Any] | None:
        factor = (contact_factors or {}).get(request.factor_id)
        if factor is None:
            return None
        base_weight = _runtime_weight(request, factor.weight)
        payload = build_world_space_contact_sample_residual_inputs(
            factor_id=request.factor_id,
            geometry_provider=factor.geometry_provider,
            object_states=object_states,
            samples=factor.samples,
            object_feature_id=factor.object_feature_id,
            weight=base_weight,
            sigma_m=_runtime_sigma(request, factor.sigma_m, "m"),
            weight_by_frame=_runtime_weights_by_frame(
                request,
                (sample.frame for sample in factor.samples),
                base_weight,
            ),
            residual_axes=factor.residual_axes,
        )
        return payload.get(request.factor_id)

    def contact_relative_velocity(request: ResidualInputRequest) -> dict[str, Any] | None:
        factor = (contact_relative_velocity_factors or {}).get(request.factor_id)
        if factor is None:
            return None
        base_weight = _runtime_weight(request, factor.weight)
        payload = build_world_space_contact_relative_velocity_residual_inputs(
            factor_id=request.factor_id,
            geometry_provider=factor.geometry_provider,
            object_states=object_states,
            samples=factor.samples,
            object_feature_id=factor.object_feature_id,
            weight=base_weight,
            sigma_m_per_frame=_runtime_sigma(request, factor.sigma_m, "m/frame"),
            weight_by_frame=_runtime_weights_by_frame(
                request,
                (sample.frame for sample in factor.samples),
                base_weight,
            ),
        )
        return payload.get(request.factor_id)

    def contact_twist_gauge(request: ResidualInputRequest) -> dict[str, Any] | None:
        factor = (contact_twist_gauge_factors or {}).get(request.factor_id)
        if factor is None:
            return None
        base_weight = _runtime_weight(request, factor.weight)
        payload = build_world_space_contact_twist_gauge_residual_inputs(
            factor_id=request.factor_id,
            geometry_provider=factor.geometry_provider,
            object_states=object_states,
            samples=factor.samples,
            object_feature_id=factor.object_feature_id,
            weight=base_weight,
            sigma_rad=_runtime_sigma(request, factor.sigma_m, "rad"),
            weight_by_frame=_runtime_weights_by_frame(
                request,
                (sample.frame for sample in factor.samples),
                base_weight,
            ),
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
        values = factor.values
        target = factor.target
        if factor.state_index is not None:
            if factor.target_by_frame is None:
                raise ValueError("state-backed periodic phase factor requires frame targets")
            frames = sorted(set(object_states) & {int(frame) for frame in factor.target_by_frame})
            values = tuple(float(object_states[frame][factor.state_index]) for frame in frames)
            target = tuple(float(factor.target_by_frame[frame]) for frame in frames)
        payload = build_periodic_phase_prior_residual_inputs(
            factor_id=request.factor_id,
            values=values,
            target=target,
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

    def audio_alignment(request: ResidualInputRequest) -> dict[str, Any] | None:
        factor = (audio_alignment_factors or {}).get(request.factor_id)
        if factor is None:
            return None
        payload = build_audio_event_alignment_residual_inputs(
            factor_id=request.factor_id,
            events=factor.events,
            timeline=factor.timeline,
            weight=factor.weight,
            sigma_s=factor.sigma_s,
        )
        return payload.get(request.factor_id)

    def line_reprojection(request: ResidualInputRequest) -> dict[str, Any] | None:
        factor = (line_reprojection_factors or {}).get(request.factor_id)
        if factor is None:
            return None
        base_weight = _runtime_weight(request, factor.weight)
        payload = build_line_reprojection_residual_inputs(
            factor_id=request.factor_id,
            geometry_provider=factor.geometry_provider,
            object_states=object_states,
            measurements=factor.measurements,
            cameras_by_frame=factor.cameras_by_frame,
            weight=base_weight,
            sigma_px=_runtime_sigma(request, factor.sigma_px, "px"),
            allow_endpoint_swap=factor.allow_endpoint_swap,
            constraint_mode=factor.constraint_mode,
            weight_by_frame=_runtime_weights_by_frame(
                request,
                (measurement.meta.frame for measurement in factor.measurements),
                base_weight,
            ),
        )
        return payload.get(request.factor_id)

    def point_reprojection(request: ResidualInputRequest) -> dict[str, Any] | None:
        factor = (point_reprojection_factors or {}).get(request.factor_id)
        if factor is None:
            return None
        base_weight = _runtime_weight(request, factor.weight)
        payload = build_point_reprojection_residual_inputs(
            factor_id=request.factor_id, geometry_provider=factor.geometry_provider,
            object_states=object_states, measurements=factor.measurements,
            cameras_by_frame=factor.cameras_by_frame, weight=base_weight,
            sigma_px=_runtime_sigma(request, factor.sigma_px, "px"),
            weight_by_frame=_runtime_weights_by_frame(request, (m.meta.frame for m in factor.measurements), base_weight),
        )
        return payload.get(request.factor_id)

    def metric_depth(request: ResidualInputRequest) -> dict[str, Any] | None:
        factor = (metric_depth_factors or {}).get(request.factor_id)
        if factor is None:
            return None
        base_weight = _runtime_weight(request, factor.weight)
        payload = build_metric_depth_measurement_residual_inputs(
            factor_id=request.factor_id, object_states=object_states,
            measurements=factor.measurements, state_index=factor.state_index,
            weight=base_weight, sigma_m=_runtime_sigma(request, factor.sigma_m, "m"),
            weight_by_frame=_runtime_weights_by_frame(request, (m.meta.frame for m in factor.measurements), base_weight),
            target_by_frame=factor.target_by_frame,
        )
        return payload.get(request.factor_id)

    def mask_silhouette(request: ResidualInputRequest) -> dict[str, Any] | None:
        factor = (mask_silhouette_factors or {}).get(request.factor_id)
        if factor is None:
            return None
        base_weight = _runtime_weight(request, factor.weight)
        payload = build_mask_silhouette_residual_inputs(
            factor_id=request.factor_id,
            geometry_provider=factor.geometry_provider,
            object_states=object_states,
            measurements=factor.measurements,
            cameras_by_frame=factor.cameras_by_frame,
            weight=base_weight,
            sigma_px=_runtime_sigma(request, factor.sigma_px, "px"),
            principal_axis_sigma_rad=factor.principal_axis_sigma_rad,
            weight_by_frame=_runtime_weights_by_frame(request, (m.meta.frame for m in factor.measurements), base_weight),
        )
        return payload.get(request.factor_id)

    def support_plane(request: ResidualInputRequest) -> dict[str, Any] | None:
        factor = (support_plane_factors or {}).get(request.factor_id)
        if factor is None:
            return None
        payload = build_support_plane_residual_inputs(
            factor_id=request.factor_id,
            geometry_provider=factor.geometry_provider,
            object_states=object_states,
            support_feature_ids=factor.support_feature_ids,
            active_frames=factor.active_frames,
            plane=factor.plane,
            support_weight=factor.support_weight,
            penetration_weight=factor.penetration_weight,
            sigma_m=factor.sigma_m,
            activation_status_by_frame=factor.activation_status_by_frame,
            activation_weight_by_frame=factor.activation_weight_by_frame,
            proximity_gate_m=factor.proximity_gate_m,
            tangent_gauge_weight=factor.tangent_gauge_weight,
            tangent_gauge_sigma_rad=factor.tangent_gauge_sigma_rad,
        )
        return payload.get(request.factor_id)

    return build_residual_input_bundle(
        residual_execution_plan,
        {
            "shadow_residual::temporal_velocity": temporal,
            "shadow_residual::temporal_acceleration": temporal,
            "shadow_residual::static_freeze": temporal,
            "shadow_residual::regularization": regularization,
            "shadow_residual::contact_distance": contact,
            "shadow_residual::contact_relative_velocity": contact_relative_velocity,
            "shadow_residual::contact_twist_gauge": contact_twist_gauge,
            "shadow_residual::pose_prior": pose_prior,
            "shadow_residual::periodic_phase_prior": periodic_phase,
            "shadow_residual::joint_limit": joint_limit,
            "shadow_residual::gauge_constraint": gauge,
            "shadow_residual::audio_event_prior": audio_alignment,
            "shadow_residual::line_reprojection": line_reprojection,
            "shadow_residual::point_reprojection": point_reprojection,
            "shadow_residual::mask_silhouette": mask_silhouette,
            "shadow_residual::metric_depth": metric_depth,
            "shadow_residual::support_and_penetration": support_plane,
        },
    )


def build_geometry_sequence_residual_dependencies(
    residual_execution_plan: dict[str, object] | object,
    *,
    object_states: Mapping[int, Sequence[float]],
    factor_ids: Sequence[str] | None = None,
    reference_states: Mapping[int, Sequence[float]] | None = None,
    contact_factors: Mapping[str, ContactFactorInput] | None = None,
    contact_relative_velocity_factors: Mapping[str, ContactFactorInput] | None = None,
    contact_twist_gauge_factors: Mapping[str, ContactFactorInput] | None = None,
    periodic_phase_factors: Mapping[str, PeriodicPhaseFactorInput] | None = None,
    line_reprojection_factors: Mapping[str, LineReprojectionFactorInput] | None = None,
    point_reprojection_factors: Mapping[str, PointReprojectionFactorInput] | None = None,
    mask_silhouette_factors: Mapping[str, MaskSilhouetteFactorInput] | None = None,
    metric_depth_factors: Mapping[str, MetricDepthFactorInput] | None = None,
    support_plane_factors: Mapping[str, SupportPlaneFactorInput] | None = None,
) -> tuple[ResidualRowDependency, ...]:
    """Compile typed runtime input ordering into row-level state dependencies.

    This mirrors ``build_geometry_sequence_residual_input_bundle`` for factors
    whose residual values actually change with object state. Factors backed by
    static audit values intentionally receive no declaration and remain dense
    at the optimizer boundary rather than claiming a false sparse dependency.
    """

    if not object_states:
        raise ValueError("residual dependencies require object states")
    frames = tuple(sorted(int(frame) for frame in object_states))
    state_widths = {len(object_states[frame]) for frame in frames}
    if len(state_widths) != 1 or not next(iter(state_widths)):
        raise ValueError("residual dependencies require one nonzero object-state width")
    state_width = next(iter(state_widths))
    selected = set(factor_ids) if factor_ids is not None else None
    if isinstance(residual_execution_plan, dict):
        records = [record for record in residual_execution_plan.get("records", []) if isinstance(record, dict)]
    else:
        records = [
            {
                "factor_id": record.factor_id,
                "residual_fn_ref": record.residual_fn_ref,
                "status": record.status,
            }
            for record in getattr(residual_execution_plan, "records", ())
        ]

    dependencies: list[ResidualRowDependency] = []
    for record in records:
        if str(record.get("status", "ready_not_executed")) != "ready_not_executed":
            continue
        factor_id = str(record.get("factor_id", ""))
        if selected is not None and factor_id not in selected:
            continue
        residual_ref = str(record.get("residual_fn_ref", ""))
        if residual_ref in {"shadow_residual::temporal_velocity", "shadow_residual::temporal_acceleration", "shadow_residual::static_freeze"}:
            if residual_ref == "shadow_residual::static_freeze":
                intervals = tuple(
                    item
                    for item in record.get("activation_intervals", ())
                    if isinstance(item, Mapping)
                )
                for residual_index, frame in enumerate(frames[1:]):
                    interval = next(
                        interval
                        for interval in intervals
                        if int(interval["start_frame"]) <= frame <= int(interval["end_frame"])
                    )
                    start = int(interval["start_frame"])
                    anchor = start
                    dependency_frames = (frame,) if anchor == frame else (anchor, frame)
                    dependencies.append(
                        ResidualRowDependency(
                            factor_id,
                            residual_index * state_width,
                            (residual_index + 1) * state_width,
                            dependency_frames,
                        )
                    )
                continue
            order = 1 if residual_ref in {"shadow_residual::temporal_velocity", "shadow_residual::static_freeze"} else 2
            for residual_index, frame_index in enumerate(range(order, len(frames))):
                dependency_frames = frames[frame_index - order : frame_index + 1]
                dependencies.append(
                    ResidualRowDependency(
                        factor_id,
                        residual_index * state_width,
                        (residual_index + 1) * state_width,
                        dependency_frames,
                    )
                )
            continue
        if residual_ref == "shadow_residual::regularization" and reference_states is not None:
            active_frames = tuple(sorted(set(frames) & {int(frame) for frame in reference_states}))
            for residual_index, frame in enumerate(active_frames):
                dependencies.append(
                    ResidualRowDependency(
                        factor_id,
                        residual_index * state_width,
                        (residual_index + 1) * state_width,
                        (frame,),
                    )
                )
            continue
        if residual_ref == "shadow_residual::contact_distance":
            factor = (contact_factors or {}).get(factor_id)
            if factor is None:
                continue
            residual_index = 0
            for sample in factor.samples:
                frame = int(sample.frame)
                if frame not in object_states:
                    continue
                dependencies.append(
                    ResidualRowDependency(factor_id, 3 * residual_index, 3 * (residual_index + 1), (frame,))
                )
                residual_index += 1
            continue
        if residual_ref == "shadow_residual::contact_relative_velocity":
            factor = (contact_relative_velocity_factors or {}).get(factor_id)
            if factor is None:
                continue
            for residual_index, (previous, current) in enumerate(
                _world_space_contact_velocity_pairs(factor.samples, object_states)
            ):
                dependencies.append(
                    ResidualRowDependency(
                        factor_id,
                        3 * residual_index,
                        3 * (residual_index + 1),
                        (previous.frame, current.frame),
                    )
                )
            continue
        if residual_ref == "shadow_residual::contact_twist_gauge":
            factor = (contact_twist_gauge_factors or {}).get(factor_id)
            if factor is None:
                continue
            rows = _contact_chord_twist_rows(
                factor.samples,
                object_states,
                factor.geometry_provider,
                factor.object_feature_id,
            )
            for residual_index, (frame, _value) in enumerate(rows):
                dependencies.append(
                    ResidualRowDependency(
                        factor_id,
                        residual_index,
                        residual_index + 1,
                        (frame - 1, frame),
                    )
                )
            continue
        if residual_ref == "shadow_residual::periodic_phase_prior":
            factor = (periodic_phase_factors or {}).get(factor_id)
            if factor is None or factor.state_index is None or factor.target_by_frame is None:
                continue
            residual_index = 0
            for frame in sorted(set(frames) & {int(value) for value in factor.target_by_frame}):
                dependencies.append(
                    ResidualRowDependency(factor_id, residual_index, residual_index + 1, (frame,))
                )
                residual_index += 1
            continue
        if residual_ref == "shadow_residual::line_reprojection":
            factor = (line_reprojection_factors or {}).get(factor_id)
            if factor is None:
                continue
            residual_index = 0
            for measurement in factor.measurements:
                frame = int(measurement.meta.frame)
                if frame not in object_states or frame not in factor.cameras_by_frame:
                    continue
                row_width = 2 if factor.constraint_mode == "axis_line" else 4
                dependencies.append(
                    ResidualRowDependency(
                        factor_id,
                        row_width * residual_index,
                        row_width * (residual_index + 1),
                        (frame,),
                    )
                )
                residual_index += 1
            continue
        if residual_ref == "shadow_residual::point_reprojection":
            factor = (point_reprojection_factors or {}).get(factor_id)
            if factor is None:
                continue
            residual_index = 0
            for measurement in factor.measurements:
                frame = int(measurement.meta.frame)
                if frame in object_states and frame in factor.cameras_by_frame:
                    dependencies.append(ResidualRowDependency(factor_id, 2 * residual_index, 2 * (residual_index + 1), (frame,)))
                    residual_index += 1
            continue
        if residual_ref == "shadow_residual::mask_silhouette":
            factor = (mask_silhouette_factors or {}).get(factor_id)
            if factor is None:
                continue
            residual_index = 0
            row_width = 5 if factor.principal_axis_sigma_rad is not None else 4
            for measurement in factor.measurements:
                frame = int(measurement.meta.frame)
                if frame in object_states and frame in factor.cameras_by_frame:
                    dependencies.append(ResidualRowDependency(factor_id, row_width * residual_index, row_width * (residual_index + 1), (frame,)))
                    residual_index += 1
            continue
        if residual_ref == "shadow_residual::metric_depth":
            factor = (metric_depth_factors or {}).get(factor_id)
            if factor is None:
                continue
            residual_index = 0
            for measurement in factor.measurements:
                frame = int(measurement.meta.frame)
                if frame in object_states:
                    dependencies.append(ResidualRowDependency(factor_id, residual_index, residual_index + 1, (frame,)))
                    residual_index += 1
            continue
        if residual_ref == "shadow_residual::support_and_penetration":
            factor = (support_plane_factors or {}).get(factor_id)
            if factor is None:
                continue
            point_frames: list[int] = []
            for frame in sorted(set(int(value) for value in factor.active_frames)):
                state = object_states.get(frame)
                if state is None:
                    continue
                for feature_id in factor.support_feature_ids:
                    points = factor.geometry_provider.feature_points_world(state, feature_id)
                    point_frames.extend([frame] * len(points))
            point_count = len(point_frames)
            for residual_index, frame in enumerate(point_frames):
                dependencies.append(ResidualRowDependency(factor_id, residual_index, residual_index + 1, (frame,)))
                dependencies.append(
                    ResidualRowDependency(
                        factor_id,
                        point_count + residual_index,
                        point_count + residual_index + 1,
                        (frame,),
                    )
                )
            if factor.tangent_gauge_weight > 0.0:
                tangent_frames = tuple(
                    frame
                    for frame in sorted(set(int(value) for value in factor.active_frames))
                    if (factor.activation_status_by_frame or {}).get(frame) == "active"
                    and frame - 1 in object_states
                    and frame in object_states
                )
                tangent_offset = 2 * point_count
                for residual_index, frame in enumerate(tangent_frames):
                    dependencies.append(
                        ResidualRowDependency(
                            factor_id,
                            tangent_offset + residual_index,
                            tangent_offset + residual_index + 1,
                            (frame - 1, frame),
                        )
                    )
    return tuple(dependencies)
