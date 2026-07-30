from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _row_weights(weight: float | np.ndarray, row_count: int) -> np.ndarray:
    values = np.asarray(weight, dtype=float)
    if values.ndim == 0:
        scalar = float(values)
        if not np.isfinite(scalar) or scalar < 0.0:
            raise ValueError("factor weight must be finite and non-negative")
        return np.full(row_count, scalar, dtype=float)
    flattened = values.reshape(-1)
    if len(flattened) != row_count:
        raise ValueError(f"factor row weights must have length {row_count}, got {len(flattened)}")
    if not np.isfinite(flattened).all() or np.any(flattened < 0.0):
        raise ValueError("factor row weights must be finite and non-negative")
    return flattened


@dataclass(frozen=True)
class FactorResidualEvaluator:
    """Shared residual block assembly for isolated factor executors.

    This keeps the arithmetic used by existing solvers but moves weighted
    residual construction out of case-specific optimization loops.
    """

    def point_reprojection(self, predicted: np.ndarray, target: np.ndarray, *, weight: float | np.ndarray, sigma_px: float) -> np.ndarray:
        if predicted.shape != target.shape or predicted.ndim != 2 or predicted.shape[1] != 2:
            raise ValueError("point reprojection residuals require matching (N, 2) arrays")
        if np.asarray(weight).ndim == 0:
            return (float(weight) * (predicted - target).reshape(-1) / float(sigma_px)).astype(float)
        return (_row_weights(weight, len(predicted))[:, None] * (predicted - target) / float(sigma_px)).reshape(-1).astype(float)

    def line_reprojection(
        self,
        predicted: np.ndarray,
        target: np.ndarray,
        *,
        weight: float | np.ndarray,
        sigma_px: float,
        allow_endpoint_swap: bool,
        constraint_mode: str = "endpoints",
    ) -> np.ndarray:
        if predicted.shape != target.shape or predicted.ndim != 3 or predicted.shape[1:] != (2, 2):
            raise ValueError("line reprojection residuals require matching (N, 2, 2) endpoint arrays")
        if constraint_mode == "axis_line":
            direction = target[:, 1, :] - target[:, 0, :]
            lengths = np.linalg.norm(direction, axis=1)
            if np.any(lengths <= 1e-9):
                raise ValueError("axis-line reprojection requires nondegenerate target lines")
            offsets = predicted - target[:, :1, :]
            signed_distance = (
                direction[:, None, 0] * offsets[:, :, 1]
                - direction[:, None, 1] * offsets[:, :, 0]
            ) / lengths[:, None]
            if np.asarray(weight).ndim == 0:
                return (float(weight) * signed_distance.reshape(-1) / float(sigma_px)).astype(float)
            return (
                _row_weights(weight, len(predicted))[:, None]
                * signed_distance
                / float(sigma_px)
            ).reshape(-1).astype(float)
        if constraint_mode != "endpoints":
            raise ValueError("unsupported line reprojection constraint mode")
        aligned = predicted.copy()
        if allow_endpoint_swap:
            direct_cost = np.sum((predicted - target) ** 2, axis=(1, 2))
            swapped = predicted[:, ::-1, :]
            swap_cost = np.sum((swapped - target) ** 2, axis=(1, 2))
            swap_mask = swap_cost < direct_cost
            aligned[swap_mask] = swapped[swap_mask]
        if np.asarray(weight).ndim == 0:
            return (float(weight) * (aligned - target).reshape(-1) / float(sigma_px)).astype(float)
        return (
            _row_weights(weight, len(aligned))[:, None, None]
            * (aligned - target)
            / float(sigma_px)
        ).reshape(-1).astype(float)

    def contact_distance(
        self,
        anchors: np.ndarray,
        targets: np.ndarray,
        *,
        weight: float | np.ndarray,
        sigma_m: float,
        sample_confidence: np.ndarray | None = None,
    ) -> np.ndarray:
        if anchors.shape != targets.shape or anchors.ndim != 2 or anchors.shape[1] != 3:
            raise ValueError("contact distance residuals require matching (N, 3) arrays")
        if sample_confidence is not None and np.asarray(sample_confidence).reshape(-1).shape != (len(anchors),):
            raise ValueError("contact sample confidence must match contact rows")
        if np.asarray(weight).ndim == 0:
            return (float(weight) * (anchors - targets).reshape(-1) / float(sigma_m)).astype(float)
        return (
            _row_weights(weight, len(anchors))[:, None]
            * (anchors - targets)
            / float(sigma_m)
        ).reshape(-1).astype(float)

    def contact_relative_velocity(
        self,
        source_displacement_m: np.ndarray,
        target_displacement_m: np.ndarray,
        *,
        weight: float | np.ndarray,
        sigma_m_per_frame: float,
    ) -> np.ndarray:
        if (
            source_displacement_m.shape != target_displacement_m.shape
            or source_displacement_m.ndim != 2
            or source_displacement_m.shape[1] != 3
        ):
            raise ValueError("contact relative velocity requires matching (N, 3) displacement arrays")
        delta = target_displacement_m - source_displacement_m
        if np.asarray(weight).ndim == 0:
            return (float(weight) * delta.reshape(-1) / float(sigma_m_per_frame)).astype(float)
        return (
            _row_weights(weight, len(delta))[:, None]
            * delta
            / float(sigma_m_per_frame)
        ).reshape(-1).astype(float)

    def contact_twist_gauge(
        self,
        twist_rad: np.ndarray,
        *,
        weight: float | np.ndarray,
        sigma_rad: float,
    ) -> np.ndarray:
        values = np.asarray(twist_rad, dtype=float).reshape(-1)
        return (_row_weights(weight, len(values)) * values / float(sigma_rad)).astype(float)

    def metric_depth(self, predicted_depth_m: np.ndarray, target_depth_m: np.ndarray, *, weight: float | np.ndarray, sigma_m: float) -> np.ndarray:
        if predicted_depth_m.shape != target_depth_m.shape:
            raise ValueError("metric depth residuals require matching depth arrays")
        delta = predicted_depth_m.reshape(-1) - target_depth_m.reshape(-1)
        return (_row_weights(weight, len(delta)) * delta / float(sigma_m)).astype(float)

    def support_penetration(self, signed_distance_m: np.ndarray, *, weight: float, sigma_m: float) -> np.ndarray:
        penetration = np.minimum(signed_distance_m.reshape(-1), 0.0)
        return (float(weight) * penetration / float(sigma_m)).astype(float)

    def support_plane(
        self,
        signed_distance_m: np.ndarray,
        *,
        support_weight: float | np.ndarray,
        penetration_weight: float | np.ndarray,
        sigma_m: float,
        tangent_twist_rad: np.ndarray | None = None,
        tangent_weight: float | np.ndarray = 0.0,
        tangent_sigma_rad: float = 1.0,
    ) -> np.ndarray:
        distances = signed_distance_m.reshape(-1)
        support = _row_weights(support_weight, len(distances)) * distances / float(sigma_m)
        penetration = (
            _row_weights(penetration_weight, len(distances))
            * np.minimum(distances, 0.0)
            / float(sigma_m)
        )
        blocks = [support, penetration]
        if tangent_twist_rad is not None:
            tangent = np.asarray(tangent_twist_rad, dtype=float).reshape(-1)
            blocks.append(
                _row_weights(tangent_weight, len(tangent))
                * tangent
                / float(tangent_sigma_rad)
            )
        return np.concatenate(blocks).astype(float)

    def pose_prior(
        self,
        x: np.ndarray,
        ref: np.ndarray,
        init: np.ndarray,
        *,
        rot_bound: float,
        xy_bound: float,
        z_bound: float,
        w_prior_rot: float,
        w_prior_xy: float,
        w_prior_z: float,
    ) -> np.ndarray:
        return np.concatenate(
            [
                float(w_prior_rot) * (x[:3] - ref[:3]) / float(rot_bound),
                float(w_prior_xy) * (x[3:5] - ref[3:5]) / float(xy_bound),
                float(w_prior_z) * (x[5:6] - init[5:6]) / float(z_bound),
            ]
        ).astype(float)

    def temporal_delta(self, x: np.ndarray, prev: np.ndarray, *, weight: float | np.ndarray, scales: np.ndarray) -> np.ndarray:
        if x.shape != prev.shape or x.ndim not in {1, 2} or x.shape[-1] != len(scales):
            raise ValueError("temporal residuals require matching state arrays with scale-aligned width")
        delta = (x - prev) / scales
        if np.asarray(weight).ndim == 0:
            return (float(weight) * (x - prev) / scales).astype(float)
        if x.ndim == 1:
            return (_row_weights(weight, 1)[0] * delta).astype(float)
        return (_row_weights(weight, len(x))[:, None] * delta).astype(float)

    def joint_limit(
        self,
        values: np.ndarray,
        *,
        lower: float | None,
        upper: float | None,
        weight: float,
        sigma_rad: float,
    ) -> np.ndarray:
        lower_violation = np.zeros_like(values, dtype=float) if lower is None else np.minimum(values - float(lower), 0.0)
        upper_violation = np.zeros_like(values, dtype=float) if upper is None else np.maximum(values - float(upper), 0.0)
        return (float(weight) * (lower_violation + upper_violation) / float(sigma_rad)).astype(float)

    def gauge_constraint(self, values: np.ndarray, *, target: float, weight: float, sigma: float) -> np.ndarray:
        return (float(weight) * (values - float(target)) / float(sigma)).astype(float)

    def regularization(self, values: np.ndarray, target: np.ndarray, *, weight: float | np.ndarray, scales: np.ndarray) -> np.ndarray:
        if values.shape != target.shape or values.shape != scales.shape:
            raise ValueError("regularization residuals require matching value, target, and scale arrays")
        delta = (values - target) / scales
        if np.asarray(weight).ndim == 0:
            return (float(weight) * (values - target) / scales).astype(float)
        if values.ndim == 1:
            return (_row_weights(weight, 1)[0] * delta).astype(float)
        return (_row_weights(weight, len(values))[:, None] * delta).astype(float)

    def periodic_phase_prior(self, values: np.ndarray, target: np.ndarray, *, weight: float, sigma_rad: float) -> np.ndarray:
        if values.shape != target.shape:
            raise ValueError("periodic phase residuals require matching phase arrays")
        wrapped = (values - target + np.pi) % (2.0 * np.pi) - np.pi
        return (float(weight) * wrapped.reshape(-1) / float(sigma_rad)).astype(float)

    def audio_event_prior(
        self,
        predicted_event_time_s: np.ndarray,
        observed_event_time_s: np.ndarray,
        *,
        weight: float,
        sigma_s: float,
    ) -> np.ndarray:
        if predicted_event_time_s.shape != observed_event_time_s.shape:
            raise ValueError("audio event residuals require matching predicted and observed event time arrays")
        return (
            float(weight)
            * (predicted_event_time_s.reshape(-1) - observed_event_time_s.reshape(-1))
            / float(sigma_s)
        ).astype(float)
