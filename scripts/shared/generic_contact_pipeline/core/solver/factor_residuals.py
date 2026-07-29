from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FactorResidualEvaluator:
    """Shared residual block assembly for isolated factor executors.

    This keeps the arithmetic used by existing solvers but moves weighted
    residual construction out of case-specific optimization loops.
    """

    def point_reprojection(self, predicted: np.ndarray, target: np.ndarray, *, weight: float, sigma_px: float) -> np.ndarray:
        if predicted.shape != target.shape or predicted.ndim != 2 or predicted.shape[1] != 2:
            raise ValueError("point reprojection residuals require matching (N, 2) arrays")
        return (float(weight) * (predicted - target).reshape(-1) / float(sigma_px)).astype(float)

    def contact_distance(self, anchors: np.ndarray, targets: np.ndarray, *, weight: float, sigma_m: float) -> np.ndarray:
        if anchors.shape != targets.shape or anchors.ndim != 2 or anchors.shape[1] != 3:
            raise ValueError("contact distance residuals require matching (N, 3) arrays")
        return (float(weight) * (anchors - targets).reshape(-1) / float(sigma_m)).astype(float)

    def metric_depth(self, predicted_depth_m: np.ndarray, target_depth_m: np.ndarray, *, weight: float, sigma_m: float) -> np.ndarray:
        if predicted_depth_m.shape != target_depth_m.shape:
            raise ValueError("metric depth residuals require matching depth arrays")
        return (float(weight) * (predicted_depth_m.reshape(-1) - target_depth_m.reshape(-1)) / float(sigma_m)).astype(float)

    def support_penetration(self, signed_distance_m: np.ndarray, *, weight: float, sigma_m: float) -> np.ndarray:
        penetration = np.minimum(signed_distance_m.reshape(-1), 0.0)
        return (float(weight) * penetration / float(sigma_m)).astype(float)

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

    def temporal_delta(self, x: np.ndarray, prev: np.ndarray, *, weight: float, scales: np.ndarray) -> np.ndarray:
        if x.shape != prev.shape or x.ndim not in {1, 2} or x.shape[-1] != len(scales):
            raise ValueError("temporal residuals require matching state arrays with scale-aligned width")
        return (float(weight) * (x - prev) / scales).astype(float)

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

    def regularization(self, values: np.ndarray, target: np.ndarray, *, weight: float, scales: np.ndarray) -> np.ndarray:
        if values.shape != target.shape or values.shape != scales.shape:
            raise ValueError("regularization residuals require matching value, target, and scale arrays")
        return (float(weight) * (values - target) / scales).astype(float)

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
