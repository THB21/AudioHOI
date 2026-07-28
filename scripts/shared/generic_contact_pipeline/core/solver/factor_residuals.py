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
        return (float(weight) * (x[: len(scales)] - prev[: len(scales)]) / scales).astype(float)

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
