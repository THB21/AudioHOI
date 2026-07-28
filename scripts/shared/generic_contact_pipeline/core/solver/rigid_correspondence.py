"""Case-agnostic rigid initializers from geometric correspondences."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


def _rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-12:
        return np.eye(3)
    axis = rotvec / theta
    kx, ky, kz = axis
    skew = np.array([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]], dtype=float)
    return np.eye(3) + math.sin(theta) * skew + (1.0 - math.cos(theta)) * (skew @ skew)


def _matrix_to_rotvec(matrix: np.ndarray) -> np.ndarray:
    trace = float(np.trace(matrix))
    cos_theta = float(np.clip((trace - 1.0) * 0.5, -1.0, 1.0))
    theta = math.acos(cos_theta)
    if theta < 1e-12:
        return np.zeros(3, dtype=float)
    if abs(math.pi - theta) < 1e-7:
        eigvals, eigvecs = np.linalg.eig(matrix)
        axis = np.real(eigvecs[:, int(np.argmin(np.abs(eigvals - 1.0)))])
        axis /= np.linalg.norm(axis)
        return axis * theta
    axis = np.array(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ],
        dtype=float,
    )
    axis /= 2.0 * math.sin(theta)
    return axis * theta


@dataclass(frozen=True)
class RigidCorrespondenceInitializer:
    """Initialize a rigid pose from explicit point correspondences.

    The two-point path fixes translation and two rotational degrees of freedom
    by aligning the local chord to the target chord with the shortest rotation
    from the provided initial orientation. It consumes no case-specific solved
    pose, object baseline, or historical seed.
    """

    base_local_to_cam: np.ndarray

    def align_two_points(
        self,
        init: np.ndarray,
        local: np.ndarray,
        target: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, float | int | bool]]:
        if local.shape != (2, 3) or target.shape != (2, 3):
            raise ValueError("two point rigid correspondence requires local and target shape (2, 3)")
        if not np.all(np.isfinite(local)) or not np.all(np.isfinite(target)):
            return init.copy(), {"used": False, "case_specific_state_used": False, "correspondence_count": 2}

        r_init = _rotvec_to_matrix(init[:3]) @ self.base_local_to_cam
        source_chord = r_init @ (local[1] - local[0])
        target_chord = target[1] - target[0]
        source_length = float(np.linalg.norm(source_chord))
        target_length = float(np.linalg.norm(target_chord))
        if source_length < 1e-8 or target_length < 1e-8:
            return init.copy(), {"used": False, "case_specific_state_used": False, "correspondence_count": 2}

        source_unit = source_chord / source_length
        target_unit = target_chord / target_length
        cross = np.cross(source_unit, target_unit)
        sin_angle = float(np.linalg.norm(cross))
        cos_angle = float(np.clip(np.dot(source_unit, target_unit), -1.0, 1.0))
        if sin_angle > 1e-8:
            delta = _rotvec_to_matrix(cross / sin_angle * math.atan2(sin_angle, cos_angle))
        elif cos_angle >= 0.0:
            delta = np.eye(3)
        else:
            basis = np.eye(3)[int(np.argmin(np.abs(source_unit)))]
            axis = np.cross(source_unit, basis)
            axis /= np.linalg.norm(axis)
            delta = _rotvec_to_matrix(axis * math.pi)

        r_seed = delta @ r_init
        t_seed = np.mean(target, axis=0) - r_seed @ np.mean(local, axis=0)
        out = init.copy()
        out[:3] = _matrix_to_rotvec(r_seed @ self.base_local_to_cam.T)
        out[3:6] = t_seed
        seeded = local @ r_seed.T + t_seed
        gaps = np.linalg.norm(seeded - target, axis=1)
        return out, {
            "used": True,
            "case_specific_state_used": False,
            "correspondence_count": 2,
            "local_chord_length_m": source_length,
            "palm_chord_length_m": target_length,
            "theoretical_min_gap_m": abs(target_length - source_length) * 0.5,
            "seed_median_contact_gap_m": float(np.median(gaps)),
            "rotation_from_stage3_rad": float(math.atan2(sin_angle, cos_angle)),
        }
