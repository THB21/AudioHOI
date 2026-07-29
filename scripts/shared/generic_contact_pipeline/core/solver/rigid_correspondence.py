"""Case-agnostic rigid initializers from geometric correspondences."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from ..state.types import DofKind, DofSpec, StateSpec


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


def _align_two_points(
    r_init: np.ndarray,
    local: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int | bool]]:
    if local.shape != (2, 3) or target.shape != (2, 3):
        raise ValueError("two point rigid correspondence requires local and target shape (2, 3)")
    if not np.all(np.isfinite(local)) or not np.all(np.isfinite(target)):
        return r_init.copy(), np.zeros(3, dtype=float), {
            "used": False,
            "case_specific_state_used": False,
            "correspondence_count": 2,
        }

    source_chord = r_init @ (local[1] - local[0])
    target_chord = target[1] - target[0]
    source_length = float(np.linalg.norm(source_chord))
    target_length = float(np.linalg.norm(target_chord))
    if source_length < 1e-8 or target_length < 1e-8:
        return r_init.copy(), np.zeros(3, dtype=float), {
            "used": False,
            "case_specific_state_used": False,
            "correspondence_count": 2,
        }

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
    seeded = local @ r_seed.T + t_seed
    gaps = np.linalg.norm(seeded - target, axis=1)
    return r_seed, t_seed, {
        "used": True,
        "case_specific_state_used": False,
        "correspondence_count": 2,
        "local_chord_length_m": source_length,
        "target_chord_length_m": target_length,
        "theoretical_min_gap_m": abs(target_length - source_length) * 0.5,
        "seed_median_contact_gap_m": float(np.median(gaps)),
        "rotation_from_initial_rad": float(math.atan2(sin_angle, cos_angle)),
    }


def _dof_layout(state_spec: StateSpec, dof_id: str, kind: DofKind) -> tuple[DofSpec, slice]:
    offset = 0
    for dof in state_spec.dofs:
        current = slice(offset, offset + dof.dimension)
        if dof.dof_id == dof_id:
            if dof.kind != kind:
                raise ValueError(f"StateSpec {dof_id} must use {kind.value}")
            return dof, current
        offset += dof.dimension
    raise ValueError(f"StateSpec is missing required rigid dof: {dof_id}")


def _component_indices(dof: DofSpec, required: tuple[str, ...]) -> tuple[int, ...]:
    names = tuple(field.rsplit(".", 1)[-1] for field in dof.source_fields)
    if len(names) != dof.dimension or set(names) != set(required):
        raise ValueError(f"StateSpec {dof.dof_id} must identify fields {required}")
    return tuple(names.index(name) for name in required)


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
        r_init = _rotvec_to_matrix(init[:3]) @ self.base_local_to_cam
        r_seed, t_seed, metrics = _align_two_points(r_init, local, target)
        if not bool(metrics["used"]):
            return init.copy(), metrics
        out = init.copy()
        out[:3] = _matrix_to_rotvec(r_seed @ self.base_local_to_cam.T)
        out[3:6] = t_seed
        return out, {
            "used": metrics["used"],
            "case_specific_state_used": metrics["case_specific_state_used"],
            "correspondence_count": metrics["correspondence_count"],
            "local_chord_length_m": metrics["local_chord_length_m"],
            "palm_chord_length_m": metrics["target_chord_length_m"],
            "theoretical_min_gap_m": metrics["theoretical_min_gap_m"],
            "seed_median_contact_gap_m": metrics["seed_median_contact_gap_m"],
            "rotation_from_stage3_rad": metrics["rotation_from_initial_rad"],
        }


@dataclass(frozen=True)
class StateSpecRigidCorrespondenceInitializer:
    """Initialize a rigid root directly in an ordered StateSpec state vector."""

    state_spec: StateSpec

    def align_two_points(
        self,
        state: Sequence[float],
        local: np.ndarray,
        target: np.ndarray,
    ) -> tuple[tuple[float, ...], dict[str, object]]:
        vector = np.asarray(state, dtype=float).copy()
        expected_width = sum(dof.dimension for dof in self.state_spec.dofs)
        if vector.shape != (expected_width,) or not np.all(np.isfinite(vector)):
            raise ValueError("rigid correspondence state must match the finite StateSpec width")
        translation_dof, translation_slice = _dof_layout(
            self.state_spec,
            "root.translation",
            DofKind.TRANSLATION,
        )
        rotation_dof, rotation_slice = _dof_layout(
            self.state_spec,
            "root.rotation",
            DofKind.ROTATION_SO3,
        )
        if not rotation_dof.observable:
            raise ValueError("rigid correspondence requires an observable root rotation")
        translation_indices = _component_indices(translation_dof, ("tx", "ty", "tz"))
        quaternion_indices = _component_indices(rotation_dof, ("qw", "qx", "qy", "qz"))
        rotation_values = vector[rotation_slice]
        initial_quaternion = rotation_values[list(quaternion_indices)].copy()
        try:
            from scipy.spatial.transform import Rotation
        except ImportError as exc:  # pragma: no cover - generic solver runtime dependency
            raise RuntimeError("StateSpec rigid correspondence requires scipy") from exc
        quaternion_norm = float(np.linalg.norm(initial_quaternion))
        if quaternion_norm < 1e-12:
            raise ValueError("rigid correspondence quaternion must be nonzero")
        initial_quaternion /= quaternion_norm
        r_init = Rotation.from_quat(
            [initial_quaternion[1], initial_quaternion[2], initial_quaternion[3], initial_quaternion[0]]
        ).as_matrix()
        r_seed, t_seed, metrics = _align_two_points(r_init, local, target)
        if not bool(metrics["used"]):
            return tuple(float(value) for value in vector), {
                **metrics,
                "state_spec_id": self.state_spec.spec_id,
            }

        qx, qy, qz, qw = Rotation.from_matrix(r_seed).as_quat()
        seeded_quaternion = np.asarray([qw, qx, qy, qz], dtype=float)
        if float(np.dot(seeded_quaternion, initial_quaternion)) < 0.0:
            seeded_quaternion *= -1.0
        out = vector.copy()
        translation_values = out[translation_slice]
        for component, source_index in enumerate(translation_indices):
            translation_values[source_index] = t_seed[component]
        seeded_rotation_values = out[rotation_slice]
        for component, source_index in enumerate(quaternion_indices):
            seeded_rotation_values[source_index] = seeded_quaternion[component]
        return tuple(float(value) for value in out), {
            **metrics,
            "state_spec_id": self.state_spec.spec_id,
        }
