"""Pure two-point rigid initializer shared by contact refinement solvers."""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation


def align_contact_chord(
    init: np.ndarray,
    local: np.ndarray,
    target: np.ndarray,
    base_local_to_cam: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | bool]]:
    """Align two local endpoints to two observed targets with minimum twist."""
    if local.shape != (2, 3) or target.shape != (2, 3):
        raise ValueError("local and target contact chords must both have shape (2, 3)")
    if not np.all(np.isfinite(local)) or not np.all(np.isfinite(target)):
        return init.copy(), {"used": False}

    r_init = Rotation.from_rotvec(init[:3]).as_matrix() @ base_local_to_cam
    source_chord = r_init @ (local[1] - local[0])
    target_chord = target[1] - target[0]
    source_length = float(np.linalg.norm(source_chord))
    target_length = float(np.linalg.norm(target_chord))
    if source_length < 1e-8 or target_length < 1e-8:
        return init.copy(), {"used": False}

    source_unit = source_chord / source_length
    target_unit = target_chord / target_length
    cross = np.cross(source_unit, target_unit)
    sin_angle = float(np.linalg.norm(cross))
    cos_angle = float(np.clip(np.dot(source_unit, target_unit), -1.0, 1.0))
    if sin_angle > 1e-8:
        delta = Rotation.from_rotvec(cross / sin_angle * math.atan2(sin_angle, cos_angle)).as_matrix()
    elif cos_angle >= 0.0:
        delta = np.eye(3)
    else:
        basis = np.eye(3)[int(np.argmin(np.abs(source_unit)))]
        axis = np.cross(source_unit, basis)
        axis /= np.linalg.norm(axis)
        delta = Rotation.from_rotvec(axis * math.pi).as_matrix()

    r_seed = delta @ r_init
    t_seed = np.mean(target, axis=0) - r_seed @ np.mean(local, axis=0)
    out = init.copy()
    out[:3] = Rotation.from_matrix(r_seed @ base_local_to_cam.T).as_rotvec()
    out[3:6] = t_seed
    seeded = local @ r_seed.T + t_seed
    gaps = np.linalg.norm(seeded - target, axis=1)
    return out, {
        "used": True,
        "local_chord_length_m": source_length,
        "palm_chord_length_m": target_length,
        "theoretical_min_gap_m": abs(target_length - source_length) * 0.5,
        "seed_median_contact_gap_m": float(np.median(gaps)),
        "rotation_from_stage3_rad": float(math.atan2(sin_angle, cos_angle)),
    }
