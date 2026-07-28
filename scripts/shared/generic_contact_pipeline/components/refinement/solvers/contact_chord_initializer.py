"""Compatibility wrapper for two-point rigid contact initialization."""
from __future__ import annotations

import numpy as np

from scripts.shared.generic_contact_pipeline.core.solver import RigidCorrespondenceInitializer


def align_contact_chord(
    init: np.ndarray,
    local: np.ndarray,
    target: np.ndarray,
    base_local_to_cam: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | bool]]:
    """Align two local endpoints to two observed targets with minimum twist."""
    seeded, metrics = RigidCorrespondenceInitializer(base_local_to_cam=base_local_to_cam).align_two_points(
        init,
        local,
        target,
    )
    return seeded, {key: value for key, value in metrics.items() if key != "case_specific_state_used"}
