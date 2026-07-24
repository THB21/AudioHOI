from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy", reason="chair runtime dependency is installed in the audiohoi environment")
from scipy.spatial.transform import Rotation

from scripts.shared.generic_contact_pipeline.components.refinement.solvers.contact_chord_initializer import align_contact_chord


BASE_LOCAL_TO_CAM = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def test_contact_chord_seed_reaches_rigidly_compatible_palms() -> None:
    left = np.array([-0.2, 0.1, 0.7])
    right = np.array([0.2, 0.1, 0.7])
    target_left = np.array([-0.1, 0.3, 2.0])
    target_right = np.array([-0.1, 0.7, 2.0])
    init = np.zeros(8)

    seeded, metrics = align_contact_chord(init, np.stack([left, right]), np.stack([target_left, target_right]), BASE_LOCAL_TO_CAM)

    rotation = Rotation.from_rotvec(seeded[:3]).as_matrix() @ BASE_LOCAL_TO_CAM
    transformed = np.stack([left, right]) @ rotation.T + seeded[3:6]
    np.testing.assert_allclose(transformed, np.stack([target_left, target_right]), atol=1e-8)
    assert metrics["used"] is True
    assert metrics["theoretical_min_gap_m"] < 1e-8


def test_contact_chord_seed_reports_unavoidable_length_residual() -> None:
    left = np.array([-0.2, 0.0, 0.0])
    right = np.array([0.2, 0.0, 0.0])
    target_left = np.array([0.0, 0.0, 2.0])
    target_right = np.array([0.6, 0.0, 2.0])

    seeded, metrics = align_contact_chord(
        np.zeros(8), np.stack([left, right]), np.stack([target_left, target_right]), BASE_LOCAL_TO_CAM
    )

    rotation = Rotation.from_rotvec(seeded[:3]).as_matrix() @ BASE_LOCAL_TO_CAM
    transformed = np.stack([left, right]) @ rotation.T + seeded[3:6]
    gaps = np.linalg.norm(transformed - np.stack([target_left, target_right]), axis=1)
    np.testing.assert_allclose(gaps, [0.1, 0.1], atol=1e-8)
    assert metrics["theoretical_min_gap_m"] == 0.1
