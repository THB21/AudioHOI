from __future__ import annotations

import numpy as np

from scripts.shared.generic_contact_pipeline.core.solver import RigidCorrespondenceInitializer


BASE_LOCAL_TO_CAM = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def test_rigid_correspondence_initializer_aligns_two_point_chord_without_case_state() -> None:
    initializer = RigidCorrespondenceInitializer(base_local_to_cam=BASE_LOCAL_TO_CAM)
    local = np.array([[-0.2, 0.1, 0.7], [0.2, 0.1, 0.7]], dtype=float)
    target = np.array([[-0.1, 0.3, 2.0], [-0.1, 0.7, 2.0]], dtype=float)
    init = np.zeros(8)

    seeded, metrics = initializer.align_two_points(init, local, target)

    assert metrics["used"] is True
    assert metrics["case_specific_state_used"] is False
    assert metrics["correspondence_count"] == 2
    assert metrics["theoretical_min_gap_m"] < 1e-8
    assert metrics["seed_median_contact_gap_m"] < 1e-8
    assert seeded.shape == init.shape


def test_rigid_correspondence_initializer_rejects_non_two_point_contract() -> None:
    initializer = RigidCorrespondenceInitializer(base_local_to_cam=BASE_LOCAL_TO_CAM)

    try:
        initializer.align_two_points(np.zeros(8), np.zeros((3, 3)), np.zeros((3, 3)))
    except ValueError as exc:
        assert "two point" in str(exc)
    else:
        raise AssertionError("non-two-point correspondence should fail")
