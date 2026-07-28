from __future__ import annotations

import numpy as np

from scripts.shared.generic_contact_pipeline.core.solver import FactorResidualEvaluator


def test_factor_residual_evaluator_matches_legacy_weighted_point_and_contact_blocks() -> None:
    evaluator = FactorResidualEvaluator()

    point = evaluator.point_reprojection(
        predicted=np.array([[12.0, 21.0], [29.0, 41.0]], dtype=float),
        target=np.array([[10.0, 20.0], [30.0, 40.0]], dtype=float),
        weight=2.0,
        sigma_px=5.0,
    )
    contact = evaluator.contact_distance(
        anchors=np.array([[1.0, 2.0, 3.0]], dtype=float),
        targets=np.array([[1.1, 1.8, 3.3]], dtype=float),
        weight=0.5,
        sigma_m=0.1,
    )

    np.testing.assert_allclose(point, [0.8, 0.4, -0.4, 0.4])
    np.testing.assert_allclose(contact, [-0.5, 1.0, -1.5])


def test_factor_residual_evaluator_supports_depth_and_support_penetration_blocks() -> None:
    evaluator = FactorResidualEvaluator()

    depth = evaluator.metric_depth(
        predicted_depth_m=np.array([1.0, 1.4, 2.2], dtype=float),
        target_depth_m=np.array([0.9, 1.5, 2.0], dtype=float),
        weight=0.5,
        sigma_m=0.1,
    )
    support = evaluator.support_penetration(
        signed_distance_m=np.array([0.05, -0.02, -0.08], dtype=float),
        weight=2.0,
        sigma_m=0.04,
    )

    np.testing.assert_allclose(depth, [0.5, -0.5, 1.0])
    np.testing.assert_allclose(support, [0.0, -1.0, -4.0])


def test_factor_residual_evaluator_matches_legacy_pose_prior_and_temporal_blocks() -> None:
    evaluator = FactorResidualEvaluator()
    x = np.array([0.2, -0.1, 0.3, 1.0, 2.0, 3.0], dtype=float)
    ref = np.zeros(6, dtype=float)
    init = np.array([0.0, 0.0, 0.0, 0.5, 2.5, 2.0], dtype=float)
    prev = np.array([0.1, -0.2, 0.1, 0.8, 1.9, 2.7], dtype=float)

    prior = evaluator.pose_prior(
        x,
        ref,
        init,
        rot_bound=0.1,
        xy_bound=0.5,
        z_bound=2.0,
        w_prior_rot=0.25,
        w_prior_xy=0.5,
        w_prior_z=0.75,
    )
    temporal = evaluator.temporal_delta(
        x,
        prev,
        weight=0.2,
        scales=np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.3], dtype=float),
    )

    np.testing.assert_allclose(prior, [0.5, -0.25, 0.75, 1.0, 2.0, 0.375])
    np.testing.assert_allclose(temporal, [0.2, 0.2, 0.4, 0.2, 0.1, 0.2])


def test_factor_residual_evaluator_supports_joint_limit_and_gauge_blocks() -> None:
    evaluator = FactorResidualEvaluator()

    joint = evaluator.joint_limit(
        values=np.array([-1.0, -0.5, 0.2, 0.8], dtype=float),
        lower=-0.5,
        upper=0.5,
        weight=2.0,
        sigma_rad=0.1,
    )
    gauge = evaluator.gauge_constraint(
        values=np.array([0.25, -0.5], dtype=float),
        target=0.0,
        weight=0.4,
        sigma=0.1,
    )

    np.testing.assert_allclose(joint, [-10.0, 0.0, 0.0, 6.0])
    np.testing.assert_allclose(gauge, [1.0, -2.0])


def test_factor_residual_evaluator_supports_generic_regularization_block() -> None:
    evaluator = FactorResidualEvaluator()

    residual = evaluator.regularization(
        values=np.array([1.0, -2.0, 0.5], dtype=float),
        target=np.array([0.0, -1.0, 0.5], dtype=float),
        weight=0.3,
        scales=np.array([0.5, 2.0, 0.25], dtype=float),
    )

    np.testing.assert_allclose(residual, [0.6, -0.15, 0.0])


def test_factor_residual_evaluator_supports_wrapped_periodic_phase_prior_block() -> None:
    evaluator = FactorResidualEvaluator()

    residual = evaluator.periodic_phase_prior(
        values=np.array([np.pi - 0.1, -np.pi + 0.2, 0.5], dtype=float),
        target=np.array([-np.pi + 0.1, np.pi - 0.2, 0.5], dtype=float),
        weight=0.5,
        sigma_rad=0.1,
    )

    np.testing.assert_allclose(residual, [-1.0, 2.0, 0.0])


def test_factor_residual_evaluator_supports_audio_event_timing_prior_block() -> None:
    evaluator = FactorResidualEvaluator()

    residual = evaluator.audio_event_prior(
        predicted_event_time_s=np.array([0.12, 0.30, 0.55], dtype=float),
        observed_event_time_s=np.array([0.10, 0.34, 0.50], dtype=float),
        weight=0.25,
        sigma_s=0.02,
    )

    np.testing.assert_allclose(residual, [0.25, -0.5, 0.625])
