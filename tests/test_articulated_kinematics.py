from __future__ import annotations

import math

import numpy as np

from scripts.shared.generic_contact_pipeline.core.state import ArticulatedKinematicProvider, SegmentJointRule


def test_articulated_provider_rotates_segment_from_data_rule() -> None:
    provider = ArticulatedKinematicProvider(
        rules=(
            SegmentJointRule(
                rule_id="rear_link",
                joint_id="joint.front_to_rear",
                parts=("rear_leg",),
                segment_ids=(),
                origin=np.array([0.0, 0.0, 0.0]),
                axis=np.array([1.0, 0.0, 0.0]),
            ),
        )
    )
    points = np.array([[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]], dtype=float)

    articulated = provider.articulate_segment("rear_leg_left", "rear_leg", points, {"joint.front_to_rear": math.pi / 2.0})

    np.testing.assert_allclose(articulated, [[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]], atol=1e-8)


def test_articulated_provider_can_apply_rule_to_only_rear_endpoint() -> None:
    provider = ArticulatedKinematicProvider(
        rules=(
            SegmentJointRule(
                rule_id="side_stretcher_rear_endpoint",
                joint_id="joint.front_to_rear",
                parts=("side_stretcher",),
                segment_ids=(),
                origin=np.array([0.0, 0.0, 0.0]),
                axis=np.array([1.0, 0.0, 0.0]),
                endpoint_selector="max_y",
            ),
        )
    )
    points = np.array([[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=float)

    articulated = provider.articulate_segment("side_lower_stretcher_left", "side_stretcher", points, {"joint.front_to_rear": math.pi / 2.0})

    np.testing.assert_allclose(articulated, [[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]], atol=1e-8)


def test_articulated_provider_leaves_unmatched_segment_unchanged() -> None:
    provider = ArticulatedKinematicProvider(rules=())
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float)

    articulated = provider.articulate_segment("top_rail", "top_rail", points, {"joint.front_to_rear": 1.0})

    np.testing.assert_allclose(articulated, points)
    assert articulated is not points
