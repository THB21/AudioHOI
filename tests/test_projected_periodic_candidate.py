from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.shared.generic_contact_pipeline.core.base.config import load_case_profile
from scripts.shared.generic_contact_pipeline.core.solver import write_candidate_sandbox_manifest
from scripts.shared.generic_contact_pipeline.components.geometry.mug_periodic import (
    MugPeriodicGeometryProvider,
    adapt_mug_periodic_observations,
)
from scripts.shared.generic_contact_pipeline.core.solver import (
    build_projected_periodic_regression_summary,
    verify_projected_periodic_regression,
)
from scripts.shared.generic_contact_pipeline.core.provenance.attempts import STAGE_ARTIFACT_KEYS


REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "tests/golden/mug_projected_periodic_migration_v1.json"


def test_mug_adapter_preserves_detector_extent_and_sorts_frames() -> None:
    rows = [
        {
            "frame": "2", "time": "0.041667", "body_center_x": "10", "body_center_y": "20",
            "body_bbox_x1": "1", "body_bbox_y1": "2", "body_bbox_x2": "11", "body_bbox_y2": "42",
            "body_bbox_w_px": "11", "body_bbox_h_px": "41", "handle_center_x": "30",
            "handle_center_y": "40", "handle_visible": "1",
        },
        {
            "frame": "1", "time": "0.000000", "body_center_x": "9", "body_center_y": "19",
            "body_bbox_x1": "1", "body_bbox_y1": "2", "body_bbox_x2": "11", "body_bbox_y2": "42",
            "body_bbox_w_px": "11", "body_bbox_h_px": "42", "handle_visible": "0",
        },
    ]
    proxies = {
        1: {"object_depth_smooth": "", "da3_depth_smooth": "3.1"},
        2: {"object_depth_smooth": "3.2", "da3_depth_smooth": "3.0"},
    }
    observations = adapt_mug_periodic_observations(rows, proxies)
    assert [item.frame for item in observations] == [1, 2]
    assert observations[0].body_bbox_xyxy == (1.0, 2.0, 11.0, 42.0)
    assert observations[0].body_extent_uv == (11.0, 42.0)
    assert observations[0].metric_depth_m == 3.1
    assert observations[0].periodic_feature_uv is None
    assert observations[1].periodic_feature_uv == (30.0, 40.0)


def test_mug_periodic_contract_is_rigid_gauge_not_articulation() -> None:
    contract = MugPeriodicGeometryProvider.kinematic_contract
    assert contract.root_node == "body"
    assert contract.periodic_feature_node == "handle"
    assert contract.physical_joint is False
    assert contract.relative_motion_allowed is False
    assert contract.gauge_constraint == "body.symmetry_phase = 0"
    assert "delta" in contract.gauge_transform


def test_generic_projected_periodic_core_has_no_case_or_accepted_pose_paths() -> None:
    core_source = (
        REPO / "scripts/shared/generic_contact_pipeline/core/solver/projected_periodic_sequence.py"
    ).read_text().lower()
    provider_source = (
        REPO / "scripts/shared/generic_contact_pipeline/components/geometry/mug_periodic.py"
    ).read_text()
    assert "mug" not in core_source
    assert "object_pose.csv" not in core_source
    assert "benchmark" not in core_source
    assert "fit_mug_" not in provider_source
    assert "render_mug_" not in provider_source


def test_periodic_seed_outputs_are_stage1_attempt_artifacts() -> None:
    assert {
        "periodic_root_seed",
        "periodic_feature_phase",
        "periodic_observation_report",
    }.issubset(STAGE_ARTIFACT_KEYS["stage1"])


def test_materialized_mug_periodic_candidate_verifier_cli_reports_candidate(tmp_path: Path) -> None:
    candidate_dir = tmp_path / "mug_candidate"
    write_candidate_sandbox_manifest(
        load_case_profile("mug"),
        REPO / "samples_known_object/02_mug/results/benchmark_vlm_qwen",
        candidate_dir,
    )
    publication = json.loads((candidate_dir / "generic_object_publication.json").read_text())
    preparation = json.loads((candidate_dir / "generic_problem_preparation.json").read_text())
    assert publication["status"] == "candidate_blocked"
    assert publication["case_dispatch_used"] is False
    assert preparation["initializer_kind"] == "observation_periodic_rigid"


@pytest.mark.repository_data
def test_repository_mug_candidate_matches_frozen_fresh_baseline() -> None:
    attempt = REPO / (
        "samples_known_object/02_mug/results/generic_mug_migration_baseline_v1/"
        "generic_projected_periodic_candidate/generic_projected_periodic_attempt.json"
    )
    if not attempt.exists():
        pytest.skip("hydrated/generated mug migration candidate is unavailable")
    assert build_projected_periodic_regression_summary(attempt) == json.loads(GOLDEN.read_text())
    assert verify_projected_periodic_regression(attempt, GOLDEN) == []
