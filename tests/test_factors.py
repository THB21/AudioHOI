from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.shared.generic_contact_pipeline.core.base.config import load_case_profile
from scripts.shared.generic_contact_pipeline.core.factors import (
    FactorInputRef,
    FactorKind,
    FactorSourceRef,
    FactorSpec,
    build_canonical_factor_shadow_summary,
    build_factor_shadow,
    verify_factor_shadow_summary,
    validate_factor_shadow,
)


REPO = Path(__file__).resolve().parents[1]
CASE_DIRECTORIES = {
    "basketball": "01_basketball",
    "football": "10_football",
    "mug": "02_mug",
    "chair": "05_chair",
    "stick": "11_stick",
}


def test_factor_spec_rejects_solver_consumption() -> None:
    with pytest.raises(ValueError, match="must not be solver-consumed"):
        FactorSpec(
            "bad",
            FactorKind.CONTACT_DISTANCE,
            1,
            (FactorInputRef("state", "StateSpec", "root"),),
            "legacy_energy",
            "test",
            None,
            FactorSourceRef("x.csv", ("E_contact",), "test"),
            consumed_by_solver=True,
        )


def test_five_case_factor_shadow_matches_frozen_manifest() -> None:
    expected = json.loads((REPO / "tests/golden/factor_shadow_v1.json").read_text())
    actual = build_canonical_factor_shadow_summary()
    assert actual == expected
    assert verify_factor_shadow_summary() == []


def test_factor_shadow_has_no_dead_no_contact_anchor_consumer() -> None:
    for case_name, directory in CASE_DIRECTORIES.items():
        result_dir = REPO / "samples_known_object" / directory / "results/benchmark_vlm_qwen"
        shadow = build_factor_shadow(load_case_profile(case_name), result_dir)
        payload = json.dumps(shadow, sort_keys=True)
        assert "no_contact_anchor" not in payload
        assert shadow["consumed_by_solver"] is False


def test_ball_factor_shadow_exposes_depth_contact_and_temporal_terms() -> None:
    result_dir = REPO / "samples_known_object/01_basketball/results/benchmark_vlm_qwen"
    shadow = build_factor_shadow(load_case_profile("basketball"), result_dir)
    kinds = shadow["factors"]["by_kind"]
    assert kinds["metric_depth"] == 1
    assert kinds["contact_distance"] >= 1
    assert kinds["temporal_velocity"] >= 1


def test_chair_factor_shadow_exposes_joint_limit_and_gauge_terms() -> None:
    result_dir = REPO / "samples_known_object/05_chair/results/benchmark_vlm_qwen"
    shadow = build_factor_shadow(load_case_profile("chair"), result_dir)
    kinds = shadow["factors"]["by_kind"]
    assert kinds["joint_limit"] == 2
    assert kinds["gauge_constraint"] == 1


def test_chair_factor_shadow_maps_audio_static_prior() -> None:
    result_dir = REPO / "samples_known_object/05_chair/results/benchmark_vlm_qwen"
    shadow = build_factor_shadow(load_case_profile("chair"), result_dir)
    gap_ids = [gap["gap_id"] for gap in shadow["gaps"]]
    assert "unsupported_loss_term:E_audio" not in gap_ids
    assert "E_audio" not in shadow["coverage"]["unmapped_nonempty_fields"]
    audio_factors = [
        factor
        for factor in shadow["factors"]["records"]
        if factor["factor_id"] == "audio_event_prior:E_audio"
    ]
    assert len(audio_factors) == 1
    assert audio_factors[0]["kind"] == FactorKind.AUDIO_EVENT_PRIOR
    assert audio_factors[0]["frame_count"] == 8
    assert audio_factors[0]["gate_source"] == "audio/contact/static gates in per_frame_residuals.csv"
    input_roles = {(ref["role"], ref["source_ir"], ref["source_id"]) for ref in audio_factors[0]["input_refs"]}
    assert ("measurement", "AudioEventIR", "audio_events") in input_roles
    assert ("constraint", "ContactConstraintIR", "audio_contact_phase") in input_roles


def test_specialized_legacy_paths_remain_explicit_gaps() -> None:
    mug = build_factor_shadow(load_case_profile("mug"), REPO / "samples_known_object/02_mug/results/benchmark_vlm_qwen")
    chair = build_factor_shadow(load_case_profile("chair"), REPO / "samples_known_object/05_chair/results/benchmark_vlm_qwen")
    stick = build_factor_shadow(load_case_profile("stick"), REPO / "samples_known_object/11_stick/results/benchmark_vlm_qwen")
    assert [gap["gap_id"] for gap in mug["gaps"]] == ["phase_snapshot_fallback"]
    assert "semantic_graph_solver_private" not in [gap["gap_id"] for gap in chair["gaps"]]
    assert [gap["gap_id"] for gap in stick["gaps"]] == ["line_contact_lock_special_refinement"]


def test_factor_shadow_export_cli_writes_reviewable_manifest(tmp_path: Path) -> None:
    out = tmp_path / "football_factor_shadow.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/shared/generic_contact_pipeline/tools/export_factor_shadow.py",
            "--case",
            "football",
            "--result-dir",
            "samples_known_object/10_football/results/benchmark_vlm_qwen",
            "--out",
            str(out),
        ],
        cwd=REPO,
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(out.read_text())
    assert json.loads(completed.stdout) == payload
    assert payload["mode"] == "read_only_shadow"
    assert payload["factors"]["by_kind"]["metric_depth"] == 1


def test_factor_shadow_verifier_cli_reports_all_cases() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/shared/generic_contact_pipeline/tools/verify_factor_shadow.py",
        ],
        cwd=REPO,
        check=True,
        text=True,
        capture_output=True,
    )
    lines = completed.stdout.strip().splitlines()
    assert len(lines) == len(CASE_DIRECTORIES)
    assert lines[0].startswith("basketball: factors=10")
    assert any("line_contact_lock_special_refinement" in line for line in lines)


def test_factor_shadow_validation_checks_composition_and_sources() -> None:
    shadow = build_factor_shadow(load_case_profile("chair"), REPO / "samples_known_object/05_chair/results/benchmark_vlm_qwen")
    assert validate_factor_shadow(shadow) == []


def test_factor_shadow_validation_rejects_absolute_residual_sources() -> None:
    shadow = build_factor_shadow(load_case_profile("basketball"), REPO / "samples_known_object/01_basketball/results/benchmark_vlm_qwen")
    shadow["factors"]["records"][0]["residual_source"]["artifact"] = "/tmp/not_repo_relative.csv"
    errors = validate_factor_shadow(shadow)
    assert any("repo-relative" in error for error in errors)
