from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.shared.generic_contact_pipeline.core.base.config import load_case_profile
from scripts.shared.generic_contact_pipeline.core.state import (
    DofKind,
    GeometryKind,
    StateSpec,
    adapt_legacy_state_rows,
    build_canonical_state_parity_reports,
    build_canonical_state_shadow_summary,
    build_state_parity_report,
    build_state_shadow,
    verify_state_shadow_summary,
)


REPO = Path(__file__).resolve().parents[1]
CASE_DIRECTORIES = {
    "basketball": "01_basketball",
    "football": "10_football",
    "mug": "02_mug",
    "chair": "05_chair",
    "stick": "11_stick",
}


@pytest.mark.parametrize(
    ("case_name", "directory", "schema", "geometry_kind"),
    [
        ("basketball", "01_basketball", "translation3_sphere_v1", GeometryKind.SPHERE),
        ("football", "10_football", "translation3_sphere_v1", GeometryKind.SPHERE),
        ("mug", "02_mug", "rigid6_plus_periodic_phase_v1", GeometryKind.RIGID_MESH),
        ("chair", "05_chair", "semantic_graph_6d_v1", GeometryKind.ARTICULATED_URDF),
        ("stick", "11_stick", "translation3_line_capsule_v1", GeometryKind.LINE_CAPSULE),
    ],
)
def test_five_case_state_specs_adapt_read_only(
    case_name: str,
    directory: str,
    schema: str,
    geometry_kind: GeometryKind,
) -> None:
    profile = load_case_profile(case_name)
    path = REPO / "samples_known_object" / directory / "results/benchmark_vlm_qwen/object_pose_init.csv"
    before = path.read_bytes()
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    result = adapt_legacy_state_rows(profile, rows, str(path.relative_to(REPO)))

    assert result.schema == schema
    assert isinstance(result.state_spec, StateSpec)
    assert result.state_spec.consumed_by_solver is False
    assert result.geometry.kind == geometry_kind
    assert "frame" in result.mapped_fields and "time" in result.mapped_fields
    assert path.read_bytes() == before


def test_sphere_rotation_is_declared_unobservable_not_zero_filled() -> None:
    profile = load_case_profile("basketball")
    path = REPO / "samples_known_object/01_basketball/results/benchmark_vlm_qwen/object_pose_init.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    result = adapt_legacy_state_rows(profile, rows, str(path.relative_to(REPO)))

    rotation = next(dof for dof in result.state_spec.dofs if dof.dof_id == "root.rotation")
    assert rotation.kind == DofKind.ROTATION_SO3
    assert rotation.observable is False
    assert result.state_spec.gauge_constraints[0].gauge_id == "sphere.rotation_unobservable"


def test_mug_phase_is_periodic_and_yaw_is_not_forced_zero() -> None:
    profile = load_case_profile("mug")
    path = REPO / "samples_known_object/02_mug/results/benchmark_vlm_qwen/object_pose_init.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    result = adapt_legacy_state_rows(profile, rows, str(path.relative_to(REPO)))

    phase = next(dof for dof in result.state_spec.dofs if dof.dof_id == "handle.phase")
    assert phase.kind == DofKind.PERIODIC
    assert phase.bound is not None
    assert phase.bound.source == "periodic_wrap"
    assert result.state_spec.gauge_constraints[0].gauge_id == "mug_yaw_not_forced_zero"
    assert "yaw" in result.mapped_fields


def test_chair_joint_limits_come_from_urdf_source() -> None:
    profile = load_case_profile("chair")
    path = REPO / "samples_known_object/05_chair/results/benchmark_vlm_qwen/object_pose_init.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    result = adapt_legacy_state_rows(profile, rows, str(path.relative_to(REPO)))

    limits = {dof.dof_id: dof.bound for dof in result.state_spec.dofs if dof.bound}
    assert limits["joint.front_to_rear"].lower == -0.82
    assert limits["joint.front_to_rear"].upper == 0.12
    assert limits["joint.front_to_rear"].source == "chair_urdf:front_to_rear"
    assert limits["joint.front_to_seat"].lower == 0.0
    assert limits["joint.front_to_seat"].upper == 1.35
    assert result.geometry.resource_path is not None and result.geometry.resource_path.endswith("model.urdf")
    assert result.geometry.resource_sha256


def test_state_shadow_is_deterministic_and_never_solver_consumed() -> None:
    profile = load_case_profile("stick")
    path = REPO / "samples_known_object/11_stick/results/benchmark_vlm_qwen/object_pose_init.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    first = build_state_shadow(profile, path, rows)
    second = build_state_shadow(profile, path, rows)

    assert first == second
    assert first["mode"] == "read_only_shadow"
    assert first["consumed_by_solver"] is False
    assert first["legacy_schema"] == "translation3_line_capsule_v1"
    assert first["state_spec"]["gauge_constraints"][0]["gauge_id"] == "line.roll_unobservable"
    assert first["geometry"]["resource_sha256"]
    assert first["canonical_sha256"]


def test_state_shadow_cli_writes_reviewable_manifest(tmp_path: Path) -> None:
    pose_csv = REPO / "samples_known_object/05_chair/results/benchmark_vlm_qwen/object_pose_init.csv"
    out = tmp_path / "chair_state_shadow.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/shared/generic_contact_pipeline/tools/export_state_shadow.py",
            "--case",
            "chair",
            "--pose-csv",
            str(pose_csv),
            "--out",
            str(out),
        ],
        cwd=REPO,
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(out.read_text())
    echoed = json.loads(completed.stdout)
    assert echoed == payload
    assert payload["mode"] == "read_only_shadow"
    assert payload["legacy_schema"] == "semantic_graph_6d_v1"
    assert payload["state_spec"]["dofs"][2]["bound"]["source"] == "chair_urdf:front_to_rear"


def test_five_case_state_shadow_matches_frozen_manifest() -> None:
    expected = json.loads((REPO / "tests/golden/state_shadow_v1.json").read_text())
    actual = build_canonical_state_shadow_summary()
    assert actual == expected
    assert verify_state_shadow_summary() == []


def test_state_shadow_verifier_cli_reports_all_cases() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/shared/generic_contact_pipeline/tools/verify_state_shadow.py",
        ],
        cwd=REPO,
        check=True,
        text=True,
        capture_output=True,
    )

    lines = completed.stdout.strip().splitlines()
    assert len(lines) == len(CASE_DIRECTORIES)
    assert lines[0].startswith("basketball: translation3_sphere_v1")
    assert any(line.startswith("chair: semantic_graph_6d_v1") for line in lines)


def test_five_case_state_parity_reports_pass_without_solver_consumption() -> None:
    reports = build_canonical_state_parity_reports()
    assert set(reports["cases"]) == set(CASE_DIRECTORIES)
    for case_name, report in reports["cases"].items():
        assert report["mode"] == "read_only_shadow"
        assert report["consumed_by_solver"] is False
        assert report["summary"]["status"] == "pass", case_name
        assert report["summary"]["failed"] == 0
        assert report["summary"]["canonical_sha256"]
    assert reports["cases"]["chair"]["summary"]["warnings"] == 2


def test_state_parity_cli_writes_reviewable_report(tmp_path: Path) -> None:
    pose_csv = REPO / "samples_known_object/02_mug/results/benchmark_vlm_qwen/object_pose_init.csv"
    out = tmp_path / "mug_state_parity.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/shared/generic_contact_pipeline/tools/export_state_parity.py",
            "--case",
            "mug",
            "--pose-csv",
            str(pose_csv),
            "--out",
            str(out),
        ],
        cwd=REPO,
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(out.read_text())
    echoed = json.loads(completed.stdout)
    assert echoed == payload
    assert payload["summary"]["status"] == "pass"
    assert any(check["check_id"] == "handle_phase.deg_matches_rad" for check in payload["checks"])


def test_state_parity_report_is_deterministic() -> None:
    profile = load_case_profile("football")
    pose_csv = REPO / "samples_known_object/10_football/results/benchmark_vlm_qwen/object_pose_init.csv"
    with pose_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    first = build_state_parity_report(profile, pose_csv, rows)
    second = build_state_parity_report(profile, pose_csv, rows)
    assert first == second


def test_state_parity_verifier_cli_reports_warnings_without_default_failure() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/shared/generic_contact_pipeline/tools/verify_state_parity.py",
        ],
        cwd=REPO,
        check=True,
        text=True,
        capture_output=True,
    )

    lines = completed.stdout.strip().splitlines()
    assert len(lines) == len(CASE_DIRECTORIES)
    assert any(line.startswith("chair: status=pass") and "warnings=2" in line for line in lines)
    assert "warning: chair:joint.front_to_rear.within_urdf_limit" in completed.stderr


def test_state_parity_verifier_cli_can_strictly_fail_warnings() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/shared/generic_contact_pipeline/tools/verify_state_parity.py",
            "--strict-warnings",
        ],
        cwd=REPO,
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 1
    assert "chair:joint.front_to_seat.within_urdf_limit" in completed.stderr
