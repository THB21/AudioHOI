from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.shared.generic_contact_pipeline.core.contact_constraints import (
    ContactMode,
    ContactState,
    ContactConstraint,
    FrameInterval,
    HumanSite,
    LocalXYZ,
    apply_contact_state_gate,
    adapt_legacy_contact_rows,
    build_contact_constraint_shadow,
)
from scripts.shared.generic_contact_pipeline.core.measurements import FeatureRef, SourceRef


REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("directory", "schema", "coordinate_kind"),
    [
        ("01_basketball", "feature_contact_v1", "none"),
        ("10_football", "feature_contact_v1", "none"),
        ("02_mug", "stable_local_xyz_contact_v1", "local_xyz"),
        ("05_chair", "local_xyz_contact_v1", "local_xyz"),
        ("11_stick", "line_s_contact_v1", "line_s"),
    ],
)
def test_five_case_contacts_adapt_read_only(directory: str, schema: str, coordinate_kind: str) -> None:
    path = REPO / "samples_known_object" / directory / "results/benchmark_vlm_qwen/object_contact_points.csv"
    before = path.read_bytes()
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = adapt_legacy_contact_rows(directory, rows, str(path.relative_to(REPO)))
    assert result.schema == schema
    assert len(result.constraints) == len(rows)
    assert path.read_bytes() == before
    kinds = {item.object_coordinate.kind if item.object_coordinate else "none" for item in result.constraints}
    assert coordinate_kind in kinds
    assert all(item.object_feature.geometry_feature_id for item in result.constraints)


def test_mug_occluded_hold_keeps_explicit_local_coordinate() -> None:
    path = REPO / "samples_known_object/02_mug/results/benchmark_vlm_qwen/object_contact_points.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = adapt_legacy_contact_rows("mug", rows, str(path.relative_to(REPO)))
    held = [item for item in result.constraints if item.state == ContactState.OCCLUDED_HOLD]
    assert len(held) == 65
    assert all(isinstance(item.object_coordinate, LocalXYZ) for item in held)


def test_shadow_is_deterministic_and_reports_unmapped_fields() -> None:
    path = REPO / "samples_known_object/11_stick/results/benchmark_vlm_qwen/object_contact_points.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    first = build_contact_constraint_shadow("stick", path, rows)
    second = build_contact_constraint_shadow("stick", path, rows)
    assert first == second
    assert first["consumed_by_solver"] is False
    assert first["constraints"]["by_coordinate"] == {"line_s": 480}
    assert "palm_to_line_px" in first["coverage"]["unmapped_nonempty_fields"]


def test_gate_changes_only_discrete_state_confidence_and_provenance() -> None:
    coordinate = LocalXYZ(0.1, 0.2, 0.3)
    constraint = ContactConstraint(
        "s:1:c", "s", FrameInterval(1, 1), 0.0, 0.0,
        HumanSite("palm", "left"), FeatureRef("handle", "object:handle"), coordinate,
        ContactMode.GRASP, ContactState.CANDIDATE, 0.5, None, SourceRef("contacts.csv", ("contact_active",)),
    )
    gated = apply_contact_state_gate(
        constraint,
        state=ContactState.ACTIVE,
        confidence=0.9,
        evidence=SourceRef("vlm_gates.csv", ("decision", "confidence"), producer="vlm_gate"),
    )
    assert gated.state == ContactState.ACTIVE and gated.confidence == 0.9
    assert gated.object_coordinate is coordinate
    assert gated.object_feature == constraint.object_feature
    assert len(gated.gate_provenance) == 1
