from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.shared.generic_contact_pipeline.core.measurements import (
    CoordinateFrame,
    FeatureRef,
    MeasurementMeta,
    Point2DMeasurement,
    SourceRef,
    Unit,
    adapt_legacy_observation_rows,
)
from scripts.shared.generic_contact_pipeline.core.measurements.shadow import build_measurement_shadow


REPO = Path(__file__).resolve().parents[1]


def test_measurement_rejects_zero_filled_missing_coordinate() -> None:
    meta = MeasurementMeta("s:1:p", "s", 1, 0.0, FeatureRef("center", "object:center"), CoordinateFrame.IMAGE_PIXELS, Unit.PIXEL, None, SourceRef("x.csv", ("u", "v")))
    with pytest.raises(ValueError, match="finite"):
        Point2DMeasurement(meta, float("nan"), 0.0)


def test_point_covariance_requires_valid_pixel_covariance() -> None:
    meta = MeasurementMeta("s:1:p", "s", 1, 0.0, FeatureRef("center", "object:center"), CoordinateFrame.IMAGE_PIXELS, Unit.PIXEL, 0.8, SourceRef("x.csv", ("u", "v")))
    Point2DMeasurement(meta, 10.0, 20.0, ((4.0, 0.5), (0.5, 9.0)))
    with pytest.raises(ValueError, match="symmetric"):
        Point2DMeasurement(meta, 10.0, 20.0, ((4.0, 1.0), (0.0, 9.0)))


@pytest.mark.parametrize(
    ("directory", "schema"),
    [
        ("01_basketball", "proxy_center_depth_v1"),
        ("10_football", "proxy_center_depth_v1"),
        ("02_mug", "rigid_body_parts_v1"),
        ("05_chair", "semantic_graph_v1"),
        ("11_stick", "proxy_center_depth_v1"),
    ],
)
def test_five_case_legacy_observations_adapt_read_only(directory: str, schema: str) -> None:
    path = REPO / "samples_known_object" / directory / "results/benchmark_vlm_qwen/object_observations.csv"
    before = path.read_bytes()
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = adapt_legacy_observation_rows(directory, rows, str(path.relative_to(REPO)))
    assert result.schema == schema
    assert result.measurements
    assert path.read_bytes() == before
    assert all(item.meta.sample_id == directory for item in result.measurements)
    assert all(item.meta.feature.geometry_feature_id for item in result.measurements)
    assert "frame" in result.mapped_fields and "time" in result.mapped_fields


def test_missing_optional_handle_point_is_absent_not_zero() -> None:
    path = REPO / "samples_known_object/02_mug/results/benchmark_vlm_qwen/object_observations.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = adapt_legacy_observation_rows("mug", rows, str(path.relative_to(REPO)))
    handle_points = [m for m in result.measurements if m.meta.feature.geometry_feature_id == "object:handle" and m.kind == "point2d"]
    assert len(handle_points) == 159


def test_shadow_manifest_is_deterministic_and_never_solver_consumed() -> None:
    path = REPO / "samples_known_object/05_chair/results/benchmark_vlm_qwen/object_observations.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    first = build_measurement_shadow("chair", path, rows)
    second = build_measurement_shadow("chair", path, rows)
    assert first == second
    assert first["mode"] == "read_only_shadow"
    assert first["consumed_by_solver"] is False
    assert first["measurements"]["frames"] == 192
    assert first["measurements"]["canonical_sha256"]
    assert first["coverage"]["unmapped_nonempty_fields"]


def test_shadow_hash_is_independent_of_absolute_worktree_prefix() -> None:
    relative = Path("samples_known_object/05_chair/results/benchmark_vlm_qwen/object_observations.csv")
    absolute = REPO / relative
    with absolute.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    relative_shadow = build_measurement_shadow("chair", relative, rows)
    absolute_shadow = build_measurement_shadow("chair", absolute, rows)
    assert relative_shadow["source"] == absolute_shadow["source"]
    assert relative_shadow["measurements"]["canonical_sha256"] == absolute_shadow["measurements"]["canonical_sha256"]
