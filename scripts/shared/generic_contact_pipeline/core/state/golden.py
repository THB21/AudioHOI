from __future__ import annotations

import csv
import json
from pathlib import Path

from ..base.config import load_case_profile
from ..base.io import REPO
from .shadow import build_state_shadow


CANONICAL_CASE_DIRECTORIES = {
    "basketball": "01_basketball",
    "football": "10_football",
    "mug": "02_mug",
    "chair": "05_chair",
    "stick": "11_stick",
}
DEFAULT_STATE_SHADOW_GOLDEN = REPO / "tests/golden/state_shadow_v1.json"


def state_shadow_summary(case_name: str, pose_csv: Path) -> dict[str, object]:
    profile = load_case_profile(case_name)
    with pose_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    shadow = build_state_shadow(profile, pose_csv, rows)
    geometry_kind = shadow["geometry"]["kind"]
    return {
        "legacy_schema": shadow["legacy_schema"],
        "source_sha256": shadow["source"]["sha256"],
        "rows": shadow["source"]["rows"],
        "state_spec_id": shadow["state_spec"]["spec_id"],
        "geometry_kind": geometry_kind.value if hasattr(geometry_kind, "value") else geometry_kind,
        "geometry_resource_sha256": shadow["geometry"].get("resource_sha256"),
        "canonical_sha256": shadow["canonical_sha256"],
    }


def build_canonical_state_shadow_summary(result_name: str = "benchmark_vlm_qwen") -> dict[str, object]:
    return {
        "schema_version": 1,
        "cases": {
            case_name: state_shadow_summary(
                case_name,
                REPO / "samples_known_object" / directory / "results" / result_name / "object_pose_init.csv",
            )
            for case_name, directory in CANONICAL_CASE_DIRECTORIES.items()
        },
    }


def verify_state_shadow_summary(
    expected_path: Path = DEFAULT_STATE_SHADOW_GOLDEN,
    *,
    result_name: str = "benchmark_vlm_qwen",
) -> list[str]:
    expected = json.loads(expected_path.read_text())
    actual = build_canonical_state_shadow_summary(result_name=result_name)
    errors: list[str] = []
    for case_name, expected_case in expected.get("cases", {}).items():
        actual_case = actual.get("cases", {}).get(case_name)
        if actual_case is None:
            errors.append(f"{case_name}: missing actual state shadow summary")
            continue
        for key, expected_value in expected_case.items():
            actual_value = actual_case.get(key)
            if actual_value != expected_value:
                errors.append(f"{case_name}:{key}: expected {expected_value!r}, got {actual_value!r}")
    for case_name in set(actual.get("cases", {})) - set(expected.get("cases", {})):
        errors.append(f"{case_name}: unexpected actual state shadow summary")
    return errors
