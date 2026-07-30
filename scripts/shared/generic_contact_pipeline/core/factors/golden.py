from __future__ import annotations

import json
from pathlib import Path

from ..base.config import load_case_profile
from ..base.io import REPO
from ..state.golden import CANONICAL_CASE_DIRECTORIES
from .shadow import build_factor_shadow


DEFAULT_FACTOR_SHADOW_GOLDEN = REPO / "tests/golden/factor_shadow_v1.json"


def factor_shadow_summary(case_name: str, result_dir: Path) -> dict[str, object]:
    shadow = build_factor_shadow(load_case_profile(case_name), result_dir)
    return {
        "factor_count": shadow["factors"]["count"],
        "factor_kinds": shadow["factors"]["by_kind"],
        "gap_ids": [gap["gap_id"] for gap in shadow["gaps"]],
        "unmapped_nonempty_fields": shadow["coverage"]["unmapped_nonempty_fields"],
        "canonical_sha256": shadow["canonical_sha256"],
    }


def build_canonical_factor_shadow_summary(result_name: str = "benchmark_vlm_qwen") -> dict[str, object]:
    return {
        "schema_version": 1,
        "cases": {
            case_name: factor_shadow_summary(
                case_name,
                REPO / "samples_known_object" / directory / "results" / result_name,
            )
            for case_name, directory in CANONICAL_CASE_DIRECTORIES.items()
        },
    }


def verify_factor_shadow_summary(
    expected_path: Path = DEFAULT_FACTOR_SHADOW_GOLDEN,
    *,
    result_name: str = "benchmark_vlm_qwen",
) -> list[str]:
    expected = json.loads(expected_path.read_text())
    actual = build_canonical_factor_shadow_summary(result_name=result_name)
    errors: list[str] = []
    for case_name, expected_case in expected.get("cases", {}).items():
        actual_case = actual.get("cases", {}).get(case_name)
        if actual_case is None:
            errors.append(f"{case_name}: missing actual factor shadow summary")
            continue
        for key, expected_value in expected_case.items():
            actual_value = actual_case.get(key)
            if actual_value != expected_value:
                errors.append(f"{case_name}:{key}: expected {expected_value!r}, got {actual_value!r}")
    for case_name in set(actual.get("cases", {})) - set(expected.get("cases", {})):
        errors.append(f"{case_name}: unexpected actual factor shadow summary")
    return errors
