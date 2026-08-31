#!/usr/bin/env python3
"""Validate the compact AudioHOI numeric results release."""

from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / "results_release"
EXPECTED_CASES = {
    "basketball", "football", "mug", "chair", "stick",
    "back_view_basketball", "volleyball", "pingpong", "suitcase",
}
CHALLENGE_CASE_MAP = {
    "back_view_basketball": "back_view_basketball",
    "volleyball": "volleyball",
    "pingpong_wall": "pingpong",
    "suitcase_drag": "suitcase",
}
EXPECTED_VARIANTS = {"full", "no_audio", "no_vlm", "vision_only"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return [
            {(key or "").strip(): (value or "").strip() for key, value in row.items()}
            for row in reader
        ]


def validate_pose(entry: dict[str, str]) -> None:
    path = ROOT / entry["path"]
    assert path.is_file(), f"missing pose: {path}"
    assert sha256(path) == entry["sha256"], f"pose hash mismatch: {path}"
    pose_rows = rows(path)
    assert len(pose_rows) == int(entry["frames"]), f"frame count mismatch: {path}"
    assert [int(float(row["frame"])) for row in pose_rows] == list(
        range(1, len(pose_rows) + 1)
    ), (
        f"non-contiguous frames: {path}"
    )
    for row in pose_rows:
        for field in ("tx", "ty", "tz"):
            assert field in row and math.isfinite(float(row[field])), f"invalid {field}: {path}"
        if {"qw", "qx", "qy", "qz"} <= row.keys():
            quat = [float(row[name]) for name in ("qw", "qx", "qy", "qz")]
            assert all(math.isfinite(value) for value in quat), f"invalid quaternion: {path}"
            norm = math.sqrt(sum(value * value for value in quat))
            assert abs(norm - 1.0) < 5e-3, f"unnormalized quaternion ({norm}): {path}"


def main() -> None:
    pose_manifest = rows(RELEASE / "manifests" / "pose_manifest.csv")
    assert {row["case"] for row in pose_manifest} == EXPECTED_CASES
    for entry in pose_manifest:
        validate_pose(entry)

    main_metrics = rows(RELEASE / "metrics" / "main_object_metrics.csv")
    metric_cases = {CHALLENGE_CASE_MAP.get(row["case"], row["case"]) for row in main_metrics}
    assert metric_cases == EXPECTED_CASES, f"metric coverage mismatch: {metric_cases}"

    ablations = rows(RELEASE / "ablations" / "multimodal_ablation_metrics.csv")
    matrix: dict[str, set[str]] = {}
    for row in ablations:
        case = CHALLENGE_CASE_MAP.get(row["case"], row["case"])
        matrix.setdefault(case, set()).add(row["variant"])
    assert set(matrix) == set(CHALLENGE_CASE_MAP.values())
    for case, variants in matrix.items():
        assert variants == EXPECTED_VARIANTS, f"incomplete ablation matrix for {case}: {variants}"

    loss_rows = rows(RELEASE / "ablations" / "objective_loss_ablation_comparison.csv")
    assert loss_rows, "loss-removal comparison is empty"

    sums_path = RELEASE / "SHA256SUMS"
    if sums_path.exists():
        for line in sums_path.read_text(encoding="utf-8").splitlines():
            expected, relative = line.split("  ", 1)
            target = RELEASE / relative
            assert target.is_file(), f"missing checksummed file: {relative}"
            assert sha256(target) == expected, f"checksum mismatch: {relative}"

    print(
        f"results_release valid: {len(pose_manifest)} poses, "
        f"{len(main_metrics)} metric rows, {len(ablations)} multimodal rows, "
        f"{len(loss_rows)} loss-removal rows"
    )


if __name__ == "__main__":
    main()
