from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.shared.generic_contact_pipeline.core.interaction import (
    ContactStateAxis,
    MotionMode,
    VisibilityState,
    build_interaction_timeline,
    validate_frame_interaction_state,
)


REPO = Path(__file__).resolve().parents[1]
CASE_DIRECTORIES = {
    "basketball": "01_basketball",
    "football": "10_football",
    "mug": "02_mug",
    "chair": "05_chair",
    "stick": "11_stick",
}
EXPECTED_FRAME_COUNTS = {
    "basketball": 192,
    "football": 242,
    "mug": 240,
    "chair": 192,
    "stick": 240,
}


@pytest.mark.parametrize("case_name", sorted(CASE_DIRECTORIES))
def test_interaction_timeline_covers_five_cases_without_final_pose_reads(case_name: str) -> None:
    result_dir = REPO / "samples_known_object" / CASE_DIRECTORIES[case_name] / "results/benchmark_vlm_qwen"
    timeline = build_interaction_timeline(case_name, result_dir)

    assert len(timeline.frames) == EXPECTED_FRAME_COUNTS[case_name]
    assert timeline.metrics["final_pose_read"] is False
    assert timeline.metrics["frame_count"] == EXPECTED_FRAME_COUNTS[case_name]
    assert all(validate_frame_interaction_state(frame) == [] for frame in timeline.frames)


def test_interaction_state_axes_do_not_merge_occlusion_contact_and_motion() -> None:
    result_dir = REPO / "samples_known_object/11_stick/results/benchmark_vlm_qwen"
    timeline = build_interaction_timeline("stick", result_dir)

    first = timeline.frames[0]
    assert first.visibility_state in {VisibilityState.VISIBLE, VisibilityState.UNKNOWN}
    assert first.contact_state == ContactStateAxis.ACTIVE
    assert first.motion_mode == MotionMode.ATTACHED
    assert first.active_contact_ids


def test_interaction_timeline_export_cli_writes_jsonl_intervals_and_metrics(tmp_path: Path) -> None:
    out_dir = tmp_path / "interaction"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/shared/generic_contact_pipeline/tools/export_interaction_timeline.py",
            "--case",
            "basketball",
            "--result-dir",
            "samples_known_object/01_basketball/results/benchmark_vlm_qwen",
            "--out-dir",
            str(out_dir),
        ],
        cwd=REPO,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "basketball: interaction_timeline frames=192" in completed.stdout
    jsonl = out_dir / "interaction_timeline.jsonl"
    intervals = out_dir / "interaction_intervals.csv"
    metrics = out_dir / "interaction_state_metrics.json"
    assert jsonl.exists()
    assert intervals.exists()
    assert metrics.exists()
    first = json.loads(jsonl.read_text().splitlines()[0])
    assert first["target_entity_id"] == "target_object"
    with intervals.open(newline="") as handle:
        assert list(csv.DictReader(handle))
    assert json.loads(metrics.read_text())["frame_count"] == 192
