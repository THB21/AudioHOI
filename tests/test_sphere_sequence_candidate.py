from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.shared.generic_contact_pipeline.core.base.config import CaseProfile
from scripts.shared.generic_contact_pipeline.core.contact_constraints import (
    ContactMode,
    adapt_contact_event_rows,
    adapt_contact_state_rows,
)
from scripts.shared.generic_contact_pipeline.core.human_sites import adapt_human_site_rows
from scripts.shared.generic_contact_pipeline.core.provenance.attempts import STAGE_ARTIFACT_KEYS
from scripts.shared.generic_contact_pipeline.core.solver import (
    SPHERE_ATTEMPT_NAME,
    SPHERE_CANDIDATE_NAME,
    SPHERE_RESIDUAL_NAME,
    solve_sphere_sequence_candidate,
)


REPO = Path(__file__).resolve().parents[1]


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize(
    ("case", "directory", "human_events", "support_events"),
    [("basketball", "01_basketball", 15, 15), ("football", "10_football", 6, 9)],
)
def test_ball_contact_event_timeline_is_typed_and_result_owned(
    case: str,
    directory: str,
    human_events: int,
    support_events: int,
) -> None:
    result_dir = REPO / "samples_known_object" / directory / "results/benchmark_vlm_qwen"
    event_path = result_dir / "contact_candidates_internal/contact_candidates_labeled.csv"
    state_path = result_dir / "contact_state_frames.csv"
    events = adapt_contact_event_rows(case, _read(event_path), str(event_path.relative_to(REPO)))
    states = adapt_contact_state_rows(case, _read(state_path), str(state_path.relative_to(REPO)))
    assert len([event for event in events if event.mode != ContactMode.SUPPORT]) == human_events
    assert len([event for event in events if event.mode == ContactMode.SUPPORT]) == support_events
    assert all(event.interval.start_frame <= event.peak_frame <= event.interval.end_frame for event in events)
    assert len(states) in {192, 242}
    assert any(state.human_active for state in states)
    assert any(state.support_active for state in states)


def test_human_site_adapter_keeps_camera_xyz_and_semantic_site() -> None:
    rows = [
        {
            "frame": "1", "time": "0.000000", "site_id": "left_hand", "body_part": "hand", "side": "left",
            "x_m": "0.100000000", "y_m": "0.200000000", "z_m": "3.400000000",
            "coordinate_frame": "gvhmr_incam", "confidence": "1.000000", "source": "gvhmr_smplx_contact_site_provider",
        }
    ]
    result = adapt_human_site_rows("sample", rows, "human_sites.csv")
    assert result.schema == "human_site_xyz_v1"
    assert result.unmapped_nonempty_fields == ()
    measurement = result.measurements[0]
    assert measurement.site.body_part == "hand" and measurement.site.side == "left"
    assert measurement.xyz_m == (0.1, 0.2, 3.4)


def test_sphere_candidate_solves_only_to_safe_sandbox_artifacts(tmp_path: Path) -> None:
    pytest.importorskip("scipy")
    result_dir = tmp_path / "result"
    candidate_dir = tmp_path / "candidate"
    observation_rows = []
    state_rows = []
    human_rows = []
    for frame in range(1, 6):
        time = (frame - 1) / 24.0
        observation_rows.append(
            {
                "frame": frame, "time": f"{time:.6f}", "ref_u": "640.000", "ref_v": f"{350 + frame:.3f}",
                "ref_u_smooth": "640.000", "ref_v_smooth": f"{350 + frame:.3f}",
                "support_u": "640.000", "support_v": "520.000", "object_ref_depth_m": "3.000000",
                "observation_conf": "1.000000", "support_conf": "1.000000", "depth_conf": "1.000000",
            }
        )
        active = frame in {2, 4}
        state_rows.append(
            {
                "frame": frame, "time": f"{time:.6f}", "human_contact_state": int(active),
                "floor_contact_state": 0, "contact_label": "right_hand", "contact_depth_offset_m": "0.050000",
                "active_object_u": "640.000", "active_object_v": f"{350 + frame:.3f}",
                "anchor_score": "0.800000", "support_conf": "0.000000",
            }
        )
        for site_id, body_part, side, z in (("left_hand", "hand", "left", 3.4), ("right_hand", "hand", "right", 3.5), ("left_foot", "foot", "left", 3.8), ("right_foot", "foot", "right", 3.8)):
            human_rows.append(
                {
                    "frame": frame, "time": f"{time:.6f}", "site_id": site_id, "body_part": body_part, "side": side,
                    "x_m": "0.000000000", "y_m": "0.000000000", "z_m": f"{z:.9f}",
                    "coordinate_frame": "gvhmr_incam", "confidence": "1.000000", "source": "synthetic_human_site",
                }
            )
    event_rows = [
        {
            "frame": frame, "window_start": frame, "window_end": frame, "time": f"{(frame - 1) / 24.0:.6f}",
            "contact_type": "anchor_contact_event", "target": "right_hand", "score": "0.800000", "confidence_level": "strong",
        }
        for frame in (2, 4)
    ]
    _write(result_dir / "object_observations.csv", observation_rows)
    _write(result_dir / "contact_state_frames.csv", state_rows)
    _write(result_dir / "contact_events.csv", event_rows)
    _write(result_dir / "human_sites.csv", human_rows)
    (result_dir / "support_geometry.json").write_text(json.dumps({"support_type": "floor", "floor_v": 520.0, "source": "synthetic", "confidence": 1.0}))
    profile = CaseProfile(
        {
            "case_name": "synthetic_ball", "sample_dir": str(tmp_path), "result_name": "result",
            "geometry_model": "sphere_proxy", "pose_model": "translation3", "sphere": {"radius_m": 0.12},
            "camera": {"fx": 1468.604736328125, "fy": 1468.604736328125, "cx": 640.0, "cy": 360.0},
        }
    )
    attempt = solve_sphere_sequence_candidate(
        profile,
        result_dir,
        contact_events_csv=result_dir / "contact_events.csv",
        human_sites_csv=result_dir / "human_sites.csv",
        support_geometry_json=result_dir / "support_geometry.json",
        candidate_dir=candidate_dir,
    )
    assert attempt["solver_executed"] is True
    assert attempt["accepted_outputs_written"] is False
    assert attempt["baseline_pose_read"] is False
    assert {path.name for path in candidate_dir.iterdir()} == {SPHERE_CANDIDATE_NAME, SPHERE_RESIDUAL_NAME, SPHERE_ATTEMPT_NAME}
    assert not (result_dir / "object_pose.csv").exists()
    assert len(_read(candidate_dir / SPHERE_CANDIDATE_NAME)) == 5


def test_sphere_solver_source_has_no_pose_or_legacy_event_input() -> None:
    source = (REPO / "scripts/shared/generic_contact_pipeline/core/solver/sphere_sequence.py").read_text()
    anchor_policy = (REPO / "scripts/shared/generic_contact_pipeline/components/refinement/policies/anchor_depth.py").read_text()
    assert "object_pose_init.csv" not in source
    assert "object_pose.csv" not in source
    assert "contact_candidates_object_proxy/contact_candidates_labeled.csv" not in anchor_policy


def test_typed_sphere_inputs_and_candidate_outputs_are_stage_attempt_artifacts() -> None:
    assert {"contact_events", "human_sites", "support_geometry"}.issubset(STAGE_ARTIFACT_KEYS["stage2"])
    assert {"sphere_candidate", "sphere_residuals", "sphere_attempt"}.issubset(STAGE_ARTIFACT_KEYS["stage4"])
