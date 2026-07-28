#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

if __package__ in {None, ""}:
    REPO = Path(__file__).resolve().parents[4]
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))

from scripts.shared.generic_contact_pipeline.components.geometry.mug_periodic import (  # noqa: E402
    MugPeriodicGeometryProvider,
    adapt_mug_periodic_observations,
    load_camera_matrices,
)
from scripts.shared.generic_contact_pipeline.core.base.io import repo_relative_value, write_json  # noqa: E402
from scripts.shared.generic_contact_pipeline.core.solver.projected_periodic_sequence import (  # noqa: E402
    ProjectedPeriodicParameters,
    solve_projected_periodic_sequence,
)


BODY_CANDIDATE_NAME = "generic_periodic_body_candidate.csv"
PHASE_CANDIDATE_NAME = "generic_periodic_phase_candidate.csv"
ATTEMPT_NAME = "generic_projected_periodic_attempt.json"


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve an isolated projected-root + periodic-feature candidate.")
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--observations-csv", type=Path, required=True)
    parser.add_argument("--proxy-csv", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    args = parser.parse_args()
    sample_dir = args.sample_dir.resolve()
    observations_csv = args.observations_csv.resolve()
    proxy_csv = args.proxy_csv.resolve()
    candidate_dir = args.candidate_dir.resolve()
    candidate_dir.mkdir(parents=True, exist_ok=True)

    observation_rows = _read(observations_csv)
    proxy_rows = {int(float(row["frame"])): row for row in _read(proxy_csv)}
    observations = adapt_mug_periodic_observations(observation_rows, proxy_rows)
    cameras_array = load_camera_matrices(sample_dir)
    cameras = {observation.frame: cameras_array[observation.frame - 1] for observation in observations}
    geometry = MugPeriodicGeometryProvider(sample_dir / "articraft/materialized_mug_mesh")
    parameters = ProjectedPeriodicParameters()
    solution = solve_projected_periodic_sequence(observations, cameras, geometry, parameters)

    body_path = candidate_dir / BODY_CANDIDATE_NAME
    body_fields = ["frame", "time", "x", "y", "z", "yaw", "yaw_deg", "pitch", "pitch_deg", "roll", "roll_deg", "scale", "source"]
    body_rows = []
    for observation in observations:
        root = solution.root_states[observation.frame]
        body_rows.append(
            {
                "frame": observation.frame,
                "time": f"{observation.time:.6f}",
                "x": f"{root[0]:.9f}", "y": f"{root[1]:.9f}", "z": f"{root[2]:.9f}",
                "yaw": "0.000000000", "yaw_deg": "0.000000",
                "pitch": f"{root[4]:.9f}", "pitch_deg": f"{math.degrees(root[4]):.6f}",
                "roll": f"{root[5]:.9f}", "roll_deg": f"{math.degrees(root[5]):.6f}",
                "scale": f"{root[6]:.9f}",
                "source": "observation_bbox_da3_fit_axial_gauge_yaw_zero",
            }
        )
    _write_csv(body_path, body_fields, body_rows)

    phase_path = candidate_dir / PHASE_CANDIDATE_NAME
    phase_fields = ["frame", "time", "m17_phase_rad", "m17_phase_deg", "m43_phase_rad", "m43_phase_deg", "vlm_visibility", "source"]
    phase_rows = []
    for observation in observations:
        phase = solution.periodic_phase_rad[observation.frame]
        phase_rows.append(
            {
                "frame": observation.frame,
                "time": f"{observation.time:.6f}",
                "m17_phase_rad": f"{phase:.9f}", "m17_phase_deg": f"{math.degrees(phase):.6f}",
                "m43_phase_rad": f"{phase:.9f}", "m43_phase_deg": f"{math.degrees(phase):.6f}",
                "vlm_visibility": "visible" if observation.periodic_feature_visible else "hidden",
                "source": "observed_handle_center" if observation.periodic_feature_visible else "interpolated_hidden_handle_span",
            }
        )
    _write_csv(phase_path, phase_fields, phase_rows)

    attempt_core = {
        "state_spec": solution.report["state_spec"],
        "kinematic_contract": solution.report["kinematic_contract"],
        "geometry_provider": solution.report["geometry_provider"],
        "parameters": asdict(parameters),
        "inputs": {
            "observations": {"path": str(repo_relative_value(observations_csv)), "sha256": _sha256(observations_csv)},
            "proxy_depth": {"path": str(repo_relative_value(proxy_csv)), "sha256": _sha256(proxy_csv)},
        },
        "body_candidate_sha256": _sha256(body_path),
        "phase_candidate_sha256": _sha256(phase_path),
    }
    canonical = hashlib.sha256(json.dumps(attempt_core, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    attempt = {
        "schema_version": 1,
        "mode": "generic_projected_periodic_candidate",
        "attempt_id": f"periodic-{canonical[:12]}",
        "solver_executed": True,
        "accepted_outputs_written": False,
        "baseline_pose_read": False,
        "historical_phase_read": False,
        "executor_scope": "isolated_candidate_dir",
        "candidate_dir": str(repo_relative_value(candidate_dir)),
        **attempt_core,
        "frames": solution.report["rows"],
        "body_fit_success_frames": solution.report["body_fit_success_frames"],
        "body_residual_rms_median": solution.report["body_residual_rms_median"],
        "body_residual_rms_p90": solution.report["body_residual_rms_p90"],
        "phase": solution.report["phase"],
        "body_candidate_artifact": BODY_CANDIDATE_NAME,
        "phase_candidate_artifact": PHASE_CANDIDATE_NAME,
        "canonical_sha256": canonical,
    }
    write_json(candidate_dir / ATTEMPT_NAME, attempt)
    print(json.dumps(attempt, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
