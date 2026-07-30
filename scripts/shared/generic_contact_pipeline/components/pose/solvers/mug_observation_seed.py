#!/usr/bin/env python3
"""Build a mug body pose and axial phase from declared image/depth observations."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[6]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.shared.generic_contact_pipeline.components.geometry.mug_periodic import (  # noqa: E402
    MugPeriodicGeometryProvider,
    adapt_mug_periodic_observations,
    load_camera_matrices,
)
from scripts.shared.generic_contact_pipeline.core.solver.projected_periodic_sequence import (  # noqa: E402
    ProjectedPeriodicParameters,
    solve_projected_periodic_sequence,
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(
    *,
    sample_dir: Path,
    observations_csv: Path,
    proxy_csv: Path,
    out_dir: Path,
    max_nfev: int = 80,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    observation_rows = _read_csv(observations_csv)
    proxy_rows = {int(float(row["frame"])): row for row in _read_csv(proxy_csv)}
    observations = adapt_mug_periodic_observations(observation_rows, proxy_rows)
    if not observations:
        raise RuntimeError("Mug observation seed has no common observation/depth frames")
    observations_by_frame = {int(float(row["frame"])): row for row in observation_rows}
    cameras_array = load_camera_matrices(sample_dir)
    cameras = {observation.frame: cameras_array[observation.frame - 1] for observation in observations}
    mesh_root = sample_dir / "articraft/materialized_mug_mesh"
    geometry = MugPeriodicGeometryProvider(mesh_root)
    parameters = ProjectedPeriodicParameters(max_function_evaluations=max_nfev)
    solution = solve_projected_periodic_sequence(observations, cameras, geometry, parameters)

    pose_path = out_dir / "body_pose.csv"
    pose_fields = [
        "frame", "time", "x", "y", "z", "yaw", "yaw_deg", "pitch", "pitch_deg",
        "roll", "roll_deg", "scale", "source",
    ]
    with pose_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=pose_fields)
        writer.writeheader()
        for observation in observations:
            pose = solution.root_states[observation.frame]
            writer.writerow(
                {
                    "frame": observation.frame,
                    "time": observations_by_frame[observation.frame].get(
                        "time", f"{(observation.frame - 1) / 24.0:.6f}"
                    ),
                    "x": f"{pose[0]:.9f}", "y": f"{pose[1]:.9f}", "z": f"{pose[2]:.9f}",
                    "yaw": "0.000000000", "yaw_deg": "0.000000",
                    "pitch": f"{pose[4]:.9f}", "pitch_deg": f"{math.degrees(pose[4]):.6f}",
                    "roll": f"{pose[5]:.9f}", "roll_deg": f"{math.degrees(pose[5]):.6f}",
                    "scale": f"{pose[6]:.9f}",
                    "source": "observation_bbox_da3_fit_axial_gauge_yaw_zero",
                }
            )

    phase_path = out_dir / "axial_phase.csv"
    phase_fields = [
        "frame", "time", "m17_phase_rad", "m17_phase_deg", "m43_phase_rad", "m43_phase_deg",
        "vlm_visibility", "source",
    ]
    with phase_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=phase_fields)
        writer.writeheader()
        for observation in observations:
            phase = solution.periodic_phase_rad[observation.frame]
            writer.writerow(
                {
                    "frame": observation.frame,
                    "time": observations_by_frame[observation.frame].get(
                        "time", f"{(observation.frame - 1) / 24.0:.6f}"
                    ),
                    "m17_phase_rad": f"{phase:.9f}", "m17_phase_deg": f"{math.degrees(phase):.6f}",
                    "m43_phase_rad": f"{phase:.9f}", "m43_phase_deg": f"{math.degrees(phase):.6f}",
                    "vlm_visibility": "visible" if observation.periodic_feature_visible else "hidden",
                    "source": (
                        "observed_handle_center"
                        if observation.periodic_feature_visible
                        else "interpolated_hidden_handle_span"
                    ),
                }
            )

    report = {
        "schema_version": 1,
        "policy": "observation_derived_body_pose_and_axial_phase",
        "body_pose_csv": str(pose_path),
        "phase_source": str(phase_path),
        "observations_csv": str(observations_csv),
        "proxy_depth_csv": str(proxy_csv),
        "mesh_root": str(mesh_root),
        "rows": solution.report["rows"],
        "body_fit_success_frames": solution.report["body_fit_success_frames"],
        "body_residual_rms_median": solution.report["body_residual_rms_median"],
        "body_residual_rms_p90": solution.report["body_residual_rms_p90"],
        "phase": solution.report["phase"],
        "inputs": {
            "observations_sha256": _sha256(observations_csv),
            "proxy_depth_sha256": _sha256(proxy_csv),
        },
        "historical_solved_seed_used": False,
    }
    report_path = out_dir / "observation_seed_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return {**report, "report": str(report_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--observations-csv", type=Path, required=True)
    parser.add_argument("--proxy-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-nfev", type=int, default=80)
    args = parser.parse_args()
    report = build(
        sample_dir=args.sample_dir.resolve(),
        observations_csv=args.observations_csv.resolve(),
        proxy_csv=args.proxy_csv.resolve(),
        out_dir=args.out_dir.resolve(),
        max_nfev=args.max_nfev,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
