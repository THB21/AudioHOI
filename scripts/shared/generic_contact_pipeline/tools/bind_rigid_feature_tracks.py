#!/usr/bin/env python3
"""Bind persistent CoTracker rows to descriptor-declared rigid feature points."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.shared.generic_contact_pipeline.core.base.config import CaseProfile, load_case_profile, with_runtime_overrides
from scripts.shared.generic_contact_pipeline.core.base.io import REPO, repo_relative_value
from scripts.shared.generic_contact_pipeline.core.measurements.rigid_feature_tracks import bind_rigid_feature_tracks
from scripts.shared.generic_contact_pipeline.core.solver.capability_production_problem import prepare_capability_object_problem
from scripts.shared.generic_contact_pipeline.core.state import PinholeCamera
from scripts.shared.generic_contact_pipeline.core.state.asset_geometry import build_rigid_geometry_from_asset_descriptor
from scripts.shared.generic_contact_pipeline.core.state.asset_state_contract import build_asset_state_contract


FIELDNAMES = (
    "frame",
    "time",
    "u",
    "v",
    "geometry_feature_id",
    "semantic_role",
    "track_id",
    "confidence",
    "source_anchor_frames",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def _atomic_csv(path: Path, rows: tuple[dict[str, object], ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def _initializer_only_profile(profile: CaseProfile) -> CaseProfile:
    data = deepcopy(profile.data)
    configured = data.get("supplemental_measurements", ())
    data["supplemental_measurements"] = [
        dict(spec)
        for spec in configured
        if str(spec.get("adapter", "")) != "rigid_feature_points_v1"
    ]
    return CaseProfile(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--result-name", required=True)
    parser.add_argument("--track-artifact", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--maximum-anchor-error-px", type=float, required=True)
    parser.add_argument("--minimum-track-visibility", type=float, required=True)
    parser.add_argument("--pose-hypotheses", type=Path)
    parser.add_argument("--minimum-pose-mask-iou", type=float, default=0.60)
    parser.add_argument(
        "--body-models-root",
        type=Path,
        default=REPO / "third-party/GVHMR/inputs/checkpoints/body_models",
    )
    args = parser.parse_args()

    profile = with_runtime_overrides(load_case_profile(args.case), result_name=args.result_name)
    initializer_profile = _initializer_only_profile(profile)
    descriptor_path = REPO / str(profile.data["geometry_asset_descriptor"])
    descriptor = json.loads(descriptor_path.read_text())
    feature_ids = tuple(
        str(value)
        for value in descriptor.get("visual_tracking_features", {}).get(
            "point_feature_ids", ()
        )
    )
    if not feature_ids:
        raise ValueError("asset descriptor has no visual_tracking_features.point_feature_ids")
    contract = build_asset_state_contract(descriptor_path, REPO)
    geometry = build_rigid_geometry_from_asset_descriptor(
        descriptor_path=descriptor_path,
        repository_root=REPO,
        state_spec=contract.state_spec,
    )
    prepared = prepare_capability_object_problem(
        profile=initializer_profile,
        result_dir=initializer_profile.result_dir,
        repository_root=REPO,
        body_models_root=args.body_models_root,
        factor_arbitration_mode="off",
    )
    if prepared.baseline_pose_read or prepared.human_state_optimized:
        raise ValueError("rigid feature binding requires object-only observation initialization")
    problem = prepared.preparation.problem
    initializer_states = dict(zip(problem.frames, problem.initial_states))
    pose_hypotheses = args.pose_hypotheses or (
        profile.sample_dir / "results/megapose/rigid_pose_hypotheses.jsonl"
    )
    selected_pose_rows: dict[int, dict[str, object]] = {}
    if pose_hypotheses.is_file():
        for line in pose_hypotheses.read_text().splitlines():
            row = json.loads(line)
            if (
                bool(row.get("selected_by_visual_geometry", False))
                and str(row.get("provider_status", "")) == "reliable_visible_keyframe"
                and float(row.get("official_render_mask_iou", 0.0))
                >= args.minimum_pose_mask_iou
            ):
                selected_pose_rows[int(row["frame"])] = row
    states_by_frame = {
        frame: (
            float(row["tx_m"]),
            float(row["ty_m"]),
            float(row["tz_m"]),
            float(row["qw"]),
            float(row["qx"]),
            float(row["qy"]),
            float(row["qz"]),
        )
        for frame, row in selected_pose_rows.items()
    }
    if not states_by_frame:
        states_by_frame = initializer_states
    cameras = {frame: PinholeCamera(**profile.camera) for frame in states_by_frame}
    configured_anchor_frames = {
        int(value) for value in profile.data.get("preprocess", {}).get("rigid_pose_keyframes", ())
    }
    reliable_anchor_frames = tuple(
        sorted(frame for frame in states_by_frame if frame in configured_anchor_frames)
    )
    with args.track_artifact.open(newline="") as handle:
        track_rows = list(csv.DictReader(handle))
    binding = bind_rigid_feature_tracks(
        track_rows=track_rows,
        states_by_frame=states_by_frame,
        cameras=cameras,
        geometry_provider=geometry.provider,
        feature_ids=feature_ids,
        reliable_anchor_frames=reliable_anchor_frames,
        maximum_anchor_error_px=args.maximum_anchor_error_px,
        minimum_track_visibility=args.minimum_track_visibility,
    )
    if not binding.measurement_rows:
        raise ValueError("no rigid feature tracks passed descriptor association")

    _atomic_csv(args.output_csv, binding.measurement_rows)
    role_counts: dict[str, int] = {}
    for row in binding.measurement_rows:
        role = str(row["semantic_role"])
        role_counts[role] = role_counts.get(role, 0) + 1
    manifest = {
        "schema_version": 1,
        "producer": "descriptor_rigid_feature_track_binding",
        "case_name": profile.case_name,
        "result_name": profile.result_name,
        "track_artifact": str(repo_relative_value(args.track_artifact)),
        "track_artifact_sha256": _sha256(args.track_artifact),
        "asset_descriptor": str(repo_relative_value(descriptor_path)),
        "asset_descriptor_sha256": _sha256(descriptor_path),
        "initializer_input_sha256": prepared.initializer_input_sha256,
        "anchor_pose_artifact": (
            str(repo_relative_value(pose_hypotheses)) if pose_hypotheses.is_file() else None
        ),
        "anchor_pose_artifact_sha256": (
            _sha256(pose_hypotheses) if pose_hypotheses.is_file() else None
        ),
        "anchor_pose_source": (
            "external_megapose_selected_visual_geometry"
            if selected_pose_rows
            else "generic_capability_initializer_fallback"
        ),
        "minimum_pose_mask_iou": args.minimum_pose_mask_iou,
        "output_csv": str(repo_relative_value(args.output_csv)),
        "output_csv_sha256": _sha256(args.output_csv),
        "association_count": len(binding.associations),
        "measurement_count": len(binding.measurement_rows),
        "feature_coverage": sorted(
            association.geometry_feature_id for association in binding.associations
        ),
        "semantic_role_rows": dict(sorted(role_counts.items())),
        "reliable_anchor_frames": list(binding.reliable_anchor_frames),
        "associations": [
            {
                "track_id": association.track_id,
                "geometry_feature_id": association.geometry_feature_id,
                "anchor_frame": association.anchor_frame,
                "anchor_error_px": association.anchor_error_px,
                "confidence": association.confidence,
                "source_anchor_frames": list(association.source_anchor_frames),
            }
            for association in binding.associations
        ],
        "rejected_by_reason": dict(binding.rejected_by_reason),
        "baseline_pose_read": False,
        "human_state_optimized": False,
        "case_dispatch_used": False,
    }
    _atomic_json(args.output_manifest, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
