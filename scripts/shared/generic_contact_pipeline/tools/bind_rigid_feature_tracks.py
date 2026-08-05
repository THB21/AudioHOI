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
from pathlib import Path

import numpy as np
import cv2
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.shared.generic_contact_pipeline.core.base.config import load_case_profile, with_runtime_overrides
from scripts.shared.generic_contact_pipeline.core.base.io import REPO, repo_relative_value
from scripts.shared.generic_contact_pipeline.core.measurements.rigid_feature_tracks import bind_rigid_feature_tracks
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
    "local_x",
    "local_y",
    "local_z",
    "source_anchor_frames",
)


def _upright_normalized_state(
    row: dict[str, object],
    initializer: dict[str, object],
    camera: PinholeCamera,
    body_points: np.ndarray,
    target_body_bbox: np.ndarray,
) -> tuple[float, ...]:
    """Resolve the common cuboid pitch/roll symmetry before surface binding."""

    upright_local = np.asarray(initializer["upright_axis_local"], dtype=float)
    upright_camera = np.asarray(initializer["preferred_upright_camera"], dtype=float)
    heading_local = np.asarray(initializer.get("heading_axis_local", (1.0, 0.0, 0.0)), dtype=float)
    upright_local /= np.linalg.norm(upright_local)
    upright_camera /= np.linalg.norm(upright_camera)
    heading_local -= upright_local * float(heading_local @ upright_local)
    heading_local /= np.linalg.norm(heading_local)
    side_local = np.cross(upright_local, heading_local)
    raw_rotation = Rotation.from_quat(
        [float(row[key]) for key in ("qx", "qy", "qz", "qw")]
    ).as_matrix()
    heading_camera = raw_rotation @ heading_local
    heading_camera -= upright_camera * float(heading_camera @ upright_camera)
    if np.linalg.norm(heading_camera) <= 1e-8:
        heading_camera = np.asarray((1.0, 0.0, 0.0), dtype=float)
        heading_camera -= upright_camera * float(heading_camera @ upright_camera)
    heading_camera /= np.linalg.norm(heading_camera)
    side_camera = np.cross(upright_camera, heading_camera)
    corrected = (
        np.column_stack((heading_camera, side_camera, upright_camera))
        @ np.column_stack((heading_local, side_local, upright_local)).T
    )
    body_center = np.mean(body_points, axis=0)
    target_center = 0.5 * (target_body_bbox[:2] + target_body_bbox[2:])
    target_size = np.maximum(target_body_bbox[2:] - target_body_bbox[:2], 1.0)
    best: tuple[float, np.ndarray] | None = None
    for depth in np.linspace(1.0, 6.0, 501):
        target_center_camera = np.asarray(
            (
                (target_center[0] - camera.cx) * depth / camera.fx,
                (target_center[1] - camera.cy) * depth / camera.fy,
                depth,
            ),
            dtype=float,
        )
        translation = target_center_camera - corrected @ body_center
        world = body_points @ corrected.T + translation
        projected = camera.project(world)
        bbox = np.asarray(
            (projected[:, 0].min(), projected[:, 1].min(), projected[:, 0].max(), projected[:, 1].max())
        )
        score = float(np.sum(((bbox - target_body_bbox) / np.tile(target_size, 2)) ** 2))
        if best is None or score < best[0]:
            best = (score, translation)
    assert best is not None
    translation = best[1]
    qx, qy, qz, qw = Rotation.from_matrix(corrected).as_quat()
    return (
        float(translation[0]), float(translation[1]), float(translation[2]),
        float(qw), float(qx), float(qy), float(qz),
    )


def _main_body_bbox(mask_path: Path, minimum_row_width_ratio: float = 0.45) -> np.ndarray:
    """Exclude thin rails and retain the dense suitcase-body silhouette."""

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(mask_path)
    binary = mask > 0
    widths = binary.sum(axis=1)
    valid_rows = np.flatnonzero(widths >= max(1.0, minimum_row_width_ratio * widths.max()))
    if len(valid_rows) < 2:
        raise ValueError(f"object mask has no dense main-body rows: {mask_path}")
    ys, xs = np.nonzero(binary[valid_rows])
    actual_y = valid_rows[ys]
    return np.asarray((xs.min(), actual_y.min(), xs.max(), actual_y.max()), dtype=float)


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


def _gate_tracks_by_object_mask(
    rows: list[dict[str, str]], mask_dir: Path, dilation_px: int = 5
) -> tuple[list[dict[str, str]], int]:
    """Invalidate tracker visibility when its point has left the object mask."""

    kernel = np.ones((dilation_px, dilation_px), dtype=np.uint8)
    masks: dict[int, np.ndarray] = {}
    gated: list[dict[str, str]] = []
    rejected = 0
    for source in rows:
        row = dict(source)
        if float(row.get("visible", "1")) < 0.5:
            gated.append(row)
            continue
        frame = int(row["frame"])
        if frame not in masks:
            mask = cv2.imread(str(mask_dir / f"{frame:05d}_mask.png"), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(mask_dir / f"{frame:05d}_mask.png")
            masks[frame] = cv2.dilate((mask > 0).astype(np.uint8), kernel)
        x = int(round(float(row.get("x", row.get("u", "nan")))))
        y = int(round(float(row.get("y", row.get("v", "nan")))))
        mask = masks[frame]
        if not (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x] > 0):
            row["visible"] = "0"
            rejected += 1
        gated.append(row)
    return gated, rejected


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
    args = parser.parse_args()

    profile = with_runtime_overrides(load_case_profile(args.case), result_name=args.result_name)
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
    initializer = dict(descriptor.get("initializer", {}))
    camera = PinholeCamera(**profile.camera)
    body_feature_id = str(initializer.get("body_feature_id", "object:body"))
    body_points = np.asarray(descriptor["feature_points"][body_feature_id], dtype=float)
    states_by_frame = {
        frame: _upright_normalized_state(
            row,
            initializer,
            camera,
            body_points,
            _main_body_bbox(
                profile.sample_dir / "results/segmentation/masks" / f"{frame:05d}_mask.png"
            ),
        )
        for frame, row in selected_pose_rows.items()
    }
    if not states_by_frame:
        raise ValueError(
            "no reliable external rigid pose anchor passes the configured render-mask IoU gate"
        )
    cameras = {frame: camera for frame in states_by_frame}
    configured_anchor_frames = {
        int(value) for value in profile.data.get("preprocess", {}).get("rigid_pose_keyframes", ())
    }
    reliable_anchor_frames = tuple(
        sorted(
            (frame for frame in states_by_frame if frame in configured_anchor_frames),
            key=lambda frame: float(selected_pose_rows[frame].get("official_render_mask_iou", 0.0)),
            reverse=True,
        )
    )
    with args.track_artifact.open(newline="") as handle:
        track_rows = list(csv.DictReader(handle))
    track_rows, mask_rejected_rows = _gate_tracks_by_object_mask(
        track_rows,
        profile.sample_dir / "results/segmentation/masks",
    )
    binding = bind_rigid_feature_tracks(
        track_rows=track_rows,
        states_by_frame=states_by_frame,
        cameras=cameras,
        geometry_provider=geometry.provider,
        feature_ids=feature_ids,
        reliable_anchor_frames=reliable_anchor_frames,
        maximum_anchor_error_px=args.maximum_anchor_error_px,
        minimum_track_visibility=args.minimum_track_visibility,
        surface_feature_id=str(
            descriptor.get("initializer", {}).get("body_feature_id", "object:body")
        ),
        maximum_track_count=int(
            profile.data.get("preprocess", {}).get("rigid_surface_maximum_track_count", 16)
        ),
        frame_stride=int(
            profile.data.get("preprocess", {}).get("rigid_surface_track_frame_stride", 3)
        ),
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
        "initializer_input_sha256": hashlib.sha256(
            (_sha256(descriptor_path) + _sha256(pose_hypotheses)).encode()
        ).hexdigest(),
        "anchor_pose_artifact": (
            str(repo_relative_value(pose_hypotheses)) if pose_hypotheses.is_file() else None
        ),
        "anchor_pose_artifact_sha256": (
            _sha256(pose_hypotheses) if pose_hypotheses.is_file() else None
        ),
        "anchor_pose_source": "external_megapose_selected_visual_geometry",
        "anchor_pose_symmetry_resolution": "descriptor_upright_axis_preserve_external_heading_fit_body_silhouette",
        "minimum_pose_mask_iou": args.minimum_pose_mask_iou,
        "output_csv": str(repo_relative_value(args.output_csv)),
        "output_csv_sha256": _sha256(args.output_csv),
        "association_count": len(binding.associations),
        "measurement_count": len(binding.measurement_rows),
        "feature_coverage": sorted(
            association.geometry_feature_id for association in binding.associations
        ),
        "semantic_role_rows": dict(sorted(role_counts.items())),
        "measurement_reliability_gate": {
            "kind": "object_mask_membership",
            "mask_source": str(
                repo_relative_value(profile.sample_dir / "results/segmentation/masks")
            ),
            "dilation_px": 5,
            "rejected_rows": mask_rejected_rows,
        },
        "reliable_anchor_frames": list(binding.reliable_anchor_frames),
        "associations": [
            {
                "track_id": association.track_id,
                "geometry_feature_id": association.geometry_feature_id,
                "anchor_frame": association.anchor_frame,
                "anchor_error_px": association.anchor_error_px,
                "confidence": association.confidence,
                "source_anchor_frames": list(association.source_anchor_frames),
                "local_xyz_m": list(association.local_xyz_m or ()),
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
