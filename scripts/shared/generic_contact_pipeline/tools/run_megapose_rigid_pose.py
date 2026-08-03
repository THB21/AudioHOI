#!/usr/bin/env python3
"""Estimate mesh-aware rigid poses at selected RGB keyframes with MegaPose.

This tool is deliberately asset/case neutral.  It consumes only an RGB frame,
camera intrinsics, a provider mesh, and an object mask-derived detection box.
It writes hypotheses to an isolated artifact and never publishes object_pose.csv.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import traceback
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
import trimesh
import yaml
from PIL import Image
from scipy.spatial.transform import Rotation


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_frames(value: str) -> list[int]:
    frames = sorted({int(token) for token in value.split(",") if token.strip()})
    if not frames or any(frame < 1 for frame in frames):
        raise ValueError("--frames must contain positive, one-based frame numbers")
    return frames


def mask_bbox(mask_path: Path, *, padding_ratio: float) -> tuple[list[float], int]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(mask_path)
    ys, xs = np.nonzero(mask > 0)
    if len(xs) < 64:
        raise ValueError(f"mask has too few foreground pixels: {mask_path}")
    height, width = mask.shape
    x0, x1 = float(xs.min()), float(xs.max() + 1)
    y0, y1 = float(ys.min()), float(ys.max() + 1)
    padding = padding_ratio * max(x1 - x0, y1 - y0)
    bbox = [
        max(0.0, x0 - padding),
        max(0.0, y0 - padding),
        min(float(width), x1 + padding),
        min(float(height), y1 + padding),
    ]
    return bbox, int(len(xs))


def pose_rows(
    predictions: Any,
    *,
    frame: int,
    frame_path: Path,
    mask_path: Path,
    bbox: list[float],
    mask_pixels: int,
) -> list[dict[str, Any]]:
    infos = predictions.infos.reset_index(drop=True)
    poses = predictions.poses.detach().cpu().numpy()
    score_field = "pose_logit" if "pose_logit" in infos.columns else "coarse_logit"
    rows: list[dict[str, Any]] = []
    for index, matrix in enumerate(poses):
        quaternion = Rotation.from_matrix(matrix[:3, :3]).as_quat()
        info = infos.iloc[index]
        rows.append(
            {
                "schema_version": 1,
                "frame": frame,
                "hypothesis_rank": 0,
                "score": float(info.get(score_field, float("nan"))),
                "score_field": score_field,
                "tx_m": float(matrix[0, 3]),
                "ty_m": float(matrix[1, 3]),
                "tz_m": float(matrix[2, 3]),
                "qx": float(quaternion[0]),
                "qy": float(quaternion[1]),
                "qz": float(quaternion[2]),
                "qw": float(quaternion[3]),
                "T_camera_object": matrix.tolist(),
                "detection_bbox_xyxy": bbox,
                "mask_pixels": mask_pixels,
                "frame_artifact": str(frame_path),
                "frame_sha256": sha256(frame_path),
                "mask_artifact": str(mask_path),
                "mask_sha256": sha256(mask_path),
                "source": "megapose_rgb_multi_hypothesis",
            }
        )
    rows.sort(key=lambda row: row["score"], reverse=True)
    for rank, row in enumerate(rows):
        row["hypothesis_rank"] = rank
    return rows


def write_pose_overlays(
    rows: list[dict[str, Any]], *, mesh_path: Path, mesh_units: str, K: np.ndarray, output_dir: Path
) -> None:
    mesh = trimesh.load_mesh(mesh_path, process=False)
    vertices = np.asarray(mesh.vertices, dtype=float)
    if mesh_units == "mm":
        vertices = vertices / 1000.0
    output_dir.mkdir(parents=True, exist_ok=True)
    for frame in sorted({int(row["frame"]) for row in rows}):
        frame_rows = sorted(
            (row for row in rows if int(row["frame"]) == frame),
            key=lambda row: int(row["hypothesis_rank"]),
        )
        panels: list[np.ndarray] = []
        for row in frame_rows:
            panel = cv2.imread(str(row["frame_artifact"]), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(row["mask_artifact"]), cv2.IMREAD_GRAYSCALE)
            contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(panel, contours, -1, (0, 255, 0), 2)
            transform = np.asarray(row["T_camera_object"], dtype=float)
            camera_points = vertices @ transform[:3, :3].T + transform[:3, 3]
            valid = camera_points[:, 2] > 1e-4
            projected = camera_points[valid, :2] / camera_points[valid, 2:3]
            projected = projected @ np.diag([K[0, 0], K[1, 1]]) + np.asarray([K[0, 2], K[1, 2]])
            if len(projected) >= 3:
                hull = cv2.convexHull(np.rint(projected).astype(np.int32))
                cv2.polylines(panel, [hull], True, (0, 170, 255), 3)
            x0, y0, x1, y1 = (int(round(value)) for value in row["detection_bbox_xyxy"])
            cv2.rectangle(panel, (x0, y0), (x1, y1), (255, 180, 0), 2)
            cv2.putText(
                panel,
                f"frame {frame} rank {row['hypothesis_rank']} score {row['score']:.3f}",
                (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )
            panels.append(cv2.resize(panel, (640, 360), interpolation=cv2.INTER_AREA))
        if panels:
            cv2.imwrite(str(output_dir / f"frame{frame:05d}_hypotheses.jpg"), np.hstack(panels))


def annotate_visual_geometry(
    rows: list[dict[str, Any]],
    *,
    mesh_path: Path,
    mesh_units: str,
    K: np.ndarray,
    track_artifact: Path | None,
    minimum_visible_tracks: int,
    minimum_hull_iou: float,
) -> None:
    mesh = trimesh.load_mesh(mesh_path, process=False)
    vertices = np.asarray(mesh.vertices, dtype=float)
    if mesh_units == "mm":
        vertices = vertices / 1000.0
    visible_counts: dict[int, int] = {}
    if track_artifact is not None and track_artifact.exists():
        tracks = pd.read_csv(track_artifact)
        visible = pd.to_numeric(tracks["visible"], errors="coerce").fillna(0.0) > 0.5
        visible_counts = tracks.loc[visible].groupby("frame").size().astype(int).to_dict()
    for row in rows:
        mask = cv2.imread(str(row["mask_artifact"]), cv2.IMREAD_GRAYSCALE) > 0
        transform = np.asarray(row["T_camera_object"], dtype=float)
        points = vertices @ transform[:3, :3].T + transform[:3, 3]
        valid = points[:, 2] > 1e-4
        projected = points[valid, :2] / points[valid, 2:3]
        projected = projected @ np.diag([K[0, 0], K[1, 1]]) + np.asarray([K[0, 2], K[1, 2]])
        hull_mask = np.zeros(mask.shape, dtype=np.uint8)
        if len(projected) >= 3:
            hull = cv2.convexHull(np.rint(projected).astype(np.int32))
            cv2.fillConvexPoly(hull_mask, hull, 1)
        intersection = int(np.logical_and(mask, hull_mask > 0).sum())
        union = int(np.logical_or(mask, hull_mask > 0).sum())
        row["projected_hull_mask_iou"] = float(intersection / union) if union else 0.0
        row["persistent_visible_track_count"] = visible_counts.get(int(row["frame"]))
    for frame in sorted({int(row["frame"]) for row in rows}):
        frame_rows = [row for row in rows if int(row["frame"]) == frame]
        best = max(frame_rows, key=lambda row: float(row["projected_hull_mask_iou"]))
        visible_count = visible_counts.get(frame)
        reliable = (
            (visible_count is None or visible_count >= minimum_visible_tracks)
            and float(best["projected_hull_mask_iou"]) >= minimum_hull_iou
        )
        for row in frame_rows:
            row["visual_geometry_rank"] = sorted(
                frame_rows,
                key=lambda candidate: float(candidate["projected_hull_mask_iou"]),
                reverse=True,
            ).index(row)
            row["selected_by_visual_geometry"] = bool(reliable and row is best)
            row["provider_status"] = "reliable_visible_keyframe" if reliable else "blocked_visual_evidence"


def run(args: argparse.Namespace) -> dict[str, Any]:
    case_config = args.case_config.resolve()
    sample_dir = args.sample_dir.resolve()
    asset_mesh = args.asset_mesh.resolve()
    mask_dir = args.mask_dir.resolve()
    output = args.output.resolve()
    # MegaPose reads this during module import.
    os.environ["MEGAPOSE_DATA_DIR"] = str(args.megapose_data_dir.resolve())
    from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset
    from megapose.inference.types import ObservationTensor
    from megapose.utils.load_model import NAMED_MODELS, load_named_model
    from megapose.utils.tensor_collection import PandasTensorCollection

    config = yaml.safe_load(case_config.read_text())
    camera = config["camera"]
    K = np.asarray(
        [
            [float(camera["fx"]), 0.0, float(camera["cx"])],
            [0.0, float(camera["fy"]), float(camera["cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    if args.model not in NAMED_MODELS or NAMED_MODELS[args.model]["requires_depth"]:
        raise ValueError(f"runner requires a registered RGB-only model: {args.model}")

    object_dataset = RigidObjectDataset(
        [RigidObject(label=args.entity_id, mesh_path=asset_mesh, mesh_units=args.mesh_units)]
    )
    estimator = load_named_model(
        args.model,
        object_dataset,
        n_workers=args.renderer_workers,
        bsz_images=args.batch_size,
    ).cuda().eval()
    model_info = NAMED_MODELS[args.model]

    output_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for frame in parse_frames(args.frames):
        frame_path = sample_dir / "frames" / f"{frame:05d}.png"
        mask_path = mask_dir / f"{frame:05d}_mask.png"
        try:
            bbox, mask_pixels = mask_bbox(mask_path, padding_ratio=args.bbox_padding_ratio)
            rgb = np.asarray(Image.open(frame_path).convert("RGB"), dtype=np.uint8)
            observation = ObservationTensor.from_numpy(rgb, depth=None, K=K).cuda()
            detections = PandasTensorCollection(
                infos=pd.DataFrame(
                    [{"label": args.entity_id, "batch_im_id": 0, "instance_id": 0, "score": 1.0}]
                ),
                bboxes=torch.as_tensor([bbox], dtype=torch.float32),
            ).cuda()
            with torch.no_grad():
                _, extra = estimator.run_inference_pipeline(
                    observation,
                    detections=detections,
                    n_refiner_iterations=int(model_info["inference_parameters"]["n_refiner_iterations"]),
                    n_pose_hypotheses=int(model_info["inference_parameters"]["n_pose_hypotheses"]),
                    bsz_images=args.batch_size,
                    bsz_objects=args.object_batch_size,
                )
            output_rows.extend(
                pose_rows(
                    extra["scoring"]["preds"],
                    frame=frame,
                    frame_path=frame_path,
                    mask_path=mask_path,
                    bbox=bbox,
                    mask_pixels=mask_pixels,
                )
            )
        except Exception as exc:  # preserve per-frame attempt provenance
            failures.append(
                {
                    "frame": frame,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            if args.fail_fast:
                raise

    annotate_visual_geometry(
        output_rows,
        mesh_path=asset_mesh,
        mesh_units=args.mesh_units,
        K=K,
        track_artifact=args.track_artifact.resolve() if args.track_artifact is not None else None,
        minimum_visible_tracks=args.minimum_visible_tracks,
        minimum_hull_iou=args.minimum_hull_iou,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as stream:
        for row in output_rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    if args.overlay_dir is not None and output_rows:
        write_pose_overlays(
            output_rows,
            mesh_path=asset_mesh,
            mesh_units=args.mesh_units,
            K=K,
            output_dir=args.overlay_dir.resolve(),
        )
    manifest = {
        "schema_version": 1,
        "provider": "megapose6d_official",
        "provider_repository": "https://github.com/megapose6d/megapose6d",
        "model": args.model,
        "entity_id": args.entity_id,
        "asset_mesh": str(asset_mesh),
        "asset_mesh_sha256": sha256(asset_mesh),
        "mesh_units": args.mesh_units,
        "case_config": str(case_config),
        "case_config_sha256": sha256(case_config),
        "camera": {key: float(camera[key]) for key in ("fx", "fy", "cx", "cy")},
        "requested_frames": parse_frames(args.frames),
        "successful_frames": sorted({int(row["frame"]) for row in output_rows}),
        "hypothesis_count": len(output_rows),
        "failures": failures,
        "output": str(output),
        "output_sha256": sha256(output),
        "publication_status": "isolated_candidate_not_accepted_pose",
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {**manifest, "manifest": str(manifest_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-config", type=Path, required=True)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--asset-mesh", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--frames", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--megapose-data-dir", type=Path, required=True)
    parser.add_argument("--overlay-dir", type=Path)
    parser.add_argument("--track-artifact", type=Path)
    parser.add_argument("--minimum-visible-tracks", type=int, default=16)
    parser.add_argument("--minimum-hull-iou", type=float, default=0.35)
    parser.add_argument("--entity-id", default="target_object")
    parser.add_argument("--mesh-units", default="mm", choices=("mm", "m"))
    parser.add_argument("--model", default="megapose-1.0-RGB-multi-hypothesis")
    parser.add_argument("--bbox-padding-ratio", type=float, default=0.04)
    parser.add_argument("--renderer-workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--object-batch-size", type=int, default=5)
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
