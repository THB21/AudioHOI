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


def select_visible_diverse_keyframes(
    *,
    sample_dir: Path,
    mask_dir: Path,
    track_artifact: Path | None,
    count: int,
    minimum_visible_tracks: int,
) -> tuple[list[int], dict[str, Any]]:
    """Select generic rigid-pose anchors from visual quality and temporal coverage."""

    if count < 2:
        raise ValueError("automatic keyframe count must be at least two")
    track_quality: dict[int, tuple[int, float]] = {}
    if track_artifact is not None and track_artifact.is_file():
        tracks = pd.read_csv(track_artifact)
        visible = pd.to_numeric(tracks["visible"], errors="coerce").fillna(0.0) > 0.5
        confidence = pd.to_numeric(tracks["confidence"], errors="coerce").fillna(0.0)
        for frame, group in tracks.assign(_visible=visible, _confidence=confidence).groupby("frame"):
            active = group.loc[group["_visible"]]
            track_quality[int(frame)] = (
                int(len(active)),
                float(active["_confidence"].mean()) if len(active) else 0.0,
            )

    candidates: list[dict[str, float | int]] = []
    for mask_path in sorted(mask_dir.glob("*_mask.png")):
        try:
            frame = int(mask_path.stem.split("_")[0])
        except ValueError:
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        frame_path = sample_dir / "frames" / f"{frame:05d}.png"
        image = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
        if mask is None or image is None:
            continue
        binary = (mask > 0).astype(np.uint8)
        area = int(binary.sum())
        if area < 64:
            continue
        count_visible, mean_confidence = track_quality.get(frame, (0, 0.0))
        if track_quality and count_visible < minimum_visible_tracks:
            continue
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        largest = int(stats[1:, cv2.CC_STAT_AREA].max()) if component_count > 1 else area
        dominance = float(largest / area)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        hull_area = max(float(cv2.contourArea(cv2.convexHull(contour))), 1.0)
        solidity = min(1.0, float(cv2.contourArea(contour)) / hull_area)
        ys, xs = np.nonzero(binary)
        crop = image[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        sharpness = float(cv2.Laplacian(crop, cv2.CV_64F).var()) if crop.size else 0.0
        candidates.append(
            {
                "frame": frame,
                "visible_tracks": count_visible,
                "track_confidence": mean_confidence,
                "component_dominance": dominance,
                "mask_solidity": solidity,
                "sharpness": sharpness,
            }
        )
    if len(candidates) < count:
        raise ValueError(
            f"only {len(candidates)} visually reliable frames are available for {count} anchors"
        )

    sharpness_values = np.asarray([float(item["sharpness"]) for item in candidates])
    sharp_low, sharp_high = np.percentile(sharpness_values, [10.0, 90.0])
    sharp_span = max(float(sharp_high - sharp_low), 1e-6)
    max_visible = max(int(item["visible_tracks"]) for item in candidates) or 1
    for item in candidates:
        sharp_score = np.clip((float(item["sharpness"]) - sharp_low) / sharp_span, 0.0, 1.0)
        visible_score = float(item["visible_tracks"]) / max_visible if track_quality else 1.0
        item["quality"] = float(
            0.35 * visible_score
            + 0.20 * float(item["track_confidence"])
            + 0.20 * float(item["component_dominance"])
            + 0.15 * float(item["mask_solidity"])
            + 0.10 * sharp_score
        )

    first, last = min(int(item["frame"]) for item in candidates), max(int(item["frame"]) for item in candidates)
    edges = np.linspace(first, last + 1, count + 1)
    selected: list[dict[str, float | int]] = []
    for index in range(count):
        bin_items = [
            item for item in candidates
            if edges[index] <= int(item["frame"]) < edges[index + 1]
        ]
        if not bin_items:
            continue
        selected.append(max(bin_items, key=lambda item: (float(item["quality"]), -int(item["frame"]))))
    if len(selected) != count:
        remaining = sorted(
            (item for item in candidates if item not in selected),
            key=lambda item: float(item["quality"]),
            reverse=True,
        )
        selected.extend(remaining[: count - len(selected)])
    frames = sorted(int(item["frame"]) for item in selected)
    return frames, {
        "mode": "visible_quality_temporal_stratification_v1",
        "requested_count": count,
        "candidate_count": len(candidates),
        "minimum_visible_tracks": minimum_visible_tracks,
        "selected": [item for item in sorted(selected, key=lambda item: int(item["frame"]))],
        "uses_vlm": False,
        "uses_manual_frame_ids": False,
    }


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


def load_initial_pose(path: Path, frame: int) -> np.ndarray:
    """Load one object-to-camera pose from a generic trajectory CSV."""
    poses = pd.read_csv(path)
    required = {"frame", "tx", "ty", "tz", "qw", "qx", "qy", "qz"}
    missing = required - set(poses.columns)
    if missing:
        raise ValueError(f"initial pose CSV is missing columns: {sorted(missing)}")
    selected = poses.loc[pd.to_numeric(poses["frame"], errors="coerce") == frame]
    if len(selected) != 1:
        raise ValueError(f"initial pose CSV must contain exactly one row for frame {frame}")
    row = selected.iloc[0]
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = Rotation.from_quat(
        [float(row["qx"]), float(row["qy"]), float(row["qz"]), float(row["qw"])]
    ).as_matrix()
    transform[:3, 3] = [float(row["tx"]), float(row["ty"]), float(row["tz"])]
    return transform


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


def annotate_official_render_geometry(
    rows: list[dict[str, Any]],
    *,
    renderer: Any,
    entity_id: str,
    K: np.ndarray,
    track_artifact: Path | None,
    minimum_visible_tracks: int,
    minimum_render_iou: float,
    overlay_dir: Path | None,
) -> None:
    from megapose.lib3d.transform import Transform
    from megapose.panda3d_renderer import Panda3dLightData
    from megapose.panda3d_renderer.types import Panda3dCameraData, Panda3dObjectData

    visible_counts: dict[int, int] = {}
    if track_artifact is not None and track_artifact.exists():
        tracks = pd.read_csv(track_artifact)
        visible = pd.to_numeric(tracks["visible"], errors="coerce").fillna(0.0) > 0.5
        visible_counts = tracks.loc[visible].groupby("frame").size().astype(int).to_dict()
    if overlay_dir is not None:
        overlay_dir.mkdir(parents=True, exist_ok=True)
    for frame in sorted({int(row["frame"]) for row in rows}):
        frame_rows = [row for row in rows if int(row["frame"]) == frame]
        rgb = np.asarray(Image.open(frame_rows[0]["frame_artifact"]).convert("RGB"), dtype=np.uint8)
        observed_mask = cv2.imread(str(frame_rows[0]["mask_artifact"]), cv2.IMREAD_GRAYSCALE) > 0
        camera = Panda3dCameraData(
            TWC=Transform(np.eye(4)),
            K=K,
            resolution=rgb.shape[:2],
        )
        light = Panda3dLightData(light_type="ambient", color=(1.0, 1.0, 1.0, 1.0))
        panels: list[np.ndarray] = []
        for row in frame_rows:
            rendering = renderer.render_scene(
                object_datas=[
                    Panda3dObjectData(
                        label=entity_id,
                        TWO=Transform(np.asarray(row["T_camera_object"], dtype=float)),
                    )
                ],
                camera_datas=[camera],
                light_datas=[light],
                render_depth=True,
                render_binary_mask=False,
                copy_arrays=True,
            )[0]
            render_mask = np.asarray(rendering.depth[..., 0] > 0, dtype=bool)
            intersection = int(np.logical_and(observed_mask, render_mask).sum())
            union = int(np.logical_or(observed_mask, render_mask).sum())
            row["official_render_mask_iou"] = float(intersection / union) if union else 0.0
            row["persistent_visible_track_count"] = visible_counts.get(frame)
            if overlay_dir is not None:
                contour = cv2.Canny(render_mask.astype(np.uint8) * 255, 30, 100)
                contour = cv2.dilate(contour, np.ones((3, 3), np.uint8), iterations=1)
                panel = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                observed_contours, _ = cv2.findContours(
                    observed_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(panel, observed_contours, -1, (0, 255, 0), 2)
                panel[contour > 0] = (0, 170, 255)
                cv2.putText(
                    panel,
                    f"frame {frame} net {row['hypothesis_rank']} render IoU {row['official_render_mask_iou']:.3f}",
                    (20, 38),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    3,
                    cv2.LINE_AA,
                )
                panels.append(cv2.resize(panel, (640, 360), interpolation=cv2.INTER_AREA))
                render_path = overlay_dir / f"frame{frame:05d}_rank{row['hypothesis_rank']}_render.png"
                cv2.imwrite(str(render_path), cv2.cvtColor(rendering.rgb, cv2.COLOR_RGB2BGR))
                row["official_render_artifact"] = str(render_path)
        ordered = sorted(
            frame_rows,
            key=lambda candidate: float(candidate["official_render_mask_iou"]),
            reverse=True,
        )
        visible_count = visible_counts.get(frame)
        reliable = (
            (visible_count is None or visible_count >= minimum_visible_tracks)
            and float(ordered[0]["official_render_mask_iou"]) >= minimum_render_iou
        )
        for rank, row in enumerate(ordered):
            row["visual_geometry_rank"] = rank
            row["selected_by_visual_geometry"] = bool(reliable and rank == 0)
            row["provider_status"] = "reliable_visible_keyframe" if reliable else "blocked_visual_evidence"
        if overlay_dir is not None and panels:
            cv2.imwrite(
                str(overlay_dir / f"frame{frame:05d}_official_hypotheses.jpg"),
                np.hstack(panels),
            )


def run(args: argparse.Namespace) -> dict[str, Any]:
    case_config = args.case_config.resolve()
    sample_dir = args.sample_dir.resolve()
    asset_mesh = args.asset_mesh.resolve()
    mask_dir = args.mask_dir.resolve()
    output = args.output.resolve()
    track_artifact = args.track_artifact.resolve() if args.track_artifact is not None else None
    if args.frames.strip().lower() == "auto":
        requested_frames, keyframe_selection = select_visible_diverse_keyframes(
            sample_dir=sample_dir,
            mask_dir=mask_dir,
            track_artifact=track_artifact,
            count=args.auto_frame_count,
            minimum_visible_tracks=args.minimum_visible_tracks,
        )
    else:
        requested_frames = parse_frames(args.frames)
        keyframe_selection = {
            "mode": "explicit",
            "selected_frames": requested_frames,
            "uses_vlm": False,
            "uses_manual_frame_ids": True,
        }
    if args.display and "DISPLAY" not in os.environ:
        os.environ["DISPLAY"] = args.display
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
    for frame in requested_frames:
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
            coarse_estimates = None
            if args.initial_pose_csv is not None:
                initial_transform = load_initial_pose(args.initial_pose_csv.resolve(), frame)
                coarse_estimates = PandasTensorCollection(
                    infos=pd.DataFrame(
                        [
                            {
                                "label": args.entity_id,
                                "batch_im_id": 0,
                                "instance_id": 0,
                                "hypothesis_id": 0,
                            }
                        ]
                    ),
                    poses=torch.as_tensor(initial_transform[None], dtype=torch.float32),
                ).cuda()
            with torch.no_grad():
                _, extra = estimator.run_inference_pipeline(
                    observation,
                    detections=detections,
                    coarse_estimates=coarse_estimates,
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

    annotate_official_render_geometry(
        output_rows,
        renderer=estimator.refiner_model.renderer._sync_renderer,
        entity_id=args.entity_id,
        K=K,
        track_artifact=track_artifact,
        minimum_visible_tracks=args.minimum_visible_tracks,
        minimum_render_iou=args.minimum_render_iou,
        overlay_dir=args.overlay_dir.resolve() if args.overlay_dir is not None else None,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as stream:
        for row in output_rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
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
        "requested_frames": requested_frames,
        "keyframe_selection": keyframe_selection,
        "successful_frames": sorted({int(row["frame"]) for row in output_rows}),
        "hypothesis_count": len(output_rows),
        "failures": failures,
        "output": str(output),
        "output_sha256": sha256(output),
        "publication_status": "isolated_candidate_not_accepted_pose",
        "initial_pose_csv": (
            str(args.initial_pose_csv.resolve()) if args.initial_pose_csv is not None else None
        ),
        "initial_pose_csv_sha256": (
            sha256(args.initial_pose_csv.resolve()) if args.initial_pose_csv is not None else None
        ),
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
    parser.add_argument("--auto-frame-count", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial-pose-csv", type=Path)
    parser.add_argument("--megapose-data-dir", type=Path, required=True)
    parser.add_argument("--display", default="")
    parser.add_argument("--overlay-dir", type=Path)
    parser.add_argument("--track-artifact", type=Path)
    parser.add_argument("--minimum-visible-tracks", type=int, default=16)
    parser.add_argument("--minimum-hull-iou", type=float, default=0.35)
    parser.add_argument("--minimum-render-iou", type=float, default=0.30)
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
