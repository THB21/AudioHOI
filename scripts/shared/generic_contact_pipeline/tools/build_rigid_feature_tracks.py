#!/usr/bin/env python3
"""Track named GeometryProvider features bidirectionally over a full sequence.

This tool is asset/profile driven.  It does not select behavior by case name and
does not write any accepted pose artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation


def _csv_list(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _pose_matrix(row: pd.Series) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_quat(
        [float(row.qx), float(row.qy), float(row.qz), float(row.qw)]
    ).as_matrix()
    transform[:3, 3] = [float(row.tx), float(row.ty), float(row.tz)]
    return transform


def _load_state(path: Path) -> pd.DataFrame:
    rows = pd.read_csv(path).rename(
        columns={
            "state_000": "tx",
            "state_001": "ty",
            "state_002": "tz",
            "state_003": "qw",
            "state_004": "qx",
            "state_005": "qy",
            "state_006": "qz",
        }
    )
    required = {"frame", "tx", "ty", "tz", "qw", "qx", "qy", "qz"}
    if not required.issubset(rows.columns):
        raise ValueError(f"pose source lacks fields: {sorted(required - set(rows.columns))}")
    return rows.set_index("frame")


def _feature_rows(
    descriptor: dict[str, object],
    *,
    body_feature: str,
    support_features: Iterable[str],
    line_features: Iterable[str],
    grasp_feature: str,
) -> list[dict[str, object]]:
    raw = descriptor.get("feature_points")
    if not isinstance(raw, dict):
        raise ValueError("geometry descriptor lacks feature_points")
    groups = (
        ((body_feature,), "body_corner"),
        (tuple(support_features), "support_point"),
        (tuple(line_features), "line_endpoint"),
        ((grasp_feature,), "grasp_point"),
    )
    rows: list[dict[str, object]] = []
    for feature_ids, kind in groups:
        for feature_id in feature_ids:
            points = raw.get(feature_id)
            if not isinstance(points, list) or not points:
                raise ValueError(f"geometry descriptor lacks declared feature: {feature_id}")
            for index, point in enumerate(points):
                xyz = np.asarray(point, dtype=float)
                if xyz.shape != (3,) or not np.isfinite(xyz).all():
                    raise ValueError(f"invalid point {index} for {feature_id}")
                rows.append(
                    {
                        "feature_id": str(feature_id),
                        "feature_kind": kind,
                        "point_index": index,
                        "local_x": float(xyz[0]),
                        "local_y": float(xyz[1]),
                        "local_z": float(xyz[2]),
                    }
                )
    expected = {
        "body_corner": 8,
        "support_point": 4,
        "line_endpoint": 4,
        "grasp_point": 1,
    }
    actual = pd.DataFrame(rows).groupby("feature_kind").size().to_dict()
    if actual != expected:
        raise ValueError(f"rigid feature contract mismatch: expected {expected}, got {actual}")
    return rows


def _project(points_local: np.ndarray, transform: np.ndarray, camera: np.ndarray) -> np.ndarray:
    points_camera = points_local @ transform[:3, :3].T + transform[:3, 3]
    if np.any(points_camera[:, 2] <= 1e-6):
        raise ValueError("feature projection crossed the camera plane")
    pixels = points_camera @ camera.T
    return pixels[:, :2] / pixels[:, 2:3]


def _mask_diagnostics(mask: np.ndarray, x: float, y: float) -> tuple[int, float]:
    binary = (mask > 0).astype(np.uint8)
    dilated = cv2.dilate(binary, np.ones((9, 9), np.uint8))
    xi, yi = int(round(x)), int(round(y))
    compatible = int(0 <= yi < mask.shape[0] and 0 <= xi < mask.shape[1] and dilated[yi, xi] > 0)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return compatible, float("inf")
    contour = max(contours, key=cv2.contourArea)
    distance = abs(float(cv2.pointPolygonTest(contour, (float(x), float(y)), True)))
    return compatible, distance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--geometry-descriptor", type=Path, required=True)
    parser.add_argument("--camera-json", type=Path)
    parser.add_argument("--anchor-frames", required=True)
    parser.add_argument("--early-pose", type=Path, required=True)
    parser.add_argument("--late-pose", type=Path, required=True)
    parser.add_argument("--late-pose-start", type=int, required=True)
    parser.add_argument("--body-feature", required=True)
    parser.add_argument("--support-features", required=True)
    parser.add_argument("--line-features", required=True)
    parser.add_argument("--grasp-feature", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resize-width", type=int, default=512)
    args = parser.parse_args()

    sample_dir = args.sample_dir.resolve()
    descriptor_path = args.geometry_descriptor.resolve()
    output = args.output.resolve()
    descriptor = json.loads(descriptor_path.read_text())
    features = _feature_rows(
        descriptor,
        body_feature=args.body_feature,
        support_features=_csv_list(args.support_features),
        line_features=_csv_list(args.line_features),
        grasp_feature=args.grasp_feature,
    )
    feature_table = pd.DataFrame(features)
    points_local = feature_table[["local_x", "local_y", "local_z"]].to_numpy(float)
    early = _load_state(args.early_pose.resolve())
    late = _load_state(args.late_pose.resolve())
    anchors = sorted({int(value) for value in _csv_list(args.anchor_frames)})

    if args.camera_json:
        camera_values = json.loads(args.camera_json.resolve().read_text())
        camera = np.asarray(camera_values["K"], dtype=float)
    else:
        camera = np.asarray(
            [[1468.604736328125, 0.0, 640.0], [0.0, 1468.604736328125, 360.0], [0.0, 0.0, 1.0]],
            dtype=float,
        )

    frame_paths = sorted((sample_dir / "frames").glob("*.png"))
    if not frame_paths:
        raise FileNotFoundError(f"no frames found below {sample_dir / 'frames'}")
    first = cv2.imread(str(frame_paths[0]))
    scale = min(1.0, float(args.resize_width) / first.shape[1])
    size = (int(round(first.shape[1] * scale)), int(round(first.shape[0] * scale)))
    frames = [
        cv2.cvtColor(cv2.resize(cv2.imread(str(path)), size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        for path in frame_paths
    ]
    tracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online", trust_repo=True).cuda().eval()

    def track_sequence(sequence: list[np.ndarray], query_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(sequence) == 1:
            return query_points[None].copy(), np.ones((1, len(query_points)), dtype=float)
        queries = torch.zeros((1, len(query_points), 3), dtype=torch.float32, device="cuda")
        queries[0, :, 1:] = torch.from_numpy((query_points * scale).astype(np.float32)).cuda()
        window: list[np.ndarray] = []
        first_step = True
        predicted_tracks = predicted_visibility = None

        def process(chunk_frames: list[np.ndarray], initial: bool):
            video = torch.from_numpy(np.stack(chunk_frames[-tracker.step * 2 :]))
            video = video.permute(0, 3, 1, 2)[None].float().cuda()
            return tracker(video, is_first_step=initial, queries=queries if initial else None, grid_size=0)

        with torch.inference_mode():
            for frame_index, frame in enumerate(sequence):
                if frame_index % tracker.step == 0 and frame_index != 0:
                    predicted_tracks, predicted_visibility = process(window, first_step)
                    first_step = False
                window.append(frame)
            final_start = -(frame_index % tracker.step) - tracker.step - 1
            predicted_tracks, predicted_visibility = process(window[final_start:], first_step)
        xy = predicted_tracks[0].cpu().numpy()
        visibility = predicted_visibility[0].cpu().numpy()
        xy[..., 0] /= scale
        xy[..., 1] /= scale
        return xy, visibility

    rows: list[dict[str, object]] = []
    for anchor in anchors:
        source = late if anchor >= args.late_pose_start else early
        if anchor not in source.index:
            raise ValueError(f"anchor frame {anchor} missing from its pose source")
        query_points = _project(points_local, _pose_matrix(source.loc[anchor]), camera)
        for direction, sequence, frame_numbers in (
            ("forward", frames[anchor - 1 :], list(range(anchor, len(frames) + 1))),
            ("backward", list(reversed(frames[:anchor])), list(range(anchor, 0, -1))),
        ):
            tracked, visibility = track_sequence(sequence, query_points)
            for sequence_index, frame in enumerate(frame_numbers):
                mask = cv2.imread(
                    str(sample_dir / "results/segmentation/masks" / f"{frame:05d}_mask.png"),
                    cv2.IMREAD_GRAYSCALE,
                )
                if mask is None:
                    raise FileNotFoundError(f"missing SAM2 mask for frame {frame}")
                for feature_index, feature in feature_table.iterrows():
                    x, y = tracked[sequence_index, feature_index]
                    compatible, edge_distance = _mask_diagnostics(mask, float(x), float(y))
                    rows.append(
                        {
                            **feature.to_dict(),
                            "query_id": f"{feature.feature_id}:{int(feature.point_index)}",
                            "anchor_frame": anchor,
                            "anchor_pose_source": "annotation_2" if source is late else "annotation_1",
                            "direction": direction,
                            "frame": frame,
                            "x": float(x),
                            "y": float(y),
                            "cotracker_visibility": float(visibility[sequence_index, feature_index]),
                            "mask_compatible": compatible,
                            "mask_edge_distance_px": edge_distance,
                        }
                    )

    table = pd.DataFrame(rows)
    medians = (
        table[table.cotracker_visibility >= 0.65]
        .groupby(["frame", "query_id"])[["x", "y"]]
        .median()
        .rename(columns={"x": "bank_median_x", "y": "bank_median_y"})
    )
    table = table.join(medians, on=["frame", "query_id"])
    table["cross_bank_error_px"] = np.hypot(
        table.x - table.bank_median_x, table.y - table.bank_median_y
    )
    visibility_score = np.clip((table.cotracker_visibility - 0.45) / 0.45, 0.0, 1.0)
    bank_score = np.exp(-np.square(table.cross_bank_error_px.fillna(1e6) / 12.0))
    mask_score = np.where(table.mask_compatible == 1, 1.0, 0.20)
    table["reliability"] = visibility_score * bank_score * mask_score
    table["usable"] = (
        (table.cotracker_visibility >= 0.65)
        & (table.cross_bank_error_px <= 18.0)
        & (table.mask_compatible == 1)
    ).astype(int)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    metrics = {
        "schema_version": 1,
        "frame_count": len(frame_paths),
        "anchor_frames": anchors,
        "feature_counts": feature_table.groupby("feature_kind").size().to_dict(),
        "row_count": len(table),
        "usable_row_count": int(table.usable.sum()),
        "usable_frames": int(table.loc[table.usable == 1, "frame"].nunique()),
        "resize_width": args.resize_width,
        "descriptor_sha256": hashlib.sha256(descriptor_path.read_bytes()).hexdigest(),
        "case_dispatch_used": False,
        "accepted_pose_read": False,
        "publication_status": "isolated_named_feature_evidence",
    }
    output.with_name("rigid_feature_track_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
