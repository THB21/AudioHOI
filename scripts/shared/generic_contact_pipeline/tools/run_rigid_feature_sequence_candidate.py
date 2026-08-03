#!/usr/bin/env python3
"""Solve an isolated root-SE(3) sequence from named rigid feature evidence.

Trusted intervals are immutable inputs.  Only explicitly free frames are
parameterized, and the tool never publishes an accepted pose.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.shared.generic_contact_pipeline.core.human_sites import extract_gvhmr_site_measurements


def _pose(row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    rotation = Rotation.from_quat([row.qx, row.qy, row.qz, row.qw]).as_matrix()
    translation = np.asarray([row.tx, row.ty, row.tz], dtype=float)
    return rotation, translation


def _project(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray, camera: np.ndarray) -> np.ndarray:
    world = points @ rotation.T + translation
    z = np.maximum(world[:, 2], 1e-6)
    return np.column_stack(
        (camera[0, 0] * world[:, 0] / z + camera[0, 2], camera[1, 1] * world[:, 1] / z + camera[1, 2])
    )


def _plane_from_feet(feet: np.ndarray, surface_offset_m: float) -> tuple[np.ndarray, float]:
    center = np.median(feet, axis=0)
    _u, _s, vectors = np.linalg.svd(feet - center, full_matrices=False)
    normal = vectors[-1]
    if normal[1] > 0.0:
        normal = -normal
    normal /= np.linalg.norm(normal)
    return normal, -float(normal @ center) + float(surface_offset_m)


def _principal_axis(points: np.ndarray) -> tuple[np.ndarray, float]:
    centered = points - np.mean(points, axis=0, keepdims=True)
    covariance = centered.T @ centered / max(1, len(centered))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)
    axis = eigenvectors[:, int(order[-1])]
    axis /= max(np.linalg.norm(axis), 1e-8)
    minor, major = float(eigenvalues[int(order[0])]), float(eigenvalues[int(order[-1])])
    anisotropy = max(0.0, (major - minor) / max(major + minor, 1e-9))
    return axis, anisotropy


def _mask_body_bbox(path: Path) -> tuple[np.ndarray, int, np.ndarray, float]:
    """Separate a stable main-body span from thin connected appendages."""

    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    ys, xs = np.nonzero(mask > 0)
    if len(xs) < 64:
        raise ValueError(f"mask has too few pixels: {path}")
    row_stats: list[tuple[int, int, int]] = []
    for y in range(int(ys.min()), int(ys.max()) + 1):
        row_x = np.flatnonzero(mask[y] > 0)
        width = 0 if len(row_x) == 0 else int(row_x.max() - row_x.min() + 1)
        row_stats.append((y, width, int(len(row_x))))
    maximum_width = max(width for _y, width, _count in row_stats)
    stable_rows = [
        y
        for y, width, count in row_stats
        if width >= 0.55 * maximum_width and count >= 0.45 * width
    ]
    runs: list[list[int]] = []
    for y in stable_rows:
        if not runs or y != runs[-1][-1] + 1:
            runs.append([y])
        else:
            runs[-1].append(y)
    body_start = int(ys.min())
    if runs:
        body_start = max(runs, key=len)[0]
    body_y, body_x = np.nonzero((mask > 0) & (np.indices(mask.shape)[0] >= body_start))
    axis, anisotropy = _principal_axis(np.column_stack((body_x, body_y)).astype(float))
    return (
        np.asarray([body_x.min(), body_y.min(), body_x.max(), body_y.max()], dtype=float),
        int(len(xs)),
        axis,
        anisotropy,
    )


def _mask_body_boundary_distance(path: Path) -> np.ndarray:
    """Return a dense pixel distance field for the stable rigid-body boundary."""

    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    ys, xs = np.nonzero(mask > 0)
    if len(xs) < 64:
        raise ValueError(f"mask has too few pixels: {path}")
    row_widths = []
    for y in range(int(ys.min()), int(ys.max()) + 1):
        row_x = np.flatnonzero(mask[y] > 0)
        width = 0 if len(row_x) == 0 else int(row_x.max() - row_x.min() + 1)
        row_widths.append((y, width, int(len(row_x))))
    maximum_width = max(width for _y, width, _count in row_widths)
    stable_rows = [
        y
        for y, width, count in row_widths
        if width >= 0.55 * maximum_width and count >= 0.45 * width
    ]
    runs: list[list[int]] = []
    for y in stable_rows:
        if not runs or y != runs[-1][-1] + 1:
            runs.append([y])
        else:
            runs[-1].append(y)
    stable_body_run = None if not runs else max(runs, key=len)
    body_start = int(ys.min()) if stable_body_run is None else stable_body_run[0]
    body_end = int(ys.max()) if stable_body_run is None else stable_body_run[-1]
    body_mask = np.zeros_like(mask)
    body_mask[body_start : body_end + 1] = mask[body_start : body_end + 1]
    contours, _hierarchy = cv2.findContours(body_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError(f"mask has no body contour: {path}")
    contour = max(contours, key=cv2.contourArea)
    boundary = np.zeros_like(mask)
    cv2.drawContours(boundary, [contour], -1, 255, thickness=1)
    return cv2.distanceTransform(255 - boundary, cv2.DIST_L2, 5).astype(float)


def _sample_projected_hull(points: np.ndarray, samples_per_edge: int = 6) -> np.ndarray:
    hull = cv2.convexHull(np.asarray(points, dtype=np.float32)).reshape(-1, 2).astype(float)
    samples = []
    fractions = np.linspace(0.0, 1.0, samples_per_edge, endpoint=False)
    for index, start in enumerate(hull):
        end = hull[(index + 1) % len(hull)]
        samples.extend(start[None, :] + fractions[:, None] * (end - start)[None, :])
    return np.asarray(samples, dtype=float)


def _bilinear_sample(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    height, width = image.shape
    x = np.clip(points[:, 0], 0.0, width - 1.001)
    y = np.clip(points[:, 1], 0.0, height - 1.001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
    wx, wy = x - x0, y - y0
    return (
        (1.0 - wx) * (1.0 - wy) * image[y0, x0]
        + wx * (1.0 - wy) * image[y0, x1]
        + (1.0 - wx) * wy * image[y1, x0]
        + wx * wy * image[y1, x1]
    )


def write_candidate_with_locked_reference_text(
    candidate: pd.DataFrame,
    reference_path: Path,
    output: Path,
) -> None:
    """Preserve locked pose fields textually, not merely within float tolerance."""

    pose_fields = ("tx", "ty", "tz", "qw", "qx", "qy", "qz")
    raw_reference = pd.read_csv(reference_path, dtype=str, keep_default_na=False).set_index("frame")
    formatted = candidate.copy()
    for field in ("time", *pose_fields):
        formatted[field] = formatted[field].map(lambda value: format(float(value), ".17g"))
    formatted["frame"] = formatted.frame.astype(int).astype(str)
    formatted["locked"] = formatted.locked.astype(int).astype(str)
    for index, row in formatted.iterrows():
        if row["locked"] != "1":
            continue
        raw = raw_reference.loc[row["frame"]]
        for field in pose_fields:
            formatted.at[index, field] = raw[field]
    formatted.to_csv(output, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--geometry-descriptor", type=Path, required=True)
    parser.add_argument("--reference-pose", type=Path, required=True)
    parser.add_argument("--feature-tracks", type=Path, required=True)
    parser.add_argument("--free-start", type=int, required=True)
    parser.add_argument("--free-end", type=int, required=True)
    parser.add_argument("--body-feature", required=True)
    parser.add_argument("--support-features", required=True)
    parser.add_argument("--line-features", required=True)
    parser.add_argument("--grasp-feature", required=True)
    parser.add_argument("--contact-facing-feature")
    parser.add_argument("--contact-facing-ramp-frames", type=int, default=12)
    parser.add_argument("--contact-facing-sigma-rad", type=float, default=0.12)
    parser.add_argument("--main-body-mask-weight", type=float, default=1.0)
    parser.add_argument("--mask-principal-axis-sigma-rad", type=float, default=0.07)
    parser.add_argument("--mask-silhouette-sigma-px", type=float, default=6.0)
    parser.add_argument("--mask-silhouette-weight", type=float, default=0.25)
    parser.add_argument("--rotation-acceleration-sigma-rad", type=float, default=0.035)
    parser.add_argument("--rotation-step-margin-deg", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-nfev", type=int, default=100)
    parser.add_argument("--max-rotation-step-deg", type=float, default=8.0)
    args = parser.parse_args()

    sample_dir = args.sample_dir.resolve()
    output = args.output.resolve()
    reference = pd.read_csv(args.reference_pose.resolve()).sort_values("frame").reset_index(drop=True)
    frames = reference.frame.astype(int).to_numpy()
    free_frames = np.arange(args.free_start, args.free_end + 1, dtype=int)
    if not np.array_equal(free_frames, frames[(frames >= args.free_start) & (frames <= args.free_end)]):
        raise ValueError("free interval must be contiguous and present in reference pose")
    if np.any(reference.loc[reference.frame.isin(free_frames), "locked"].astype(int) != 0):
        raise ValueError("free interval overlaps a locked reference frame")
    if np.any(reference.loc[~reference.frame.isin(free_frames), "locked"].astype(int) != 1):
        raise ValueError("every non-free frame must be reference locked")

    descriptor = json.loads(args.geometry_descriptor.resolve().read_text())
    raw_points = descriptor["feature_points"]
    body = np.asarray(raw_points[args.body_feature], dtype=float)
    support_ids = tuple(item.strip() for item in args.support_features.split(",") if item.strip())
    line_ids = tuple(item.strip() for item in args.line_features.split(",") if item.strip())
    support_groups = [np.asarray(raw_points[item], dtype=float) for item in support_ids]
    lines_local = [np.asarray(raw_points[item], dtype=float) for item in line_ids]
    grasp = np.asarray(raw_points[args.grasp_feature], dtype=float)
    if body.shape != (8, 3) or [len(item) for item in support_groups] != [2, 2] or [len(item) for item in lines_local] != [2, 2] or grasp.shape != (1, 3):
        raise ValueError("geometry descriptor does not satisfy the declared rigid feature contract")

    facing_axis = None
    if args.contact_facing_feature:
        declaration = descriptor.get("interaction_feature_frames", {}).get(args.contact_facing_feature)
        if declaration is None or declaration.get("relation") != "grasp_face_toward_human":
            raise ValueError(f"missing grasp_face_toward_human declaration for {args.contact_facing_feature}")
        facing_axis = np.asarray(declaration["facing_axis_local"], dtype=float)
        facing_axis /= max(np.linalg.norm(facing_axis), 1e-8)

    camera = np.asarray(
        [[1468.604736328125, 0.0, 640.0], [0.0, 1468.604736328125, 360.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    reference_by_frame = reference.set_index("frame")
    initial_rotations = {frame: _pose(reference_by_frame.loc[frame])[0] for frame in frames}
    initial_translations = {frame: _pose(reference_by_frame.loc[frame])[1] for frame in frames}

    tracks = pd.read_csv(args.feature_tracks.resolve())
    tracks = tracks[(tracks.usable == 1) & tracks.frame.isin(free_frames)].copy()
    aggregate_rows = []
    for (frame, query_id), rows in tracks.groupby(["frame", "query_id"]):
        weights = np.maximum(rows.reliability.to_numpy(float), 1e-6)
        aggregate_rows.append(
            {
                "frame": int(frame),
                "query_id": query_id,
                "feature_id": rows.feature_id.iloc[0],
                "feature_kind": rows.feature_kind.iloc[0],
                "point_index": int(rows.point_index.iloc[0]),
                "local_x": float(rows.local_x.iloc[0]),
                "local_y": float(rows.local_y.iloc[0]),
                "local_z": float(rows.local_z.iloc[0]),
                "x": float(np.average(rows.x, weights=weights)),
                "y": float(np.average(rows.y, weights=weights)),
                "confidence": float(min(1.0, np.sum(weights))),
                "bank_count": len(rows),
            }
        )
    point_observations = pd.DataFrame(aggregate_rows)

    contacts = pd.read_csv(sample_dir / "results/pure_solver_no_audio_no_vlm/contact_candidates.csv")
    contacts = contacts[(contacts.contact_active == 1) & contacts.frame.isin(free_frames)].set_index("frame")
    object_observations = pd.read_csv(
        sample_dir / "results/pure_solver_no_audio_no_vlm/object_observations.csv"
    )
    object_observations = object_observations[
        object_observations.frame.isin(free_frames)
    ].set_index("frame")
    line_observations = pd.read_csv(sample_dir / "results/pure_solver_no_audio_no_vlm/line_observations.csv")
    line_observations = line_observations[(line_observations.line_observation_trusted == 1) & line_observations.frame.isin(free_frames)]
    observed_lines_by_frame: dict[int, np.ndarray] = {}
    unassigned_lines_by_frame: dict[int, dict[str, object]] = {}
    for frame, rows in line_observations.groupby("frame"):
        modes = rows.get("line_observation_mode", pd.Series("paired", index=rows.index)).astype(str)
        paired = rows[modes != "unassigned_axis"]
        if len(paired) == 2:
            observed_lines_by_frame[int(frame)] = paired[["physical_x1", "physical_y1", "physical_x2", "physical_y2"]].to_numpy(float).reshape(2, 2, 2)
        unassigned = rows[modes == "unassigned_axis"]
        if len(unassigned) == 1:
            row = unassigned.iloc[0]
            candidate_ids = tuple(
                value for value in str(row.candidate_feature_ids).split("|") if value
            )
            candidate_indices = tuple(line_ids.index(value) for value in candidate_ids if value in line_ids)
            if not candidate_indices:
                raise ValueError(f"unassigned line frame {frame} has no declared geometry candidates")
            unassigned_lines_by_frame[int(frame)] = {
                "target": row[["physical_x1", "physical_y1", "physical_x2", "physical_y2"]].to_numpy(float).reshape(2, 2),
                "confidence": float(row.endpoint_track_conf),
                "candidate_indices": candidate_indices,
            }

    body_mask_bboxes = {}
    body_mask_axes = {}
    body_mask_anisotropy = {}
    body_mask_boundary_distances = {}
    for frame in free_frames:
        mask_path = sample_dir / "results/segmentation/masks" / f"{frame:05d}_mask.png"
        body_mask_bboxes[frame], _mask_area, body_mask_axes[frame], body_mask_anisotropy[frame] = _mask_body_bbox(mask_path)
        body_mask_boundary_distances[frame] = _mask_body_boundary_distance(mask_path)
    mask_bbox_sigma_px = float(descriptor.get("initializer", {}).get("mask_bbox_sigma_px", 8.0))

    sites = extract_gvhmr_site_measurements(
        sample_id="rigid_feature_sequence_candidate",
        result_pkl=sample_dir / "results/gvhmr/result.pkl",
        body_models_root=Path("third-party/GVHMR/inputs/checkpoints/body_models").resolve(),
        frame_times={int(frame): float(reference_by_frame.loc[frame, "time"]) for frame in frames},
    ).measurements
    hands = {
        (site.frame, site.site.side): np.asarray(site.xyz_m, dtype=float)
        for site in sites
        if site.site.body_part == "hand"
    }
    feet = np.asarray([site.xyz_m for site in sites if site.site.body_part == "foot"], dtype=float)
    human_reference_by_frame: dict[int, np.ndarray] = {}
    for site in sites:
        if site.site.body_part == "foot":
            human_reference_by_frame.setdefault(site.frame, []).append(np.asarray(site.xyz_m, dtype=float))
    human_reference_by_frame = {
        frame: np.mean(np.stack(points), axis=0)
        for frame, points in human_reference_by_frame.items()
        if points
    }
    plane_normal, plane_offset = _plane_from_feet(feet, 0.05)

    boundary_facing_angle = 0.0
    if facing_axis is not None:
        boundary_frame = args.free_start - 1
        predicted = initial_rotations[boundary_frame] @ facing_axis
        desired = human_reference_by_frame[boundary_frame] - initial_translations[boundary_frame]
        predicted -= plane_normal * float(predicted @ plane_normal)
        desired -= plane_normal * float(desired @ plane_normal)
        predicted /= max(np.linalg.norm(predicted), 1e-8)
        desired /= max(np.linalg.norm(desired), 1e-8)
        boundary_facing_angle = abs(float(np.arctan2(
            plane_normal @ np.cross(desired, predicted),
            desired @ predicted,
        )))

    initial_line_assignments: dict[int, tuple[int, int]] = {}
    for frame, observed in observed_lines_by_frame.items():
        projected = np.asarray([_project(line, initial_rotations[frame], initial_translations[frame], camera) for line in lines_local])

        def line_cost(points: np.ndarray, target: np.ndarray) -> float:
            direction = target[1] - target[0]
            normal = np.asarray([-direction[1], direction[0]]) / max(np.linalg.norm(direction), 1e-8)
            return float(np.abs((points - target[0]) @ normal).sum())

        initial_line_assignments[frame] = min(
            ((0, 1), (1, 0)),
            key=lambda assignment: sum(line_cost(projected[i], observed[j]) for i, j in enumerate(assignment)),
        )

    support_group_by_frame = {}
    for frame in free_frames:
        distances = [
            np.abs((group @ initial_rotations[frame].T + initial_translations[frame]) @ plane_normal + plane_offset).mean()
            for group in support_groups
        ]
        support_group_by_frame[frame] = int(np.argmin(distances))

    frame_to_slot = {frame: index for index, frame in enumerate(free_frames)}
    x0 = np.zeros((len(free_frames), 6), dtype=float).reshape(-1)

    def states(parameters: np.ndarray) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
        values = parameters.reshape(len(free_frames), 6)
        rotations = dict(initial_rotations)
        translations = dict(initial_translations)
        for frame, slot in frame_to_slot.items():
            rotations[frame] = Rotation.from_rotvec(values[slot, 3:]).as_matrix() @ initial_rotations[frame]
            translations[frame] = initial_translations[frame] + values[slot, :3]
        return rotations, translations

    def residual(parameters: np.ndarray, *, ledger: bool = False):
        rotations, translations = states(parameters)
        blocks: list[np.ndarray] = []
        ledger_rows: list[dict[str, object]] = []

        def add(name: str, frame: int, values: np.ndarray, metadata: dict[str, object] | None = None):
            values = np.asarray(values, dtype=float).reshape(-1)
            blocks.append(values)
            if ledger:
                ledger_rows.append({
                    "factor": name,
                    "frame": frame,
                    "rows": len(values),
                    "rms": float(np.sqrt(np.mean(values * values))) if len(values) else 0.0,
                    **(metadata or {}),
                })

        for row in point_observations.itertuples():
            local = np.asarray([[row.local_x, row.local_y, row.local_z]], dtype=float)
            predicted = _project(local, rotations[row.frame], translations[row.frame], camera)[0]
            add("named_feature_reprojection", row.frame, np.sqrt(row.confidence) * (predicted - [row.x, row.y]) / 8.0)

        for frame in free_frames:
            rotation, translation = rotations[frame], translations[frame]
            if frame in contacts.index:
                contact = contacts.loc[frame]
                if isinstance(contact, pd.DataFrame):
                    contact = contact.iloc[0]
                predicted = _project(grasp, rotation, translation, camera)[0]
                confidence = max(0.1, float(contact.contact_conf))
                add("grasp_point_reprojection", frame, 1.5 * np.sqrt(confidence) * (predicted - [contact.contact_u, contact.contact_v]) / 6.0)

                if facing_axis is not None and frame in human_reference_by_frame:
                    predicted_facing = rotation @ facing_axis
                    desired_facing = human_reference_by_frame[frame] - translation
                    predicted_facing -= plane_normal * float(predicted_facing @ plane_normal)
                    desired_facing -= plane_normal * float(desired_facing @ plane_normal)
                    predicted_facing /= max(np.linalg.norm(predicted_facing), 1e-8)
                    desired_facing /= max(np.linalg.norm(desired_facing), 1e-8)
                    angle = abs(float(np.arctan2(
                        plane_normal @ np.cross(desired_facing, predicted_facing),
                        desired_facing @ predicted_facing,
                    )))
                    ramp_progress = min(
                        1.0,
                        max(0.0, (frame - args.free_start + 1) / max(1, args.contact_facing_ramp_frames)),
                    )
                    allowed_angle = (1.0 - ramp_progress) * boundary_facing_angle
                    violation = max(angle - allowed_angle, 0.0)
                    add(
                        "descriptor_contact_facing",
                        frame,
                        np.asarray([2.0 * np.sqrt(confidence) * violation / args.contact_facing_sigma_rad]),
                    )

            observed = observed_lines_by_frame.get(frame)
            if observed is not None:
                assignment = initial_line_assignments[frame]
                projected_lines = [_project(line, rotation, translation, camera) for line in lines_local]
                for line_index, observed_index in enumerate(assignment):
                    target = observed[observed_index]
                    direction = target[1] - target[0]
                    direction /= max(np.linalg.norm(direction), 1e-8)
                    normal = np.asarray([-direction[1], direction[0]])
                    predicted = projected_lines[line_index]
                    add("rail_axis_line", frame, ((predicted - target[0]) @ normal) / 5.0)
                    predicted_direction = predicted[1] - predicted[0]
                    predicted_direction /= max(np.linalg.norm(predicted_direction), 1e-8)
                    cross = predicted_direction[0] * direction[1] - predicted_direction[1] * direction[0]
                    add("rail_direction", frame, np.asarray([cross / 0.10]))
            elif frame in unassigned_lines_by_frame:
                declaration = unassigned_lines_by_frame[frame]
                target = np.asarray(declaration["target"], dtype=float)
                direction = target[1] - target[0]
                direction /= max(np.linalg.norm(direction), 1e-8)
                normal = np.asarray([-direction[1], direction[0]])
                confidence = np.sqrt(max(0.0, float(declaration["confidence"])))
                candidates: list[tuple[float, int, np.ndarray, float]] = []
                for line_index in declaration["candidate_indices"]:
                    predicted = _project(lines_local[line_index], rotation, translation, camera)
                    axis_values = confidence * ((predicted - target[0]) @ normal) / 5.0
                    predicted_direction = predicted[1] - predicted[0]
                    predicted_direction /= max(np.linalg.norm(predicted_direction), 1e-8)
                    cross = float(predicted_direction[0] * direction[1] - predicted_direction[1] * direction[0])
                    direction_value = confidence * cross / 0.10
                    cost = float(axis_values @ axis_values + direction_value * direction_value)
                    candidates.append((cost, int(line_index), axis_values, direction_value))
                _cost, selected_line_index, axis_values, direction_value = min(candidates, key=lambda item: item[0])
                metadata = {"selected_candidate_index": selected_line_index, "observation_mode": "unassigned_axis"}
                add("unassigned_rail_axis_line", frame, axis_values, metadata)
                add("unassigned_rail_direction", frame, np.asarray([direction_value]), metadata)

            silhouette_body = body
            projected_body = _project(silhouette_body, rotation, translation, camera)
            predicted_bbox = np.asarray(
                [projected_body[:, 0].min(), projected_body[:, 1].min(), projected_body[:, 0].max(), projected_body[:, 1].max()]
            )
            target_bbox = body_mask_bboxes[frame]
            containment = np.asarray(
                [
                    max(predicted_bbox[0] - target_bbox[0], 0.0),
                    max(target_bbox[2] - predicted_bbox[2], 0.0),
                ]
            )
            add("visible_main_body_horizontal_containment", frame, containment / mask_bbox_sigma_px)
            observation_row = object_observations.loc[frame] if frame in object_observations.index else None
            if isinstance(observation_row, pd.DataFrame):
                observation_row = observation_row.iloc[0]
            if observation_row is not None and str(observation_row.visibility) == "visible":
                add(
                    "gated_main_body_horizontal_bounds",
                    frame,
                    np.sqrt(args.main_body_mask_weight)
                    * (predicted_bbox[[0, 2]] - target_bbox[[0, 2]])
                    / mask_bbox_sigma_px,
                )
                hull_samples = _sample_projected_hull(projected_body)
                predicted_center = 0.5 * (predicted_bbox[:2] + predicted_bbox[2:])
                target_center = 0.5 * (target_bbox[:2] + target_bbox[2:])
                hull_samples = hull_samples + (target_center - predicted_center)[None, :]
                boundary_distance = _bilinear_sample(body_mask_boundary_distances[frame], hull_samples)
                normalized_distance = boundary_distance / args.mask_silhouette_sigma_px
                add(
                    "visible_centered_rigid_silhouette_boundary",
                    frame,
                    np.sqrt(args.mask_silhouette_weight)
                    * np.sqrt(np.log1p(normalized_distance * normalized_distance)),
                )
            if observed is None and frame not in unassigned_lines_by_frame and observation_row is not None:
                visibility_weight = {
                    "visible": 1.0,
                    "partially_visible": 0.0,
                    "occluded": 0.0,
                }.get(str(observation_row.visibility), 0.0)
                if visibility_weight > 0.0:
                    predicted_axis, _predicted_anisotropy = _principal_axis(projected_body)
                    target_axis = body_mask_axes[frame]
                    signed_sine = predicted_axis[0] * target_axis[1] - predicted_axis[1] * target_axis[0]
                    add(
                        "mask_principal_axis",
                        frame,
                        np.asarray([
                            visibility_weight
                            * body_mask_anisotropy[frame]
                            * signed_sine
                            / args.mask_principal_axis_sigma_rad
                        ]),
                    )

            group_index = support_group_by_frame[frame]
            signed_groups = [
                (group @ rotation.T + translation) @ plane_normal + plane_offset
                for group in support_groups
            ]
            add("wheel_support", frame, signed_groups[group_index] / 0.025)
            penetration = np.minimum(np.concatenate(signed_groups), 0.0)
            add("wheel_penetration", frame, 3.0 * penetration / 0.015)

        for frame in range(args.free_start, args.free_end + 2):
            previous = frame - 1
            if frame not in rotations or previous not in rotations:
                continue
            rotation_step = Rotation.from_matrix(rotations[previous].T @ rotations[frame]).as_rotvec()
            rotation_step_angle = float(np.linalg.norm(rotation_step))
            soft_rotation_step_limit = np.radians(
                max(0.0, args.max_rotation_step_deg - args.rotation_step_margin_deg)
            )
            rotation_step_excess = max(rotation_step_angle - soft_rotation_step_limit, 0.0)
            add("rotation_step_limit", frame, np.asarray([10.0 * rotation_step_excess / 0.015]))
            current_handle = grasp[0] @ rotations[frame].T + translations[frame]
            previous_handle = grasp[0] @ rotations[previous].T + translations[previous]
            current_contact = contacts.loc[frame] if frame in contacts.index else None
            previous_contact = contacts.loc[previous] if previous in contacts.index else None
            if isinstance(current_contact, pd.DataFrame):
                current_contact = current_contact.iloc[0]
            if isinstance(previous_contact, pd.DataFrame):
                previous_contact = previous_contact.iloc[0]
            if current_contact is not None and previous_contact is not None:
                current_side = str(current_contact.human_side)
                previous_side = str(previous_contact.human_side)
                current_hand = hands.get((frame, current_side))
                previous_hand = hands.get((previous, previous_side))
                if current_side == previous_side and current_hand is not None and previous_hand is not None:
                    add(
                        "grasp_comotion",
                        frame,
                        ((current_handle - previous_handle) - (current_hand - previous_hand)) / 0.018,
                    )

        for frame in range(args.free_start, args.free_end + 1):
            previous, following = frame - 1, frame + 1
            if previous not in rotations or following not in rotations:
                continue
            translation_acceleration = translations[following] - 2.0 * translations[frame] + translations[previous]
            add("translation_acceleration", frame, translation_acceleration / 0.025)
            previous_step = Rotation.from_matrix(rotations[previous].T @ rotations[frame]).as_rotvec()
            next_step = Rotation.from_matrix(rotations[frame].T @ rotations[following]).as_rotvec()
            add(
                "rotation_acceleration",
                frame,
                (next_step - previous_step) / args.rotation_acceleration_sigma_rad,
            )

        values = parameters.reshape(len(free_frames), 6)
        for frame, slot in frame_to_slot.items():
            add("bounded_interpolation_prior", frame, np.concatenate((values[slot, :3] / 0.20, values[slot, 3:] / 0.45)))
        vector = np.concatenate(blocks)
        return (vector, ledger_rows) if ledger else vector

    lower = np.tile(np.asarray([-0.45, -0.35, -0.55, -1.1, -1.1, -1.1]), len(free_frames))
    upper = -lower
    initial_residual = residual(x0)
    solved = least_squares(
        residual,
        x0,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=args.max_nfev,
        ftol=1e-5,
        xtol=1e-6,
        gtol=1e-6,
        verbose=1,
    )
    final_residual, ledger_rows = residual(solved.x, ledger=True)
    rotations, translations = states(solved.x)

    candidate = reference.copy()
    candidate["candidate_source"] = np.where(candidate.locked.astype(int) == 1, candidate.reference_source, "whole_sequence_rigid_feature_solve")
    for frame in free_frames:
        qx, qy, qz, qw = Rotation.from_matrix(rotations[frame]).as_quat()
        selected = candidate.frame == frame
        candidate.loc[selected, ["tx", "ty", "tz"]] = translations[frame]
        candidate.loc[selected, ["qw", "qx", "qy", "qz"]] = [qw, qx, qy, qz]
    output.parent.mkdir(parents=True, exist_ok=True)
    write_candidate_with_locked_reference_text(candidate, args.reference_pose.resolve(), output)
    pd.DataFrame(ledger_rows).to_csv(output.with_name("factor_ledger.csv"), index=False)

    locked = candidate.locked.astype(int) == 1
    fields = ["tx", "ty", "tz", "qw", "qx", "qy", "qz"]
    locked_exact = bool(np.array_equal(candidate.loc[locked, fields].to_numpy(), reference.loc[locked, fields].to_numpy()))
    translations_array = candidate[["tx", "ty", "tz"]].to_numpy(float)
    rotations_array = Rotation.from_quat(candidate[["qx", "qy", "qz", "qw"]].to_numpy(float))
    translation_steps = np.linalg.norm(np.diff(translations_array, axis=0), axis=1)
    rotation_steps = np.degrees((rotations_array[:-1].inv() * rotations_array[1:]).magnitude())
    wheel_distances = []
    for frame in free_frames:
        for group in support_groups:
            wheel_distances.extend(((group @ rotations[frame].T + translations[frame]) @ plane_normal + plane_offset).tolist())
    maximum_translation_step = float(np.max(translation_steps[args.free_start - 2 : args.free_end]))
    maximum_rotation_step = float(np.max(rotation_steps[args.free_start - 2 : args.free_end]))
    maximum_penetration = float(max(0.0, -np.min(wheel_distances)))
    gates = {
        "optimizer_converged": bool(solved.success),
        "locked_reference_exact": locked_exact,
        "translation_step_at_most_0_12m": maximum_translation_step <= 0.12,
        "rotation_step_within_declared_limit": maximum_rotation_step <= args.max_rotation_step_deg,
        "wheel_penetration_at_most_0_01m": maximum_penetration <= 0.01,
    }
    quality_passed = all(gates.values())
    metrics = {
        "schema_version": 1,
        "success": bool(solved.success),
        "message": solved.message,
        "function_evaluations": int(solved.nfev),
        "initial_residual_rms": float(np.sqrt(np.mean(initial_residual * initial_residual))),
        "final_residual_rms": float(np.sqrt(np.mean(final_residual * final_residual))),
        "locked_reference_exact": locked_exact,
        "max_translation_step_m_free_and_boundaries": maximum_translation_step,
        "max_rotation_step_deg_free_and_boundaries": maximum_rotation_step,
        "minimum_wheel_plane_distance_m": float(np.min(wheel_distances)),
        "maximum_wheel_penetration_m": maximum_penetration,
        "named_feature_observation_count": len(point_observations),
        "contact_facing_feature": args.contact_facing_feature,
        "contact_facing_boundary_angle_deg": float(np.degrees(boundary_facing_angle)),
        "contact_facing_ramp_frames": int(args.contact_facing_ramp_frames),
        "main_body_mask_weight": float(args.main_body_mask_weight),
        "mask_principal_axis_sigma_rad": float(args.mask_principal_axis_sigma_rad),
        "mask_silhouette_sigma_px": float(args.mask_silhouette_sigma_px),
        "mask_silhouette_weight": float(args.mask_silhouette_weight),
        "rotation_acceleration_sigma_rad": float(args.rotation_acceleration_sigma_rad),
        "rotation_step_margin_deg": float(args.rotation_step_margin_deg),
        "paired_line_frame_count": len(observed_lines_by_frame),
        "unassigned_line_frame_count": len(unassigned_lines_by_frame),
        "case_dispatch_used": False,
        "human_state_optimized": False,
        "accepted_pose_read": False,
        "accepted_pose_written": False,
        "reference_pose_sha256": hashlib.sha256(args.reference_pose.resolve().read_bytes()).hexdigest(),
        "quality_gates": gates,
        "quality_passed": quality_passed,
        "publication_status": "isolated_candidate_not_accepted_pose" if quality_passed else "isolated_candidate_rejected_by_quality_gate",
    }
    if not locked_exact:
        raise RuntimeError("locked reference values changed")
    output.with_name("trajectory_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
