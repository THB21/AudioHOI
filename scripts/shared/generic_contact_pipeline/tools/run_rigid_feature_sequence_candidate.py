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


def _sample_projected_hull(points: np.ndarray, sample_count: int = 24) -> np.ndarray:
    hull = cv2.convexHull(np.asarray(points, dtype=np.float32)).reshape(-1, 2).astype(float)
    following = np.roll(hull, -1, axis=0)
    edges = following - hull
    lengths = np.linalg.norm(edges, axis=1)
    perimeter = float(lengths.sum())
    if perimeter <= 1e-8:
        return np.repeat(hull[:1], sample_count, axis=0)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    distances = np.linspace(0.0, perimeter, sample_count, endpoint=False)
    edge_indices = np.minimum(np.searchsorted(cumulative, distances, side="right") - 1, len(hull) - 1)
    local = (distances - cumulative[edge_indices]) / np.maximum(lengths[edge_indices], 1e-8)
    return hull[edge_indices] + local[:, None] * edges[edge_indices]


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


def _select_temporally_consistent_flow_banks(
    candidates: pd.DataFrame,
    switch_penalty: float,
) -> tuple[pd.DataFrame, int]:
    """Select one locally tracked bank per feature/frame with temporal hysteresis.

    CoTracker banks overlap.  Selecting the highest score independently at
    every frame makes the measurement target jump whenever two banks exchange
    rank.  This dynamic program keeps a bank until another is sufficiently
    better, while still switching when the current bank leaves its local
    tracking window.
    """

    selected: list[pd.Series] = []
    switch_count = 0
    for _query_id, query_rows in candidates.groupby("query_id"):
        available_frames = sorted(query_rows.frame.astype(int).unique())
        segments: list[list[int]] = []
        for frame in available_frames:
            if not segments or frame != segments[-1][-1] + 1:
                segments.append([frame])
            else:
                segments[-1].append(frame)
        for segment in segments:
            frame_rows: list[pd.DataFrame] = []
            backpointers: list[dict[tuple[int, str], tuple[int, str] | None]] = []
            costs: dict[tuple[int, str], float] = {}
            for frame_index, frame in enumerate(segment):
                rows = query_rows[query_rows.frame == frame].copy()
                rows["bank_key"] = list(
                    zip(rows.anchor_frame.astype(int), rows.direction.astype(str))
                )
                rows = (
                    rows.sort_values("flow_score", ascending=False)
                    .drop_duplicates("bank_key")
                )
                frame_rows.append(rows)
                new_costs: dict[tuple[int, str], float] = {}
                predecessors: dict[tuple[int, str], tuple[int, str] | None] = {}
                for row in rows.itertuples():
                    bank = (int(row.anchor_frame), str(row.direction))
                    observation_cost = -float(np.log(max(float(row.flow_score), 1e-9)))
                    if frame_index == 0:
                        new_costs[bank] = observation_cost
                        predecessors[bank] = None
                        continue
                    transition_cost, predecessor = min(
                        (
                            previous_cost + (0.0 if previous_bank == bank else switch_penalty),
                            previous_bank,
                        )
                        for previous_bank, previous_cost in costs.items()
                    )
                    new_costs[bank] = observation_cost + transition_cost
                    predecessors[bank] = predecessor
                costs = new_costs
                backpointers.append(predecessors)
            bank = min(costs, key=costs.get)
            reversed_rows: list[pd.Series] = []
            for frame_index in range(len(segment) - 1, -1, -1):
                rows = frame_rows[frame_index]
                row = rows[rows.bank_key == bank].iloc[0].copy()
                row["selected_bank_key"] = f"{bank[0]}:{bank[1]}"
                reversed_rows.append(row)
                predecessor = backpointers[frame_index][bank]
                if predecessor is not None:
                    switch_count += int(predecessor != bank)
                    bank = predecessor
            selected.extend(reversed(reversed_rows))
    result = pd.DataFrame(selected).sort_values(["frame", "query_id"]).reset_index(drop=True)
    return result, switch_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--geometry-descriptor", type=Path, required=True)
    parser.add_argument("--reference-pose", type=Path, required=True)
    parser.add_argument("--warm-start-pose", type=Path)
    parser.add_argument("--feature-tracks", type=Path, required=True)
    parser.add_argument("--feature-flow-tracks", type=Path)
    parser.add_argument("--rigid-physics-evidence-dir", type=Path, required=True)
    parser.add_argument("--free-start", type=int, required=True)
    parser.add_argument("--free-end", type=int, required=True)
    parser.add_argument("--body-feature", required=True)
    parser.add_argument("--support-features", required=True)
    parser.add_argument("--wheel-features", default="")
    parser.add_argument("--line-features", required=True)
    parser.add_argument("--grasp-feature", required=True)
    parser.add_argument("--contact-facing-feature")
    parser.add_argument("--contact-facing-ramp-frames", type=int, default=12)
    parser.add_argument("--contact-facing-sigma-rad", type=float, default=0.12)
    parser.add_argument("--contact-facing-weight", type=float, default=5.0)
    parser.add_argument("--contact-facing-min-angle-deg", type=float, default=25.0)
    parser.add_argument("--main-body-mask-weight", type=float, default=4.0)
    parser.add_argument("--main-body-center-weight", type=float, default=12.0)
    parser.add_argument("--main-body-center-sigma-px", type=float, default=4.0)
    parser.add_argument("--main-body-aspect-weight", type=float, default=6.0)
    parser.add_argument("--main-body-aspect-sigma", type=float, default=0.05)
    parser.add_argument("--grasp-point-weight", type=float, default=4.0)
    parser.add_argument("--feature-flow-weight", type=float, default=2.0)
    parser.add_argument("--feature-flow-sigma-px", type=float, default=6.0)
    parser.add_argument("--feature-flow-max-distance", type=int, default=12)
    parser.add_argument("--feature-flow-extra-distance-decay-frames", type=float, default=8.0)
    parser.add_argument("--feature-flow-bank-switch-penalty", type=float, default=0.0)
    parser.add_argument("--line-gate-ramp-frames", type=int, default=0)
    parser.add_argument("--mask-principal-axis-sigma-rad", type=float, default=0.07)
    parser.add_argument("--mask-silhouette-sigma-px", type=float, default=6.0)
    parser.add_argument("--mask-silhouette-weight", type=float, default=0.25)
    parser.add_argument("--rotation-acceleration-sigma-rad", type=float, default=0.035)
    parser.add_argument("--translation-jerk-sigma-m", type=float, default=0.0)
    parser.add_argument("--rotation-jerk-sigma-rad", type=float, default=0.0)
    parser.add_argument("--rotation-step-margin-deg", type=float, default=0.5)
    parser.add_argument("--maximum-upright-tilt-deg", type=float, default=18.0)
    parser.add_argument("--upright-tilt-sigma-rad", type=float, default=0.06)
    parser.add_argument("--relative-depth-lag-frames", type=int, default=4)
    parser.add_argument("--relative-depth-order-sigma", type=float, default=0.012)
    parser.add_argument("--relative-depth-order-weight", type=float, default=3.0)
    parser.add_argument("--relative-depth-scale-coupling", type=float, default=0.25)
    parser.add_argument("--relative-depth-shape-sigma", type=float, default=0.03)
    parser.add_argument("--heading-initializer-max-deg", type=float, default=60.0)
    parser.add_argument("--heading-initializer-samples", type=int, default=31)
    parser.add_argument("--heading-direction-window", type=int, default=8)
    parser.add_argument("--heading-direction-sign", type=float, choices=(-1.0, 1.0))
    parser.add_argument("--heading-screen-direction", choices=("clockwise", "counterclockwise"))
    parser.add_argument("--heading-direction-source", default="trusted_boundary_motion")
    parser.add_argument("--heading-direction-weight", type=float, default=8.0)
    parser.add_argument("--heading-reversal-tolerance-deg", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-nfev", type=int, default=100)
    parser.add_argument("--loss", choices=("linear", "soft_l1"), default="soft_l1")
    parser.add_argument("--max-rotation-step-deg", type=float, default=8.0)
    parser.add_argument("--max-translation-step-m", type=float, default=0.12)
    parser.add_argument("--translation-step-weight", type=float, default=40.0)
    parser.add_argument("--translation-step-margin-m", type=float, default=0.005)
    parser.add_argument("--penetration-weight", type=float, default=30.0)
    parser.add_argument("--expand-free-interval", action="store_true")
    args = parser.parse_args()
    if args.heading_direction_sign is not None and args.heading_screen_direction is not None:
        raise ValueError("use either heading-direction-sign or heading-screen-direction, not both")

    sample_dir = args.sample_dir.resolve()
    output = args.output.resolve()
    reference = pd.read_csv(args.reference_pose.resolve()).sort_values("frame").reset_index(drop=True)
    frames = reference.frame.astype(int).to_numpy()
    free_frames = np.arange(args.free_start, args.free_end + 1, dtype=int)
    if not np.array_equal(free_frames, frames[(frames >= args.free_start) & (frames <= args.free_end)]):
        raise ValueError("free interval must be contiguous and present in reference pose")
    if args.expand_free_interval:
        reference.loc[reference.frame.isin(free_frames), "locked"] = 0
        reference.loc[~reference.frame.isin(free_frames), "locked"] = 1
    if np.any(reference.loc[reference.frame.isin(free_frames), "locked"].astype(int) != 0):
        raise ValueError("free interval overlaps a locked reference frame")
    if np.any(reference.loc[~reference.frame.isin(free_frames), "locked"].astype(int) != 1):
        raise ValueError("every non-free frame must be reference locked")

    descriptor = json.loads(args.geometry_descriptor.resolve().read_text())
    evidence_dir = args.rigid_physics_evidence_dir.resolve()
    evidence_manifest_path = evidence_dir / "rigid_physics_evidence_manifest.json"
    evidence_manifest = json.loads(evidence_manifest_path.read_text())
    if not bool(evidence_manifest.get("ready_for_solver", False)):
        failed = sorted(name for name, passed in evidence_manifest.get("gates", {}).items() if not passed)
        raise RuntimeError(f"rigid physics evidence blocks solving: {failed}")
    if [args.free_start, args.free_end] not in evidence_manifest.get("solve_intervals", []):
        raise ValueError("free interval does not match the validated rigid physics evidence")
    silhouette_evidence = pd.read_csv(evidence_dir / "rigid_silhouette_evidence.csv").set_index("frame")
    depth_evidence = pd.read_csv(evidence_dir / "relative_depth_evidence.csv").set_index("frame")
    raw_points = descriptor["feature_points"]
    body = np.asarray(raw_points[args.body_feature], dtype=float)
    support_ids = tuple(item.strip() for item in args.support_features.split(",") if item.strip())
    line_ids = tuple(item.strip() for item in args.line_features.split(",") if item.strip())
    support_groups = [np.asarray(raw_points[item], dtype=float) for item in support_ids]
    wheel_ids = tuple(item.strip() for item in args.wheel_features.split(",") if item.strip())
    wheel_points = [np.asarray(raw_points[item], dtype=float) for item in wheel_ids]
    lines_local = [np.asarray(raw_points[item], dtype=float) for item in line_ids]
    grasp = np.asarray(raw_points[args.grasp_feature], dtype=float)
    if body.shape != (8, 3) or [len(item) for item in support_groups] != [2, 2] or [len(item) for item in lines_local] != [2, 2] or grasp.shape != (1, 3):
        raise ValueError("geometry descriptor does not satisfy the declared rigid feature contract")
    if wheel_points and sum(len(item) for item in wheel_points) != 4:
        raise ValueError("declared visual wheel features must contain exactly four wheel centres")
    upright_axis = np.asarray(
        descriptor.get("initializer", {}).get("upright_axis_local", [0.0, 0.0, 1.0]),
        dtype=float,
    )
    if upright_axis.shape != (3,) or not np.isfinite(upright_axis).all() or np.linalg.norm(upright_axis) < 1e-8:
        raise ValueError("geometry descriptor has an invalid upright_axis_local")
    upright_axis /= np.linalg.norm(upright_axis)

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
    absolute_tracks = tracks[
        (tracks.usable == 1) & tracks.frame.isin(free_frames)
    ].copy()
    aggregate_rows = []
    for (frame, query_id), rows in absolute_tracks.groupby(["frame", "query_id"]):
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

    flow_observations = pd.DataFrame()
    flow_track_path = None
    if args.feature_flow_tracks is not None:
        flow_track_path = args.feature_flow_tracks.resolve()
        flow_tracks = pd.read_csv(flow_track_path)
        required_flow_columns = {
            "flow_usable",
            "anchor_distance_frames",
            "reliability",
            "query_id",
            "feature_kind",
            "anchor_frame",
            "direction",
            "frame",
            "x",
            "y",
            "local_x",
            "local_y",
            "local_z",
        }
        missing_flow_columns = sorted(required_flow_columns - set(flow_tracks.columns))
        if missing_flow_columns:
            raise ValueError(f"feature flow tracks lack fields: {missing_flow_columns}")
        flow_candidates = flow_tracks[
            (flow_tracks.flow_usable == 1)
            & flow_tracks.frame.isin(free_frames)
            & (flow_tracks.anchor_distance_frames <= args.feature_flow_max_distance)
        ].copy()
        anchor_rows = flow_tracks[flow_tracks.frame == flow_tracks.anchor_frame][
            ["query_id", "anchor_frame", "direction", "x", "y"]
        ].rename(columns={"x": "anchor_x", "y": "anchor_y"})
        flow_candidates = flow_candidates.merge(
            anchor_rows,
            on=["query_id", "anchor_frame", "direction"],
            how="inner",
            validate="many_to_one",
        )
        extra_distance_score = (
            1.0
            if args.feature_flow_extra_distance_decay_frames <= 0.0
            else np.exp(
                -flow_candidates.anchor_distance_frames
                / args.feature_flow_extra_distance_decay_frames
            )
        )
        flow_candidates["flow_score"] = (
            flow_candidates.reliability * extra_distance_score
        )
        flow_observations, flow_bank_switch_count = _select_temporally_consistent_flow_banks(
            flow_candidates,
            args.feature_flow_bank_switch_penalty,
        )
    else:
        flow_bank_switch_count = 0

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
    observed_line_confidence_by_frame: dict[int, float] = {}
    unassigned_lines_by_frame: dict[int, dict[str, object]] = {}
    for frame, rows in line_observations.groupby("frame"):
        modes = rows.get("line_observation_mode", pd.Series("paired", index=rows.index)).astype(str)
        paired = rows[modes != "unassigned_axis"]
        if len(paired) == 2:
            observed_lines_by_frame[int(frame)] = paired[["physical_x1", "physical_y1", "physical_x2", "physical_y2"]].to_numpy(float).reshape(2, 2, 2)
            observed_lengths = np.hypot(
                paired.physical_x2.to_numpy(float) - paired.physical_x1.to_numpy(float),
                paired.physical_y2.to_numpy(float) - paired.physical_y1.to_numpy(float),
            )
            observed_line_confidence_by_frame[int(frame)] = float(
                np.mean(paired.endpoint_track_conf.to_numpy(float))
                * min(1.0, float(np.mean(observed_lengths)) / 60.0)
            )
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
                "confidence": float(row.endpoint_track_conf)
                * min(
                    1.0,
                    float(np.hypot(
                        row.physical_x2 - row.physical_x1,
                        row.physical_y2 - row.physical_y1,
                    )) / 60.0,
                ),
                "candidate_indices": candidate_indices,
            }

    line_modes = {
        **{frame: "paired" for frame in observed_lines_by_frame},
        **{frame: "unassigned" for frame in unassigned_lines_by_frame},
    }
    line_gate_weights = {frame: 1.0 for frame in line_modes}
    if args.line_gate_ramp_frames > 0:
        ordered_frames = sorted(line_modes)
        segments: list[list[int]] = []
        for frame in ordered_frames:
            if (
                not segments
                or frame != segments[-1][-1] + 1
                or line_modes[frame] != line_modes[segments[-1][-1]]
            ):
                segments.append([frame])
            else:
                segments[-1].append(frame)
        for segment in segments:
            for index, frame in enumerate(segment):
                start_weight = (
                    1.0
                    if segment[0] == args.free_start
                    else min(1.0, (index + 1) / args.line_gate_ramp_frames)
                )
                end_weight = (
                    1.0
                    if segment[-1] == args.free_end
                    else min(1.0, (len(segment) - index) / args.line_gate_ramp_frames)
                )
                line_gate_weights[frame] = min(start_weight, end_weight)

    temporal_line_assignments: dict[int, tuple[int, int]] = {}
    previous_assigned_lines = None
    for frame in sorted(observed_lines_by_frame):
        observed = observed_lines_by_frame[frame]
        if previous_assigned_lines is None:
            projected = np.asarray(
                [_project(line, initial_rotations[frame], initial_translations[frame], camera) for line in lines_local]
            )
            candidates = []
            for assignment in ((0, 1), (1, 0)):
                cost = sum(
                    float(np.linalg.norm(projected[index] - observed[observed_index], axis=1).mean())
                    for index, observed_index in enumerate(assignment)
                )
                candidates.append((cost, assignment))
        else:
            candidates = []
            for assignment in ((0, 1), (1, 0)):
                current = np.asarray([observed[assignment[0]], observed[assignment[1]]])
                cost = float(np.linalg.norm(current - previous_assigned_lines, axis=2).mean())
                candidates.append((cost, assignment))
        _cost, assignment = min(candidates, key=lambda item: item[0])
        temporal_line_assignments[frame] = assignment
        previous_assigned_lines = np.asarray([observed[assignment[0]], observed[assignment[1]]])

    body_mask_bboxes = {}
    body_mask_axes = {}
    body_mask_anisotropy = {}
    body_mask_boundary_distances = {}
    for frame in free_frames:
        mask_path = sample_dir / "results/segmentation/masks" / f"{frame:05d}_mask.png"
        body_mask_bboxes[frame], _mask_area, body_mask_axes[frame], body_mask_anisotropy[frame] = _mask_body_bbox(mask_path)
        body_mask_boundary_distances[frame] = _mask_body_boundary_distance(mask_path)
    calibration_frame = args.free_start - 1
    calibration_mask_bbox, _area, _axis, _anisotropy = _mask_body_bbox(
        sample_dir / "results/segmentation/masks" / f"{calibration_frame:05d}_mask.png"
    )
    calibration_projected_body = _project(
        body,
        initial_rotations[calibration_frame],
        initial_translations[calibration_frame],
        camera,
    )
    calibration_projected_bbox = np.asarray(
        [
            calibration_projected_body[:, 0].min(),
            calibration_projected_body[:, 1].min(),
            calibration_projected_body[:, 0].max(),
            calibration_projected_body[:, 1].max(),
        ]
    )
    bbox_size_calibration = (
        calibration_projected_bbox[2:] - calibration_projected_bbox[:2]
    ) / np.maximum(calibration_mask_bbox[2:] - calibration_mask_bbox[:2], 1.0)
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

    support_group_by_frame = {}
    for frame in free_frames:
        distances = [
            np.abs((group @ initial_rotations[frame].T + initial_translations[frame]) @ plane_normal + plane_offset).mean()
            for group in support_groups
        ]
        support_group_by_frame[frame] = int(np.argmin(distances))

    frame_to_slot = {frame: index for index, frame in enumerate(free_frames)}
    heading_axis_for_direction = np.asarray(
        descriptor.get("initializer", {}).get("heading_axis_local", [1.0, 0.0, 0.0]),
        dtype=float,
    ) if facing_axis is None else facing_axis
    heading_axis_for_direction /= max(np.linalg.norm(heading_axis_for_direction), 1e-8)
    heading_reference_axis = initial_rotations[args.free_start - 1] @ heading_axis_for_direction
    heading_reference_axis -= plane_normal * float(heading_reference_axis @ plane_normal)
    heading_reference_axis /= max(np.linalg.norm(heading_reference_axis), 1e-8)
    tilt_basis_a = heading_reference_axis
    tilt_basis_b = np.cross(plane_normal, tilt_basis_a)
    tilt_basis_b /= max(np.linalg.norm(tilt_basis_b), 1e-8)

    def signed_heading(rotation: np.ndarray) -> float:
        axis = rotation @ heading_axis_for_direction
        axis -= plane_normal * float(axis @ plane_normal)
        axis /= max(np.linalg.norm(axis), 1e-8)
        return float(np.arctan2(
            plane_normal @ np.cross(heading_reference_axis, axis),
            heading_reference_axis @ axis,
        ))

    boundary_heading_frames = list(
        range(
            max(int(frames.min()), args.free_start - args.heading_direction_window),
            args.free_start,
        )
    )
    boundary_heading_values = np.unwrap(
        np.asarray([signed_heading(initial_rotations[frame]) for frame in boundary_heading_frames])
    )
    boundary_heading_steps = np.diff(boundary_heading_values)
    median_boundary_heading_step = (
        float(np.median(boundary_heading_steps)) if len(boundary_heading_steps) else 0.0
    )
    heading_direction_sign = (
        0.0
        if abs(median_boundary_heading_step) < np.radians(0.1)
        else float(np.sign(median_boundary_heading_step))
    )
    positive_support_screen_angle = 0.0
    if args.heading_screen_direction is not None:
        boundary_frame = args.free_start - 1
        center_camera = initial_translations[boundary_frame]
        axis_camera = initial_rotations[boundary_frame] @ heading_axis_for_direction
        axis_camera -= plane_normal * float(axis_camera @ plane_normal)
        axis_camera /= max(np.linalg.norm(axis_camera), 1e-8)
        rotated_axis_camera = (
            Rotation.from_rotvec(plane_normal * np.radians(1.0)).as_matrix()
            @ axis_camera
        )

        def project_camera_point(point: np.ndarray) -> np.ndarray:
            homogeneous = camera @ point
            return homogeneous[:2] / max(float(homogeneous[2]), 1e-8)

        center_uv = project_camera_point(center_camera)
        vector_before = project_camera_point(center_camera + 0.2 * axis_camera) - center_uv
        vector_after = project_camera_point(center_camera + 0.2 * rotated_axis_camera) - center_uv
        # Image y points down; flip it before measuring ordinary screen CCW.
        vector_before[1] *= -1.0
        vector_after[1] *= -1.0
        positive_support_screen_angle = float(np.arctan2(
            vector_before[0] * vector_after[1] - vector_before[1] * vector_after[0],
            vector_before @ vector_after,
        ))
        positive_appears_counterclockwise = 1.0 if positive_support_screen_angle >= 0.0 else -1.0
        heading_direction_sign = (
            positive_appears_counterclockwise
            if args.heading_screen_direction == "counterclockwise"
            else -positive_appears_counterclockwise
        )
    if args.heading_direction_sign is not None:
        heading_direction_sign = float(args.heading_direction_sign)
    base_heading = np.unwrap(
        np.asarray([signed_heading(initial_rotations[frame]) for frame in free_frames])
    )
    heading_grid = np.radians(
        np.linspace(
            -args.heading_initializer_max_deg,
            args.heading_initializer_max_deg,
            args.heading_initializer_samples,
        )
    )
    heading_unary = np.zeros((len(free_frames), len(heading_grid)), dtype=float)
    for frame_index, frame in enumerate(free_frames):
        raw_target_bbox = body_mask_bboxes[frame]
        target_size = np.maximum(
            (raw_target_bbox[2:] - raw_target_bbox[:2]) * bbox_size_calibration,
            1.0,
        )
        target_log_aspect = float(np.log(target_size[0] / target_size[1]))
        observation_row = object_observations.loc[frame] if frame in object_observations.index else None
        if isinstance(observation_row, pd.DataFrame):
            observation_row = observation_row.iloc[0]
        for heading_index, heading_delta in enumerate(heading_grid):
            candidate_rotation = (
                Rotation.from_rotvec(plane_normal * heading_delta).as_matrix()
                @ initial_rotations[frame]
            )
            projected = _project(body, candidate_rotation, initial_translations[frame], camera)
            predicted_size = np.maximum(projected.max(axis=0) - projected.min(axis=0), 1.0)
            predicted_log_aspect = float(np.log(predicted_size[0] / predicted_size[1]))
            cost = ((predicted_log_aspect - target_log_aspect) / 0.08) ** 2
            if frame in contacts.index:
                contact = contacts.loc[frame]
                if isinstance(contact, pd.DataFrame):
                    contact = contact.iloc[0]
                projected_grasp = _project(
                    grasp,
                    candidate_rotation,
                    initial_translations[frame],
                    camera,
                )[0]
                target_grasp = np.asarray([contact.contact_u, contact.contact_v], dtype=float)
                cost += float(np.sum(np.square((projected_grasp - target_grasp) / 20.0)))
            if (
                args.contact_facing_weight > 0.0
                and facing_axis is not None
                and frame in contacts.index
                and frame in human_reference_by_frame
            ):
                predicted_facing = candidate_rotation @ facing_axis
                desired_facing = human_reference_by_frame[frame] - initial_translations[frame]
                predicted_facing -= plane_normal * float(predicted_facing @ plane_normal)
                desired_facing -= plane_normal * float(desired_facing @ plane_normal)
                predicted_facing /= max(np.linalg.norm(predicted_facing), 1e-8)
                desired_facing /= max(np.linalg.norm(desired_facing), 1e-8)
                face_angle = float(np.arctan2(
                    plane_normal @ np.cross(desired_facing, predicted_facing),
                    desired_facing @ predicted_facing,
                ))
                cost += (face_angle / 0.35) ** 2
            heading_unary[frame_index, heading_index] = cost

    dynamic_cost = heading_unary[0] + np.square(heading_grid / 0.35)
    predecessors = np.zeros((len(free_frames), len(heading_grid)), dtype=int)
    for frame_index in range(1, len(free_frames)):
        transition = np.square(
            (heading_grid[:, None] - heading_grid[None, :]) / 0.12
        )
        if heading_direction_sign != 0.0:
            previous_heading = base_heading[frame_index - 1] + heading_grid[:, None]
            current_heading = base_heading[frame_index] + heading_grid[None, :]
            signed_progress = heading_direction_sign * (current_heading - previous_heading)
            reversal = np.maximum(-signed_progress - np.radians(args.heading_reversal_tolerance_deg), 0.0)
            transition = np.where(
                reversal > 0.0,
                1e12,
                transition,
            )
        total = dynamic_cost[:, None] + transition
        predecessors[frame_index] = np.argmin(total, axis=0)
        dynamic_cost = heading_unary[frame_index] + np.min(total, axis=0)
    dynamic_cost += np.square(heading_grid / 0.35)
    selected_heading_indices = np.zeros(len(free_frames), dtype=int)
    selected_heading_indices[-1] = int(np.argmin(dynamic_cost))
    for frame_index in range(len(free_frames) - 1, 0, -1):
        selected_heading_indices[frame_index - 1] = predecessors[
            frame_index, selected_heading_indices[frame_index]
        ]
    selected_heading = heading_grid[selected_heading_indices]
    selected_absolute_heading = base_heading + selected_heading
    initial_parameters = np.zeros((len(free_frames), 6), dtype=float)
    heading_total_target = 0.0
    # A support-constrained rigid object has a signed heading trajectory.  Store
    # that trajectory as same-sign cumulative increments instead of independent
    # per-frame yaw corrections.  This makes a reversal outside the feasible
    # state space; visual/contact evidence can choose the speed, but not silently
    # switch the winding direction during an occlusion.
    heading_step_cap = np.radians(args.max_rotation_step_deg)
    if heading_direction_sign != 0.0:
        selected_progress = heading_direction_sign * np.diff(
            np.concatenate(([0.0], selected_absolute_heading))
        )
        selected_progress = np.clip(selected_progress, 1e-5, heading_step_cap - 1e-5)
        heading_total_target = float(selected_progress.sum())
        initial_parameters[:, 5] = np.log(selected_progress)
        initial_parameters[:, 5] = np.clip(initial_parameters[:, 5], -7.5, 7.5)
    else:
        initial_parameters[:, 5] = selected_heading
    warm_start_sha256 = None
    if args.warm_start_pose is not None:
        warm_start_path = args.warm_start_pose.resolve()
        warm_start = pd.read_csv(warm_start_path).set_index("frame")
        warm_start_sha256 = hashlib.sha256(warm_start_path.read_bytes()).hexdigest()
        warm_headings = []
        for frame, slot in frame_to_slot.items():
            if frame not in warm_start.index:
                raise ValueError(f"warm start lacks frame {frame}")
            warm_rotation, warm_translation = _pose(warm_start.loc[frame])
            warm_headings.append(signed_heading(warm_rotation))
            initial_parameters[slot, :3] = warm_translation - initial_translations[frame]
            warm_delta = Rotation.from_matrix(
                warm_rotation @ initial_rotations[frame].T
            ).as_rotvec()
            initial_parameters[slot, 3] = float(warm_delta @ tilt_basis_a)
            initial_parameters[slot, 4] = float(warm_delta @ tilt_basis_b)
        if heading_direction_sign != 0.0:
            warm_headings = np.unwrap(np.asarray(warm_headings, dtype=float))
            directed_heading = heading_direction_sign * warm_headings
            # Isotonic projection preserves every already-correct segment while
            # replacing only reverse motion by a hold.  Subsequent optimization
            # can distribute that turn using image/contact evidence.
            directed_heading = np.maximum.accumulate(np.maximum(directed_heading, 0.0))
            directed_steps = np.diff(np.concatenate(([0.0], directed_heading)))
            directed_steps = np.clip(directed_steps, 1e-5, heading_step_cap - 1e-5)
            endpoint_frame = args.free_end + 1
            if endpoint_frame in initial_rotations:
                endpoint_directed_heading = (
                    heading_direction_sign * signed_heading(initial_rotations[endpoint_frame])
                )
                endpoint_directed_heading += 2.0 * np.pi * round(
                    (directed_steps.sum() - endpoint_directed_heading) / (2.0 * np.pi)
                )
                while endpoint_directed_heading < 0.0:
                    endpoint_directed_heading += 2.0 * np.pi
                required = endpoint_directed_heading - directed_steps.sum()
                capacity = heading_step_cap - directed_steps
                if required > capacity.sum() + 1e-8:
                    raise ValueError(
                        "locked endpoint cannot be reached with the declared signed heading step limit"
                    )
                if required > 0.0:
                    directed_steps += required * capacity / capacity.sum()
                elif required < 0.0:
                    removable = directed_steps - 1e-5
                    if -required > removable.sum() + 1e-8:
                        raise ValueError(
                            "locked endpoint requires reversing the declared signed heading direction"
                        )
                    directed_steps += required * removable / removable.sum()
            heading_total_target = float(directed_steps.sum())
            initial_parameters[:, 5] = np.log(directed_steps)
            initial_parameters[:, 5] = np.clip(initial_parameters[:, 5], -7.5, 7.5)
    x0 = initial_parameters.reshape(-1)

    def states(parameters: np.ndarray) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
        values = parameters.reshape(len(free_frames), 6)
        rotations = dict(initial_rotations)
        translations = dict(initial_translations)
        if heading_direction_sign != 0.0:
            heading_step_logits = values[:, 5] - np.max(values[:, 5])
            heading_steps = np.exp(heading_step_logits)
            heading_steps *= heading_total_target / heading_steps.sum()
            constrained_headings = heading_direction_sign * np.cumsum(heading_steps)
        else:
            constrained_headings = values[:, 5]
        for frame, slot in frame_to_slot.items():
            tilt_seed = values[slot, 3] * tilt_basis_a + values[slot, 4] * tilt_basis_b
            tilted_rotation = Rotation.from_rotvec(tilt_seed).as_matrix() @ initial_rotations[frame]
            heading_delta = constrained_headings[slot] - signed_heading(tilted_rotation)
            rotations[frame] = (
                Rotation.from_rotvec(plane_normal * heading_delta).as_matrix()
                @ tilted_rotation
            )
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

        for row in flow_observations.itertuples():
            local = np.asarray([[row.local_x, row.local_y, row.local_z]], dtype=float)
            predicted = _project(local, rotations[row.frame], translations[row.frame], camera)[0]
            predicted_anchor = _project(
                local,
                rotations[int(row.anchor_frame)],
                translations[int(row.anchor_frame)],
                camera,
            )[0]
            tracked_delta = np.asarray(
                [row.x - row.anchor_x, row.y - row.anchor_y], dtype=float
            )
            predicted_delta = predicted - predicted_anchor
            add(
                f"{row.feature_kind}_local_flow",
                int(row.frame),
                args.feature_flow_weight
                * np.sqrt(max(float(row.flow_score), 1e-6))
                * (predicted_delta - tracked_delta)
                / args.feature_flow_sigma_px,
                {
                    "query_id": row.query_id,
                    "anchor_frame": int(row.anchor_frame),
                    "anchor_distance_frames": int(row.anchor_distance_frames),
                },
            )

        for frame in free_frames:
            rotation, translation = rotations[frame], translations[frame]
            if frame in contacts.index:
                contact = contacts.loc[frame]
                if isinstance(contact, pd.DataFrame):
                    contact = contact.iloc[0]
                predicted = _project(grasp, rotation, translation, camera)[0]
                confidence = max(0.1, float(contact.contact_conf))
                add(
                    "grasp_point_reprojection",
                    frame,
                    args.grasp_point_weight
                    * np.sqrt(confidence)
                    * (predicted - [contact.contact_u, contact.contact_v])
                    / 6.0,
                )

                if (
                    args.contact_facing_weight > 0.0
                    and facing_axis is not None
                    and frame in human_reference_by_frame
                ):
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
                    allowed_angle = max(
                        np.radians(args.contact_facing_min_angle_deg),
                        (1.0 - ramp_progress) * boundary_facing_angle,
                    )
                    violation = max(angle - allowed_angle, 0.0)
                    add(
                        "descriptor_contact_facing",
                        frame,
                        np.asarray([
                            args.contact_facing_weight
                            * np.sqrt(confidence)
                            * violation
                            / args.contact_facing_sigma_rad
                        ]),
                    )

            observed = observed_lines_by_frame.get(frame)
            if observed is not None:
                line_confidence = np.sqrt(
                    max(observed_line_confidence_by_frame.get(frame, 0.0), 0.0)
                ) * line_gate_weights.get(frame, 1.0)
                projected_lines = [_project(line, rotation, translation, camera) for line in lines_local]
                assignment = temporal_line_assignments[frame]
                for line_index, observed_index in enumerate(assignment):
                    target = observed[observed_index]
                    direction = target[1] - target[0]
                    direction /= max(np.linalg.norm(direction), 1e-8)
                    normal = np.asarray([-direction[1], direction[0]])
                    predicted = projected_lines[line_index]
                    metadata = {
                        "physical_line_index": line_index,
                        "observed_line_index": observed_index,
                        "assignment_mode": "temporal_identity_continuity",
                    }
                    add("rail_axis_line", frame, line_confidence * ((predicted - target[0]) @ normal) / 5.0, metadata)
                    predicted_direction = predicted[1] - predicted[0]
                    predicted_direction /= max(np.linalg.norm(predicted_direction), 1e-8)
                    cross = predicted_direction[0] * direction[1] - predicted_direction[1] * direction[0]
                    add("rail_direction", frame, np.asarray([line_confidence * cross / 0.10]), metadata)
                if len(projected_lines) == 2 and len(observed) == 2:
                    predicted_separation = float(np.linalg.norm(
                        projected_lines[0].mean(axis=0) - projected_lines[1].mean(axis=0)
                    ))
                    observed_separation = float(np.linalg.norm(
                        observed[0].mean(axis=0) - observed[1].mean(axis=0)
                    ))
                    add(
                        "parallel_line_bundle_separation",
                        frame,
                        np.asarray([line_confidence * (predicted_separation - observed_separation) / 3.0]),
                        {
                            "predicted_separation_px": predicted_separation,
                            "observed_separation_px": observed_separation,
                        },
                    )
            elif frame in unassigned_lines_by_frame:
                declaration = unassigned_lines_by_frame[frame]
                target = np.asarray(declaration["target"], dtype=float)
                direction = target[1] - target[0]
                direction /= max(np.linalg.norm(direction), 1e-8)
                normal = np.asarray([-direction[1], direction[0]])
                confidence = (
                    np.sqrt(max(0.0, float(declaration["confidence"])))
                    * line_gate_weights.get(frame, 1.0)
                )
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
                observation_row = object_observations.loc[frame] if frame in object_observations.index else None
                if isinstance(observation_row, pd.DataFrame):
                    observation_row = observation_row.iloc[0]
                fully_visible = observation_row is not None and str(observation_row.visibility) == "visible"
                if fully_visible and len(candidates) > 1:
                    # A single detected axis on a fully visible parallel-line
                    # bundle is collapse evidence: both physical lines must
                    # project to the same image line.  Choosing only the best
                    # candidate discards the strongest side-view cue.
                    for _cost, line_index, axis_values, direction_value in candidates:
                        metadata = {
                            "selected_candidate_index": line_index,
                            "observation_mode": "visible_collapsed_parallel_bundle",
                        }
                        add("collapsed_parallel_bundle_axis_line", frame, axis_values, metadata)
                        add(
                            "collapsed_parallel_bundle_direction",
                            frame,
                            np.asarray([direction_value]),
                            metadata,
                        )
                else:
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
            target_center = 0.5 * (target_bbox[:2] + target_bbox[2:])
            target_half_size = 0.5 * (target_bbox[2:] - target_bbox[:2]) * bbox_size_calibration
            target_bbox = np.concatenate((target_center - target_half_size, target_center + target_half_size))
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
                predicted_bbox_center = 0.5 * (predicted_bbox[:2] + predicted_bbox[2:])
                target_bbox_center = 0.5 * (target_bbox[:2] + target_bbox[2:])
                add(
                    "visible_rigid_body_center",
                    frame,
                    args.main_body_center_weight
                    * (predicted_bbox_center - target_bbox_center)
                    / args.main_body_center_sigma_px,
                )
                predicted_size = np.maximum(predicted_bbox[2:] - predicted_bbox[:2], 1.0)
                target_size = np.maximum(target_bbox[2:] - target_bbox[:2], 1.0)
                add(
                    "visible_rigid_body_log_aspect",
                    frame,
                    np.asarray([
                        args.main_body_aspect_weight
                        * (
                            np.log(predicted_size[0] / predicted_size[1])
                            - np.log(target_size[0] / target_size[1])
                        )
                        / args.main_body_aspect_sigma
                    ]),
                )
                add(
                    "visible_rigid_body_bbox",
                    frame,
                    np.sqrt(args.main_body_mask_weight)
                    * (predicted_bbox - target_bbox)
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

            predicted_upright = rotation @ upright_axis
            predicted_upright /= max(np.linalg.norm(predicted_upright), 1e-8)
            upright_dot = float(np.clip(predicted_upright @ plane_normal, -1.0, 1.0))
            upright_angle = float(np.arccos(upright_dot))
            upright_excess = max(upright_angle - np.radians(args.maximum_upright_tilt_deg), 0.0)
            add(
                "support_upright_cone",
                frame,
                np.asarray([upright_excess / args.upright_tilt_sigma_rad]),
                {"upright_angle_deg": float(np.degrees(upright_angle))},
            )

            group_index = support_group_by_frame[frame]
            signed_groups = [
                (group @ rotation.T + translation) @ plane_normal + plane_offset
                for group in support_groups
            ]
            add("wheel_support", frame, signed_groups[group_index] / 0.025)
            penetration = np.minimum(np.concatenate(signed_groups), 0.0)
            add(
                "wheel_penetration",
                frame,
                args.penetration_weight * penetration / 0.015,
            )

            depth_anchor_frame = args.free_start - 1
            if (
                frame in depth_evidence.index
                and depth_anchor_frame in depth_evidence.index
                and frame in silhouette_evidence.index
                and bool(silhouette_evidence.loc[frame, "scale_reliable"])
            ):
                observed_shape = float(
                    depth_evidence.loc[frame, "log_depth"]
                    - depth_evidence.loc[depth_anchor_frame, "log_depth"]
                )
                predicted_shape = float(
                    np.log(max(translation[2], 1e-6))
                    - np.log(max(translations[depth_anchor_frame][2], 1e-6))
                )
                add(
                    "relative_depth_shape",
                    frame,
                    np.asarray([
                        (
                            predicted_shape
                            - args.relative_depth_scale_coupling * observed_shape
                        )
                        / args.relative_depth_shape_sigma
                    ]),
                    {
                        "anchor_frame": depth_anchor_frame,
                        "observed_delta_log_depth": observed_shape,
                        "predicted_delta_log_depth": predicted_shape,
                    },
                )

            previous_depth_frame = frame - args.relative_depth_lag_frames
            if (
                frame in depth_evidence.index
                and previous_depth_frame in depth_evidence.index
                and frame in silhouette_evidence.index
                and previous_depth_frame in silhouette_evidence.index
                and bool(silhouette_evidence.loc[frame, "scale_reliable"])
                and bool(silhouette_evidence.loc[previous_depth_frame, "scale_reliable"])
            ):
                observed_delta = float(
                    depth_evidence.loc[frame, "log_depth"]
                    - depth_evidence.loc[previous_depth_frame, "log_depth"]
                )
                if abs(observed_delta) >= 0.004:
                    predicted_delta = float(
                        np.log(max(translation[2], 1e-6))
                        - np.log(max(translations[previous_depth_frame][2], 1e-6))
                    )
                    signed_progress = np.sign(observed_delta) * predicted_delta
                    add(
                        "relative_depth_order",
                        frame,
                        np.asarray([
                            args.relative_depth_order_weight
                            * max(-signed_progress, 0.0)
                            / args.relative_depth_order_sigma
                        ]),
                        {
                            "previous_frame": previous_depth_frame,
                            "observed_delta_log_depth": observed_delta,
                            "predicted_delta_log_depth": predicted_delta,
                        },
                    )

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
            if heading_direction_sign != 0.0:
                previous_axis = rotations[previous] @ heading_axis_for_direction
                current_axis = rotations[frame] @ heading_axis_for_direction
                previous_axis -= plane_normal * float(previous_axis @ plane_normal)
                current_axis -= plane_normal * float(current_axis @ plane_normal)
                previous_axis /= max(np.linalg.norm(previous_axis), 1e-8)
                current_axis /= max(np.linalg.norm(current_axis), 1e-8)
                signed_step = float(np.arctan2(
                    plane_normal @ np.cross(previous_axis, current_axis),
                    previous_axis @ current_axis,
                ))
                reversal = max(
                    -heading_direction_sign * signed_step
                    - np.radians(args.heading_reversal_tolerance_deg),
                    0.0,
                )
                add(
                    "heading_direction_continuity",
                    frame,
                    np.asarray([
                        args.heading_direction_weight * reversal / np.radians(1.0)
                    ]),
                    {
                        "signed_heading_step_deg": float(np.degrees(signed_step)),
                        "expected_direction_sign": heading_direction_sign,
                    },
                )
            translation_step = float(np.linalg.norm(translations[frame] - translations[previous]))
            translation_soft_limit = max(
                0.0,
                args.max_translation_step_m - args.translation_step_margin_m,
            )
            translation_step_excess = max(translation_step - translation_soft_limit, 0.0)
            add(
                "translation_step_limit",
                frame,
                np.asarray([args.translation_step_weight * translation_step_excess / 0.02]),
            )
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

        if args.translation_jerk_sigma_m > 0.0 or args.rotation_jerk_sigma_rad > 0.0:
            for frame in range(args.free_start, args.free_end):
                previous, following, second_following = frame - 1, frame + 1, frame + 2
                if any(
                    value not in rotations
                    for value in (previous, frame, following, second_following)
                ):
                    continue
                if args.translation_jerk_sigma_m > 0.0:
                    translation_jerk = (
                        translations[second_following]
                        - 3.0 * translations[following]
                        + 3.0 * translations[frame]
                        - translations[previous]
                    )
                    add(
                        "translation_jerk",
                        frame,
                        translation_jerk / args.translation_jerk_sigma_m,
                    )
                if args.rotation_jerk_sigma_rad > 0.0:
                    previous_step = Rotation.from_matrix(
                        rotations[previous].T @ rotations[frame]
                    ).as_rotvec()
                    current_step = Rotation.from_matrix(
                        rotations[frame].T @ rotations[following]
                    ).as_rotvec()
                    following_step = Rotation.from_matrix(
                        rotations[following].T @ rotations[second_following]
                    ).as_rotvec()
                    add(
                        "rotation_jerk",
                        frame,
                        (following_step - 2.0 * current_step + previous_step)
                        / args.rotation_jerk_sigma_rad,
                    )

        values = parameters.reshape(len(free_frames), 6)
        for frame, slot in frame_to_slot.items():
            add(
                "bounded_interpolation_prior",
                frame,
                np.concatenate((values[slot, :3] / 0.20, values[slot, 3:5] / 0.45)),
            )
        vector = np.concatenate(blocks)
        return (vector, ledger_rows) if ledger else vector

    lower = np.tile(np.asarray([-0.45, -0.35, -0.55, -1.1, -1.1, -8.0]), len(free_frames))
    upper = np.tile(np.asarray([0.45, 0.35, 0.55, 1.1, 1.1, 8.0]), len(free_frames))
    initial_residual = residual(x0)
    solved = least_squares(
        residual,
        x0,
        bounds=(lower, upper),
        loss=args.loss,
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
    upright_angles = []
    for frame in free_frames:
        for group in support_groups:
            wheel_distances.extend(((group @ rotations[frame].T + translations[frame]) @ plane_normal + plane_offset).tolist())
        current_upright = rotations[frame] @ upright_axis
        current_upright /= max(np.linalg.norm(current_upright), 1e-8)
        upright_angles.append(float(np.degrees(np.arccos(np.clip(current_upright @ plane_normal, -1.0, 1.0)))))
    depth_order_violation_count = 0
    depth_order_pair_count = 0
    depth_rank_observed = []
    depth_rank_predicted = []
    for frame in free_frames:
        previous = frame - args.relative_depth_lag_frames
        if (
            frame not in depth_evidence.index
            or previous not in depth_evidence.index
            or frame not in silhouette_evidence.index
            or previous not in silhouette_evidence.index
            or not bool(silhouette_evidence.loc[frame, "scale_reliable"])
            or not bool(silhouette_evidence.loc[previous, "scale_reliable"])
        ):
            continue
        observed_delta = float(depth_evidence.loc[frame, "log_depth"] - depth_evidence.loc[previous, "log_depth"])
        if abs(observed_delta) < 0.004:
            continue
        predicted_delta = float(
            np.log(max(translations[frame][2], 1e-6))
            - np.log(max(translations[previous][2], 1e-6))
        )
        depth_order_pair_count += 1
        depth_order_violation_count += int(np.sign(observed_delta) * predicted_delta < -0.002)
        depth_rank_observed.append(float(depth_evidence.loc[frame, "log_depth"]))
        depth_rank_predicted.append(float(np.log(max(translations[frame][2], 1e-6))))
    depth_rank_correlation = 1.0
    if len(depth_rank_observed) >= 2:
        observed_rank = pd.Series(depth_rank_observed).rank().to_numpy(float)
        predicted_rank = pd.Series(depth_rank_predicted).rank().to_numpy(float)
        depth_rank_correlation = float(np.corrcoef(observed_rank, predicted_rank)[0, 1])
    solved_heading_values = np.unwrap(
        np.asarray([signed_heading(rotations[frame]) for frame in range(args.free_start - 1, args.free_end + 2)])
    )
    solved_heading_steps = np.diff(solved_heading_values)
    heading_reversal_count = int(np.sum(
        heading_direction_sign * solved_heading_steps
        < -np.radians(args.heading_reversal_tolerance_deg)
    )) if heading_direction_sign != 0.0 else 0
    maximum_translation_step = float(np.max(translation_steps[args.free_start - 2 : args.free_end]))
    maximum_rotation_step = float(np.max(rotation_steps[args.free_start - 2 : args.free_end]))
    maximum_penetration = float(max(0.0, -np.min(wheel_distances)))
    gates = {
        "optimizer_converged": bool(solved.success),
        "locked_reference_exact": locked_exact,
        "translation_step_within_declared_limit": maximum_translation_step <= args.max_translation_step_m,
        "rotation_step_within_declared_limit": maximum_rotation_step <= args.max_rotation_step_deg,
        "wheel_penetration_at_most_0_01m": maximum_penetration <= 0.01,
        "upright_tilt_within_declared_limit": max(upright_angles) <= args.maximum_upright_tilt_deg + 1.0,
        "relative_depth_order_consistent": depth_rank_correlation >= 0.30,
        "heading_direction_continuous": heading_reversal_count == 0,
    }
    quality_passed = all(gates.values())
    metrics = {
        "schema_version": 1,
        "success": bool(solved.success),
        "message": solved.message,
        "function_evaluations": int(solved.nfev),
        "optimizer_loss": args.loss,
        "initial_residual_rms": float(np.sqrt(np.mean(initial_residual * initial_residual))),
        "final_residual_rms": float(np.sqrt(np.mean(final_residual * final_residual))),
        "locked_reference_exact": locked_exact,
        "max_translation_step_m_free_and_boundaries": maximum_translation_step,
        "max_rotation_step_deg_free_and_boundaries": maximum_rotation_step,
        "minimum_wheel_plane_distance_m": float(np.min(wheel_distances)),
        "maximum_wheel_penetration_m": maximum_penetration,
        "maximum_translation_step_declared_m": float(args.max_translation_step_m),
        "translation_step_weight": float(args.translation_step_weight),
        "translation_step_margin_m": float(args.translation_step_margin_m),
        "penetration_weight": float(args.penetration_weight),
        "maximum_upright_tilt_deg": float(max(upright_angles)),
        "relative_depth_order_pair_count": depth_order_pair_count,
        "relative_depth_order_violation_count": depth_order_violation_count,
        "relative_depth_rank_correlation": depth_rank_correlation,
        "boundary_heading_direction_sign": heading_direction_sign,
        "heading_direction_source": args.heading_direction_source,
        "heading_screen_direction": args.heading_screen_direction,
        "positive_support_screen_angle_deg": float(np.degrees(positive_support_screen_angle)),
        "median_boundary_heading_step_deg": float(np.degrees(median_boundary_heading_step)),
        "heading_reversal_count": heading_reversal_count,
        "free_interval_signed_heading_change_deg": float(np.degrees(solved_heading_values[-1] - solved_heading_values[0])),
        "named_feature_observation_count": len(point_observations),
        "feature_flow_observation_count": len(flow_observations),
        "feature_flow_frame_count": int(flow_observations.frame.nunique()) if len(flow_observations) else 0,
        "feature_flow_counts_by_kind": (
            flow_observations.groupby("feature_kind").size().astype(int).to_dict()
            if len(flow_observations)
            else {}
        ),
        "feature_flow_named_feature_count": int(flow_observations.query_id.nunique()) if len(flow_observations) else 0,
        "feature_flow_weight": float(args.feature_flow_weight),
        "feature_flow_sigma_px": float(args.feature_flow_sigma_px),
        "feature_flow_max_distance": int(args.feature_flow_max_distance),
        "feature_flow_extra_distance_decay_frames": float(
            args.feature_flow_extra_distance_decay_frames
        ),
        "feature_flow_bank_switch_penalty": float(args.feature_flow_bank_switch_penalty),
        "feature_flow_bank_switch_count": int(flow_bank_switch_count),
        "feature_flow_tracks_sha256": (
            hashlib.sha256(flow_track_path.read_bytes()).hexdigest()
            if flow_track_path is not None
            else None
        ),
        "contact_facing_feature": args.contact_facing_feature,
        "contact_facing_boundary_angle_deg": float(np.degrees(boundary_facing_angle)),
        "contact_facing_ramp_frames": int(args.contact_facing_ramp_frames),
        "contact_facing_weight": float(args.contact_facing_weight),
        "contact_facing_min_angle_deg": float(args.contact_facing_min_angle_deg),
        "grasp_point_weight": float(args.grasp_point_weight),
        "body_bbox_size_calibration_xy": bbox_size_calibration.tolist(),
        "main_body_mask_weight": float(args.main_body_mask_weight),
        "main_body_center_weight": float(args.main_body_center_weight),
        "main_body_aspect_weight": float(args.main_body_aspect_weight),
        "mask_principal_axis_sigma_rad": float(args.mask_principal_axis_sigma_rad),
        "mask_silhouette_sigma_px": float(args.mask_silhouette_sigma_px),
        "mask_silhouette_weight": float(args.mask_silhouette_weight),
        "rotation_acceleration_sigma_rad": float(args.rotation_acceleration_sigma_rad),
        "translation_jerk_sigma_m": float(args.translation_jerk_sigma_m),
        "rotation_jerk_sigma_rad": float(args.rotation_jerk_sigma_rad),
        "rotation_step_margin_deg": float(args.rotation_step_margin_deg),
        "maximum_upright_tilt_declared_deg": float(args.maximum_upright_tilt_deg),
        "relative_depth_lag_frames": int(args.relative_depth_lag_frames),
        "relative_depth_order_weight": float(args.relative_depth_order_weight),
        "relative_depth_scale_coupling": float(args.relative_depth_scale_coupling),
        "heading_initializer_min_deg": float(np.degrees(selected_heading.min())),
        "heading_initializer_max_deg": float(np.degrees(selected_heading.max())),
        "warm_start_pose_sha256": warm_start_sha256,
        "paired_line_frame_count": len(observed_lines_by_frame),
        "unassigned_line_frame_count": len(unassigned_lines_by_frame),
        "line_gate_ramp_frames": int(args.line_gate_ramp_frames),
        "minimum_active_line_gate_weight": (
            float(min(line_gate_weights.values())) if line_gate_weights else 0.0
        ),
        "case_dispatch_used": False,
        "human_state_optimized": False,
        "accepted_pose_read": False,
        "accepted_pose_written": False,
        "reference_pose_sha256": hashlib.sha256(args.reference_pose.resolve().read_bytes()).hexdigest(),
        "rigid_physics_evidence_manifest_sha256": hashlib.sha256(evidence_manifest_path.read_bytes()).hexdigest(),
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
