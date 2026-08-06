#!/usr/bin/env python3
"""Compile audio/geometry impacts and enforce sphere-to-rigid-face contact.

The implementation is capability-driven: one tracked sphere, one observed
rigid face, one static image-side wall corridor, and an audio impulse stream.
Human state is neither required nor modified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import trimesh
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _camera(args: argparse.Namespace) -> np.ndarray:
    return np.asarray(((args.fx, 0.0, args.cx), (0.0, args.fy, args.cy), (0.0, 0.0, 1.0)))


def _uv(point: np.ndarray, camera: np.ndarray) -> np.ndarray:
    projected = camera @ point
    return projected[:2] / projected[2]


def _audio_peaks(path: Path) -> list[int]:
    events = pd.read_csv(path)
    peaks = events[events.event_type.eq("seam_click")].copy()
    peaks = peaks[pd.to_numeric(peaks.audio_score, errors="coerce").fillna(0.0) >= 0.90]
    return sorted({int(value) for value in peaks.audio_frame})


def _mask_gap(mask_path: Path, uv: np.ndarray) -> float:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return float("inf")
    distance = cv2.distanceTransform(np.where(mask > 0, 0, 255).astype(np.uint8), cv2.DIST_L2, 5)
    x = int(np.clip(round(float(uv[0])), 0, mask.shape[1] - 1))
    y = int(np.clip(round(float(uv[1])), 0, mask.shape[0] - 1))
    return float(distance[y, x])


def _classify_impacts(
    peaks: list[int],
    sphere: pd.DataFrame,
    mask_dir: Path,
    camera: np.ndarray,
    maximum_paddle_gap_px: float,
    refractory_frames: int,
) -> tuple[list[int], list[int], dict[int, float]]:
    sphere_by_frame = sphere.set_index("frame")
    gaps: dict[int, float] = {}
    candidate_frames: list[tuple[int, float]] = []
    for peak in peaks:
        neighbors = range(max(1, peak - 1), min(int(sphere.frame.max()), peak + 1) + 1)
        scored = []
        for frame in neighbors:
            row = sphere_by_frame.loc[frame]
            point = row[["tx", "ty", "tz"]].to_numpy(dtype=float)
            gap = _mask_gap(mask_dir / f"{frame:05d}_mask.png", _uv(point, camera))
            scored.append((frame, gap))
        frame, gap = min(scored, key=lambda item: item[1])
        gaps[peak] = gap
        if gap <= maximum_paddle_gap_px:
            candidate_frames.append((frame, gap))
    paddle: list[int] = []
    for frame, gap in candidate_frames:
        if paddle and frame - paddle[-1] < refractory_frames:
            previous_gap = _mask_gap(
                mask_dir / f"{paddle[-1]:05d}_mask.png",
                _uv(sphere_by_frame.loc[paddle[-1]][["tx", "ty", "tz"]].to_numpy(float), camera),
            )
            if gap < previous_gap:
                paddle[-1] = frame
            continue
        paddle.append(frame)
    wall: list[int] = []
    last_frame = int(sphere.frame.max())
    for index, paddle_frame in enumerate(paddle):
        stop = paddle[index + 1] - 2 if index + 1 < len(paddle) else last_frame
        choices = [peak for peak in peaks if paddle_frame + 2 <= peak <= stop]
        if not choices:
            continue
        # In a static wall-practice view, the wall-side event is the impulse
        # whose sphere projection is farthest along the declared corridor.
        wall.append(
            max(
                choices,
                key=lambda frame: float(
                    _uv(
                        sphere_by_frame.loc[frame][["tx", "ty", "tz"]].to_numpy(float),
                        camera,
                    )[0]
                ),
            )
        )
    return paddle, wall, gaps


def _mesh_contact_target(
    sphere_point: np.ndarray,
    rigid_row: pd.Series,
    mesh: trimesh.Trimesh,
    radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rotation = Rotation.from_quat(
        [rigid_row.qx, rigid_row.qy, rigid_row.qz, rigid_row.qw]
    ).as_matrix()
    translation = rigid_row[["tx", "ty", "tz"]].to_numpy(dtype=float)
    local_observation = rotation.T @ (sphere_point - translation)
    closest, _, triangle_ids = trimesh.proximity.closest_point_naive(
        mesh, local_observation[None, :]
    )
    local_surface = closest[0]
    local_direction = local_observation - local_surface
    direction_norm = float(np.linalg.norm(local_direction))
    if direction_norm > 1e-8:
        local_direction /= direction_norm
    else:
        local_direction = np.asarray(mesh.face_normals[int(triangle_ids[0])], dtype=float)
    local_center = local_surface + radius * local_direction
    target = rotation @ local_center + translation
    surface = rotation @ local_surface + translation
    return target, surface, local_surface


def _repair_unsupported_free_flight_loops(
    pose: pd.DataFrame,
    event_frames: list[int],
    *,
    maximum_turn_deg: float = 35.0,
    minimum_progress_reversal: float = 0.02,
    approved_intervals: set[tuple[int, int]] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Replace non-contact loops by a single smooth flight arc.

    A free-flight observation may curve, but it may not reverse progress along
    the event-to-event corridor without another interaction.  Generated videos
    occasionally contain a visually plausible sphere that falls, rises, and
    falls again between two verified impacts.  In those intervals we preserve
    both event endpoints and fit the smallest one-arc model supported by the
    observations.  The quadratic component is restricted to the plane normal
    to the endpoint chord, so progress toward the next interaction stays
    monotonic.  This is entity- and case-independent.
    """
    repaired = pose.copy()
    by_frame = repaired.set_index("frame")
    audit: list[dict[str, object]] = []
    coordinates = ["tx", "ty", "tz"]
    for start, stop in zip(event_frames[:-1], event_frames[1:]):
        if stop - start < 4:
            continue
        points = by_frame.loc[start:stop, coordinates].to_numpy(dtype=float)
        chord = points[-1] - points[0]
        chord_squared = float(chord @ chord)
        if chord_squared <= 1e-10:
            continue
        progress = (points - points[0]) @ chord / chord_squared
        progress_steps = np.diff(progress)
        velocities = np.diff(points, axis=0)
        turns: list[float] = []
        for before, after in zip(velocities[:-1], velocities[1:]):
            denominator = float(np.linalg.norm(before) * np.linalg.norm(after))
            if denominator <= 1e-10:
                turns.append(180.0)
            else:
                cosine = float(np.clip((before @ after) / denominator, -1.0, 1.0))
                turns.append(float(np.degrees(np.arccos(cosine))))
        maximum_observed_turn = max(turns, default=0.0)
        peak_turn_frame = int(start + 1 + int(np.argmax(turns))) if turns else int(start)
        minimum_progress_step = float(progress_steps.min(initial=0.0))
        if (
            minimum_progress_step >= -minimum_progress_reversal
            or maximum_observed_turn <= maximum_turn_deg
        ):
            continue

        vlm_approved = approved_intervals is None or (int(start), int(stop)) in approved_intervals
        if not vlm_approved:
            audit.append(
                {
                    "start_frame": int(start),
                    "stop_frame": int(stop),
                    "peak_turn_frame_before": peak_turn_frame,
                    "reason": "unsupported_free_flight_direction_reversal",
                    "minimum_progress_step_before": minimum_progress_step,
                    "maximum_turn_deg_before": maximum_observed_turn,
                    "repair_applied": False,
                    "gate": "vlm_not_approved",
                }
            )
            continue

        parameter = np.linspace(0.0, 1.0, len(points))
        linear = points[0] + parameter[:, None] * chord
        arc_weight = parameter * (1.0 - parameter)
        curvature = (
            arc_weight[:, None] * (points - linear)
        ).sum(axis=0) / float(arc_weight @ arc_weight)
        chord_axis = chord / np.sqrt(chord_squared)
        curvature -= chord_axis * float(curvature @ chord_axis)
        fitted = linear + arc_weight[:, None] * curvature
        by_frame.loc[start:stop, coordinates] = fitted

        fitted_velocity = np.diff(fitted, axis=0)
        fitted_turns: list[float] = []
        for before, after in zip(fitted_velocity[:-1], fitted_velocity[1:]):
            denominator = float(np.linalg.norm(before) * np.linalg.norm(after))
            cosine = float(np.clip((before @ after) / max(denominator, 1e-10), -1.0, 1.0))
            fitted_turns.append(float(np.degrees(np.arccos(cosine))))
        audit.append(
            {
                "start_frame": int(start),
                "stop_frame": int(stop),
                "peak_turn_frame_before": peak_turn_frame,
                "reason": "unsupported_free_flight_direction_reversal",
                "minimum_progress_step_before": minimum_progress_step,
                "maximum_turn_deg_before": maximum_observed_turn,
                "maximum_turn_deg_after": max(fitted_turns, default=0.0),
                "model": "endpoint_preserving_monotonic_quadratic_arc",
                "repair_applied": True,
                "gate": "physics_only" if approved_intervals is None else "vlm_approved",
            }
        )
    return by_frame.reset_index(), audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sphere-pose", type=Path, required=True)
    parser.add_argument("--rigid-pose", type=Path, required=True)
    parser.add_argument("--audio-events", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--depth-prior", type=Path, required=True)
    parser.add_argument("--rigid-mesh", type=Path, required=True)
    parser.add_argument("--rigid-mesh-units", choices=("m", "mm"), default="m")
    parser.add_argument("--output-pose", type=Path, required=True)
    parser.add_argument("--output-timeline", type=Path, required=True)
    parser.add_argument("--output-contacts", type=Path, required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--fx", type=float, required=True)
    parser.add_argument("--fy", type=float, required=True)
    parser.add_argument("--cx", type=float, required=True)
    parser.add_argument("--cy", type=float, required=True)
    parser.add_argument("--radius", type=float, required=True)
    parser.add_argument("--maximum-paddle-gap-px", type=float, default=35.0)
    parser.add_argument("--paddle-refractory-frames", type=int, default=18)
    parser.add_argument(
        "--free-flight-vlm-decisions",
        type=Path,
        help="Optional Qwen decision JSON. When supplied, only VLM-approved free-flight intervals are repaired.",
    )
    args = parser.parse_args()

    camera = _camera(args)
    rigid_mesh = trimesh.load_mesh(args.rigid_mesh, process=False)
    if isinstance(rigid_mesh, trimesh.Scene):
        rigid_mesh = trimesh.util.concatenate(tuple(rigid_mesh.geometry.values()))
    rigid_mesh.update_faces(rigid_mesh.nondegenerate_faces())
    rigid_mesh.remove_unreferenced_vertices()
    if args.rigid_mesh_units == "mm":
        rigid_mesh.apply_scale(0.001)
    sphere = pd.read_csv(args.sphere_pose)
    rigid = pd.read_csv(args.rigid_pose).set_index("frame")
    depth = pd.read_csv(args.depth_prior).set_index("frame")
    peaks = _audio_peaks(args.audio_events)
    paddle_frames, wall_frames, peak_gaps = _classify_impacts(
        peaks,
        sphere,
        args.mask_dir,
        camera,
        args.maximum_paddle_gap_px,
        args.paddle_refractory_frames,
    )

    sphere_by_frame = sphere.set_index("frame")
    contact_targets: dict[int, np.ndarray] = {}
    contact_normals: dict[int, np.ndarray] = {}
    contact_rows: list[dict[str, object]] = []
    for frame in paddle_frames:
        point = sphere_by_frame.loc[frame][["tx", "ty", "tz"]].to_numpy(dtype=float)
        target, surface, local_surface = _mesh_contact_target(
            point, rigid.loc[frame], rigid_mesh, args.radius
        )
        contact_targets[frame] = target
        contact_normals[frame] = (target - surface) / np.linalg.norm(target - surface)
        contact_rows.append(
            {
                "frame": frame,
                "time": (frame - 1) / args.fps,
                "source_entity": "pingpong_ball",
                "target_entity": "pingpong_paddle",
                "target_feature": "rigid_mesh_surface",
                "contact_mode": "impact",
                "surface_x": surface[0],
                "surface_y": surface[1],
                "surface_z": surface[2],
                "local_surface_x": local_surface[0],
                "local_surface_y": local_surface[1],
                "local_surface_z": local_surface[2],
                "target_center_x": target[0],
                "target_center_y": target[1],
                "target_center_z": target[2],
                "sphere_radius_m": args.radius,
                "expected_surface_gap_m": 0.0,
                "source": "audio_peak_and_rigid_mesh_geometry",
            }
        )

    frames = sphere.frame.astype(int).to_numpy()
    original_points = sphere[["tx", "ty", "tz"]].to_numpy(dtype=float)
    pixels = np.asarray([_uv(point, camera) for point in original_points])
    rays = np.column_stack(
        (
            (pixels[:, 0] - args.cx) / args.fx,
            (pixels[:, 1] - args.cy) / args.fy,
            np.ones(len(pixels)),
        )
    )
    observed_depth = depth.loc[frames].da3_depth_smooth.to_numpy(dtype=float)
    confidence = depth.loc[frames].object_depth_confidence.to_numpy(dtype=float)
    event_frames = set(paddle_frames) | set(wall_frames)

    def residual(z: np.ndarray) -> np.ndarray:
        values = list((np.sqrt(np.maximum(confidence, 0.05)) * (z - observed_depth)).tolist())
        acceleration = z[2:] - 2.0 * z[1:-1] + z[:-2]
        for index, value in enumerate(acceleration, start=2):
            frame = int(frames[index - 1])
            weight = 0.15 if frame in event_frames else 2.0
            values.append(weight * float(value))
        for frame, target in contact_targets.items():
            values.append(18.0 * float(z[frame - 1] - target[2]))
            if 1 < frame < len(frames):
                normal = contact_normals[frame]
                previous_point = rays[frame - 2] * z[frame - 2]
                next_point = rays[frame] * z[frame]
                rigid_previous = rigid.loc[frame - 1][["tx", "ty", "tz"]].to_numpy(float)
                rigid_current = rigid.loc[frame][["tx", "ty", "tz"]].to_numpy(float)
                rigid_next = rigid.loc[frame + 1][["tx", "ty", "tz"]].to_numpy(float)
                pre_normal = float(
                    ((target - previous_point) - (rigid_current - rigid_previous)) @ normal
                )
                post_normal = float(
                    ((next_point - target) - (rigid_next - rigid_current)) @ normal
                )
                values.append(30.0 * (post_normal + 0.80 * pre_normal))
                values.append(60.0 * max(0.0, pre_normal))
                values.append(60.0 * max(0.0, -post_normal))
        return np.asarray(values, dtype=float)

    solved_depth = least_squares(
        residual,
        observed_depth.copy(),
        loss="soft_l1",
        f_scale=0.08,
        max_nfev=120,
    ).x
    # Robust losses may leave a very small impact-sign violation in exchange for
    # matching the depth prior.  Project the adjacent samples back into the
    # physically feasible half spaces while keeping their original image rays.
    # This is geometry generic: no event frame or object identity is special-cased.
    minimum_normal_motion = 0.001
    maximum_depth_projection = 0.03
    for frame, target in contact_targets.items():
        if not 1 < frame < len(frames):
            continue
        normal = contact_normals[frame]
        previous_index = frame - 2
        next_index = frame
        rigid_previous = rigid.loc[frame - 1][["tx", "ty", "tz"]].to_numpy(float)
        rigid_current = rigid.loc[frame][["tx", "ty", "tz"]].to_numpy(float)
        rigid_next = rigid.loc[frame + 1][["tx", "ty", "tz"]].to_numpy(float)

        previous_point = rays[previous_index] * solved_depth[previous_index]
        pre_normal = float(
            ((target - previous_point) - (rigid_current - rigid_previous)) @ normal
        )
        previous_denominator = float(rays[previous_index] @ normal)
        if pre_normal > -minimum_normal_motion and abs(previous_denominator) > 1e-4:
            delta = (pre_normal + minimum_normal_motion) / previous_denominator
            solved_depth[previous_index] += float(
                np.clip(delta, -maximum_depth_projection, maximum_depth_projection)
            )

        next_point = rays[next_index] * solved_depth[next_index]
        post_normal = float(
            ((next_point - target) - (rigid_next - rigid_current)) @ normal
        )
        next_denominator = float(rays[next_index] @ normal)
        if post_normal < minimum_normal_motion and abs(next_denominator) > 1e-4:
            delta = (minimum_normal_motion - post_normal) / next_denominator
            solved_depth[next_index] += float(
                np.clip(delta, -maximum_depth_projection, maximum_depth_projection)
            )
    refined = sphere.copy()
    refined["tz"] = solved_depth
    refined["tx"] = (pixels[:, 0] - args.cx) * solved_depth / args.fx
    refined["ty"] = (pixels[:, 1] - args.cy) * solved_depth / args.fy
    for frame, target in contact_targets.items():
        refined.loc[refined.frame.eq(frame), ["tx", "ty", "tz"]] = target
    ordered_events = sorted(set(paddle_frames) | set(wall_frames))
    approved_intervals: set[tuple[int, int]] | None = None
    if args.free_flight_vlm_decisions is not None:
        decision_payload = json.loads(args.free_flight_vlm_decisions.read_text())
        approved_intervals = {
            (int(row["start_frame"]), int(row["stop_frame"]))
            for row in decision_payload.get("decisions", [])
            if bool(row.get("approved_repair", False))
        }
    refined, free_flight_repairs = _repair_unsupported_free_flight_loops(
        refined, ordered_events, approved_intervals=approved_intervals
    )
    # The arc repair deliberately changes samples adjacent to an impact.  Put
    # those samples back into the contact-normal half spaces so smoothing a
    # corrupted free-flight interval cannot erase the impact reversal.
    refined_by_frame = refined.set_index("frame")
    for frame, target in contact_targets.items():
        if not 1 < frame < len(frames):
            continue
        normal = contact_normals[frame]
        current = refined_by_frame.loc[frame, ["tx", "ty", "tz"]].to_numpy(float)
        previous = refined_by_frame.loc[frame - 1, ["tx", "ty", "tz"]].to_numpy(float)
        following = refined_by_frame.loc[frame + 1, ["tx", "ty", "tz"]].to_numpy(float)
        rigid_previous = rigid.loc[frame - 1, ["tx", "ty", "tz"]].to_numpy(float)
        rigid_current = rigid.loc[frame, ["tx", "ty", "tz"]].to_numpy(float)
        rigid_following = rigid.loc[frame + 1, ["tx", "ty", "tz"]].to_numpy(float)
        pre_normal = float(
            ((current - previous) - (rigid_current - rigid_previous)) @ normal
        )
        if pre_normal >= -minimum_normal_motion:
            previous += (pre_normal + minimum_normal_motion) * normal
            refined_by_frame.loc[frame - 1, ["tx", "ty", "tz"]] = previous
        post_normal = float(
            ((following - current) - (rigid_following - rigid_current)) @ normal
        )
        if post_normal <= minimum_normal_motion:
            following += (minimum_normal_motion - post_normal) * normal
            refined_by_frame.loc[frame + 1, ["tx", "ty", "tz"]] = following
    refined = refined_by_frame.reset_index()
    refined["source"] = "generic_sequence_executor_with_rigid_impact_contact"
    args.output_pose.parent.mkdir(parents=True, exist_ok=True)
    refined.to_csv(args.output_pose, index=False)

    refined_by_frame = refined.set_index("frame")
    impact_audit: list[dict[str, object]] = []
    for row in contact_rows:
        frame = int(row["frame"])
        surface = np.asarray(
            (row["surface_x"], row["surface_y"], row["surface_z"]), dtype=float
        )
        target = np.asarray(
            (row["target_center_x"], row["target_center_y"], row["target_center_z"]),
            dtype=float,
        )
        normal = contact_normals[frame]
        current = refined_by_frame.loc[frame][["tx", "ty", "tz"]].to_numpy(float)
        previous = refined_by_frame.loc[frame - 1][["tx", "ty", "tz"]].to_numpy(float)
        following = refined_by_frame.loc[frame + 1][["tx", "ty", "tz"]].to_numpy(float)
        rigid_previous = rigid.loc[frame - 1][["tx", "ty", "tz"]].to_numpy(float)
        rigid_current = rigid.loc[frame][["tx", "ty", "tz"]].to_numpy(float)
        rigid_following = rigid.loc[frame + 1][["tx", "ty", "tz"]].to_numpy(float)
        surface_gap = float(np.linalg.norm(current - surface) - args.radius)
        pre_normal = float(
            ((current - previous) - (rigid_current - rigid_previous)) @ normal
        )
        post_normal = float(
            ((following - current) - (rigid_following - rigid_current)) @ normal
        )
        reversed_direction = bool(pre_normal < 0.0 < post_normal)
        row["actual_surface_gap_m"] = surface_gap
        row["pre_relative_normal_motion_m"] = pre_normal
        row["post_relative_normal_motion_m"] = post_normal
        row["impact_direction_reversal"] = int(reversed_direction)
        impact_audit.append(
            {
                "frame": frame,
                "surface_gap_m": surface_gap,
                "pre_relative_normal_motion_m": pre_normal,
                "post_relative_normal_motion_m": post_normal,
                "impact_direction_reversal": reversed_direction,
            }
        )

    timeline_rows = []
    paddle_set, wall_set = set(paddle_frames), set(wall_frames)
    for frame in frames:
        part = "paddle_face" if frame in paddle_set else "practice_wall" if frame in wall_set else "none"
        active = int(part != "none")
        timeline_rows.append(
            {
                "frame": int(frame),
                "time": (int(frame) - 1) / args.fps,
                "contact_active": active,
                "contact_label": "audio_timed_impact" if active else "none",
                "contact_part": part,
                "visibility": "visible",
                "source": "joint_audio_rigid_geometry_impact_state",
            }
        )
    pd.DataFrame(timeline_rows).to_csv(args.output_timeline, index=False)
    pd.DataFrame(contact_rows).to_csv(args.output_contacts, index=False)

    manifest = {
        "schema_version": 1,
        "paddle_impact_frames": paddle_frames,
        "wall_impact_frames": wall_frames,
        "audio_peak_count": len(peaks),
        "peak_paddle_mask_gaps_px": {str(key): value for key, value in peak_gaps.items()},
        "paddle_contact_audit": impact_audit,
        "all_paddle_contacts_surface_valid": all(
            abs(float(row["surface_gap_m"])) <= 1e-6 for row in impact_audit
        ),
        "all_paddle_contacts_reverse_normal_motion": all(
            bool(row["impact_direction_reversal"]) for row in impact_audit
        ),
        "free_flight_repairs": free_flight_repairs,
        "free_flight_vlm_decisions": (
            str(args.free_flight_vlm_decisions.resolve())
            if args.free_flight_vlm_decisions is not None
            else None
        ),
        "human_state_optimized": False,
        "inputs": {
            "sphere_pose_sha256": _sha256(args.sphere_pose),
            "rigid_pose_sha256": _sha256(args.rigid_pose),
            "audio_events_sha256": _sha256(args.audio_events),
            "depth_prior_sha256": _sha256(args.depth_prior),
            "rigid_mesh_sha256": _sha256(args.rigid_mesh),
        },
        "outputs": {
            "pose": str(args.output_pose.resolve()),
            "timeline": str(args.output_timeline.resolve()),
            "contacts": str(args.output_contacts.resolve()),
        },
    }
    manifest_path = args.output_pose.with_suffix(args.output_pose.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({**manifest, "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
