#!/usr/bin/env python3
"""Build a continuous observed-rigid SE(3) trajectory from pose hypotheses.

The tool is entity and asset neutral.  It selects one temporally coherent
hypothesis branch using mesh-facing semantics, render overlap, and an optional
observed contact site, then interpolates the selected keyframes.  It never
optimizes or writes human state.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from scipy.signal import medfilt, savgol_filter


def _vector(value: str) -> np.ndarray:
    result = np.asarray([float(token) for token in value.split(",")], dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise argparse.ArgumentTypeError("expected three finite comma-separated values")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _camera_point(row: dict[str, object], local: np.ndarray) -> np.ndarray:
    transform = np.asarray(row["T_camera_object"], dtype=float)
    return transform[:3, :3] @ local + transform[:3, 3]


def _project(point: np.ndarray, camera: np.ndarray) -> np.ndarray:
    return np.asarray(
        (
            camera[0, 0] * point[0] / point[2] + camera[0, 2],
            camera[1, 1] * point[1] / point[2] + camera[1, 2],
        ),
        dtype=float,
    )


def _hand_site_evidence(
    path: Path | None, camera: np.ndarray, site_id: str
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    if path is None:
        return {}, {}
    pixels: dict[int, np.ndarray] = {}
    points: dict[int, np.ndarray] = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("site_id") != site_id:
                continue
            point = np.asarray([row["x_m"], row["y_m"], row["z_m"]], dtype=float)
            if point[2] > 1e-6:
                frame = int(row["frame"])
                points[frame] = point
                pixels[frame] = _project(point, camera)
    return pixels, points


def _mask_direction_reference_twist_rotations(
    reference_rotations: list[Rotation],
    *,
    pivots: list[np.ndarray],
    handle_local: np.ndarray,
    observed_local: np.ndarray,
    observed_pixels: np.ndarray,
    camera: np.ndarray,
    face_normal_local: np.ndarray,
) -> list[Rotation]:
    """Use the mask for the grasp axis and pose evidence for its remaining twist."""
    local_y = observed_local - handle_local
    length = float(np.linalg.norm(local_y))
    local_y /= max(length, 1e-12)
    local_z = face_normal_local - float(face_normal_local @ local_y) * local_y
    local_z /= max(np.linalg.norm(local_z), 1e-12)
    local_x = np.cross(local_y, local_z)
    local_x /= max(np.linalg.norm(local_x), 1e-12)
    local_z = np.cross(local_x, local_y)
    local_basis = np.column_stack((local_x, local_y, local_z))

    result: list[Rotation] = []
    previous_y: np.ndarray | None = None
    previous_z: np.ndarray | None = None
    for reference_rotation, pivot, pixel in zip(
        reference_rotations, pivots, observed_pixels
    ):
        ray = np.asarray(
            (
                (pixel[0] - camera[0, 2]) / camera[0, 0],
                (pixel[1] - camera[1, 2]) / camera[1, 1],
                1.0,
            ),
            dtype=float,
        )
        a = float(ray @ ray)
        b = -2.0 * float(ray @ pivot)
        c = float(pivot @ pivot - length**2)
        discriminant = b * b - 4.0 * a * c
        candidates: list[np.ndarray] = []
        if discriminant >= 0.0:
            root = float(np.sqrt(discriminant))
            for depth in ((-b - root) / (2.0 * a), (-b + root) / (2.0 * a)):
                if depth > 1e-6:
                    candidates.append(depth * ray - pivot)
        if not candidates:
            depth = max(1e-6, float(ray @ pivot) / a)
            vector = depth * ray - pivot
            candidates.append(length * vector / max(np.linalg.norm(vector), 1e-12))
        reference_y = (
            previous_y
            if previous_y is not None
            else reference_rotation.apply(local_y)
        )
        target_y = max(
            candidates,
            key=lambda vector: float(
                vector @ reference_y / max(np.linalg.norm(vector), 1e-12)
            ),
        )
        target_y /= max(np.linalg.norm(target_y), 1e-12)

        # The projected mask fixes the blade direction.  MegaPose supplies
        # only the remaining one-dimensional face/twist ambiguity, avoiding
        # both an edge-on paddle and an unconstrained quaternion branch flip.
        target_z = reference_rotation.apply(face_normal_local)
        target_z -= float(target_z @ target_y) * target_y
        if np.linalg.norm(target_z) < 1e-8 and previous_z is not None:
            target_z = previous_z.copy()
        target_z /= max(np.linalg.norm(target_z), 1e-12)
        if previous_z is not None and float(target_z @ previous_z) < 0.0:
            target_z *= -1.0
        target_x = np.cross(target_y, target_z)
        target_x /= max(np.linalg.norm(target_x), 1e-12)
        target_z = np.cross(target_x, target_y)
        target_z /= max(np.linalg.norm(target_z), 1e-12)
        target_basis = np.column_stack((target_x, target_y, target_z))
        result.append(Rotation.from_matrix(target_basis @ local_basis.T))
        previous_y = target_y
        previous_z = target_z
    return result


def _mask_shape(values_x: np.ndarray, values_y: np.ndarray) -> np.ndarray:
    if len(values_x) < 3:
        return np.ones(2, dtype=float)
    covariance = np.cov(np.column_stack((values_x, values_y)).T)
    eigenvalues = np.linalg.eigvalsh(covariance)
    return np.asarray(
        (
            (np.ptp(values_x) + 1.0) / (np.ptp(values_y) + 1.0),
            np.sqrt(max(eigenvalues[0], 1e-6) / max(eigenvalues[1], 1e-6)),
        ),
        dtype=float,
    )


def _silhouette_twist_sequence(
    base_rotations: list[Rotation],
    *,
    pivots: list[np.ndarray],
    handle_local: np.ndarray,
    blade_center_local: np.ndarray,
    face_normal_local: np.ndarray,
    half_width: float,
    half_height: float,
    mask_paths: list[Path],
    camera: np.ndarray,
    step_degrees: float,
    temporal_weight: float,
    maximum_step_degrees: float,
) -> tuple[list[Rotation], np.ndarray, np.ndarray]:
    """Resolve the one-dimensional grasp twist with silhouette shape evidence."""
    twist_axis = blade_center_local - handle_local
    twist_axis /= max(np.linalg.norm(twist_axis), 1e-12)
    width_axis = np.cross(twist_axis, face_normal_local)
    width_axis /= max(np.linalg.norm(width_axis), 1e-12)
    theta = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    outline = (
        blade_center_local[None, :]
        + half_width * np.cos(theta)[:, None] * width_axis[None, :]
        + half_height * np.sin(theta)[:, None] * twist_axis[None, :]
    )
    angles = np.deg2rad(
        np.arange(-90.0, 90.0 + 0.5 * step_degrees, step_degrees)
    )
    frame_rotations: list[list[Rotation]] = []
    emissions = np.empty((len(base_rotations), len(angles)), dtype=float)
    for frame_index, (base_rotation, pivot, mask_path) in enumerate(
        zip(base_rotations, pivots, mask_paths)
    ):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(mask_path)
        target_full = mask > 0
        target_y, target_x = np.nonzero(target_full)
        if len(target_x) < 64:
            raise RuntimeError(f"Insufficient silhouette pixels in {mask_path}")
        x0 = max(0, int(target_x.min()) - 20)
        x1 = min(mask.shape[1] - 1, int(target_x.max()) + 20)
        y0 = max(0, int(target_y.min()) - 20)
        y1 = min(mask.shape[0] - 1, int(target_y.max()) + 20)
        target = target_full[y0 : y1 + 1, x0 : x1 + 1]
        cropped_y, cropped_x = np.nonzero(target)
        target_shape = _mask_shape(cropped_x, cropped_y)
        candidates: list[Rotation] = []
        for candidate_index, angle in enumerate(angles):
            rotation = base_rotation * Rotation.from_rotvec(twist_axis * angle)
            translation = pivot - rotation.apply(handle_local)
            points = rotation.apply(outline) + translation
            projected = np.column_stack(
                (
                    camera[0, 0] * points[:, 0] / points[:, 2] + camera[0, 2],
                    camera[1, 1] * points[:, 1] / points[:, 2] + camera[1, 2],
                )
            )
            prediction = np.zeros(target.shape, dtype=np.uint8)
            hull = cv2.convexHull(
                np.rint(projected - np.asarray((x0, y0))).astype(np.int32)
            )
            cv2.fillConvexPoly(prediction, hull, 1)
            intersection = int(np.logical_and(prediction, target).sum())
            union = int(np.logical_or(prediction, target).sum())
            iou = intersection / max(union, 1)
            predicted_y, predicted_x = np.nonzero(prediction)
            predicted_shape = _mask_shape(predicted_x, predicted_y)
            shape_cost = float(
                np.linalg.norm(
                    np.log(
                        np.maximum(predicted_shape, 1e-3)
                        / np.maximum(target_shape, 1e-3)
                    )
                )
            )
            emissions[frame_index, candidate_index] = 2.0 * (1.0 - iou) + 0.45 * shape_cost
            candidates.append(rotation)
        frame_rotations.append(candidates)

    costs = np.empty_like(emissions)
    backpointers = np.zeros(emissions.shape, dtype=int)
    costs[0] = emissions[0]
    maximum_step = np.deg2rad(maximum_step_degrees)
    for frame_index in range(1, len(base_rotations)):
        for candidate_index, candidate in enumerate(frame_rotations[frame_index]):
            steps = np.asarray(
                [
                    (previous.inv() * candidate).magnitude()
                    for previous in frame_rotations[frame_index - 1]
                ]
            )
            transition = costs[frame_index - 1] + temporal_weight * steps**2
            transition[steps > maximum_step] = 1e9
            previous_index = int(np.argmin(transition))
            costs[frame_index, candidate_index] = (
                emissions[frame_index, candidate_index] + transition[previous_index]
            )
            backpointers[frame_index, candidate_index] = previous_index
    selected_indices = np.zeros(len(base_rotations), dtype=int)
    selected_indices[-1] = int(np.argmin(costs[-1]))
    for frame_index in range(len(base_rotations) - 1, 0, -1):
        selected_indices[frame_index - 1] = backpointers[
            frame_index, selected_indices[frame_index]
        ]
    selected = [
        frame_rotations[frame_index][candidate_index]
        for frame_index, candidate_index in enumerate(selected_indices)
    ]
    return (
        selected,
        np.rad2deg(angles[selected_indices]),
        emissions[np.arange(len(base_rotations)), selected_indices],
    )


def _face_score(row: dict[str, object], normal_local: np.ndarray) -> float:
    transform = np.asarray(row["T_camera_object"], dtype=float)
    normal_camera = transform[:3, :3] @ normal_local
    toward_camera = -transform[:3, 3]
    return float(normal_camera @ toward_camera / np.linalg.norm(toward_camera))


def _rotation(row: dict[str, object]) -> Rotation:
    return Rotation.from_matrix(np.asarray(row["T_camera_object"], dtype=float)[:3, :3])


def _translation(row: dict[str, object]) -> np.ndarray:
    return np.asarray(row["T_camera_object"], dtype=float)[:3, 3]


def _select_branch(
    grouped: dict[int, list[dict[str, object]]],
    *,
    face_normal: np.ndarray,
    handle_local: np.ndarray,
    hand_pixels: dict[int, np.ndarray],
    camera: np.ndarray,
) -> list[dict[str, object]]:
    frames = sorted(grouped)
    costs: list[np.ndarray] = []
    backpointers: list[np.ndarray] = []
    for index, frame in enumerate(frames):
        candidates = grouped[frame]
        emission = np.empty(len(candidates), dtype=float)
        for candidate_index, row in enumerate(candidates):
            iou = float(row.get("official_render_mask_iou", 0.0))
            score = float(row.get("score", 0.0))
            facing = _face_score(row, face_normal)
            # The named visible face must point toward the camera.  Near-edge
            # views are permitted, but the opposite face is strongly rejected.
            face_penalty = 45.0 * max(0.0, -facing) ** 2
            hand_penalty = 0.0
            if frame in hand_pixels:
                handle_uv = _project(_camera_point(row, handle_local), camera)
                hand_penalty = 0.002 * float(np.linalg.norm(handle_uv - hand_pixels[frame]))
            emission[candidate_index] = (
                -5.0 * iou
                - 1.5 * facing
                - 0.08 * score
                + face_penalty
                + hand_penalty
            )
        if index == 0:
            costs.append(emission)
            backpointers.append(np.full(len(candidates), -1, dtype=int))
            continue
        previous = grouped[frames[index - 1]]
        gap = max(1, frame - frames[index - 1])
        current_cost = np.empty(len(candidates), dtype=float)
        current_back = np.empty(len(candidates), dtype=int)
        for candidate_index, row in enumerate(candidates):
            transition_costs = []
            for previous_index, previous_row in enumerate(previous):
                angle_per_frame = (
                    (_rotation(previous_row).inv() * _rotation(row)).magnitude() / gap
                )
                speed = float(np.linalg.norm(_translation(row) - _translation(previous_row)) / gap)
                transition_costs.append(
                    costs[-1][previous_index]
                    + 25.0 * angle_per_frame**2
                    + 3.0 * speed**2
                )
            best_previous = int(np.argmin(transition_costs))
            current_cost[candidate_index] = emission[candidate_index] + transition_costs[best_previous]
            current_back[candidate_index] = best_previous
        costs.append(current_cost)
        backpointers.append(current_back)
    selected_indices = [int(np.argmin(costs[-1]))]
    for index in range(len(frames) - 1, 0, -1):
        selected_indices.append(int(backpointers[index][selected_indices[-1]]))
    selected_indices.reverse()
    return [grouped[frame][candidate_index] for frame, candidate_index in zip(frames, selected_indices)]


def _mask_centroid(path: Path) -> np.ndarray | None:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    ys, xs = np.nonzero(mask > 0)
    if len(xs) < 64:
        return None
    return np.asarray((float(np.median(xs)), float(np.median(ys))), dtype=float)


def _pivot_constrained_rotation(
    rotation: Rotation,
    pivot_camera: np.ndarray,
    handle_local: np.ndarray,
    observed_local: np.ndarray,
    observed_pixel: np.ndarray,
    camera: np.ndarray,
    reference_vector: np.ndarray | None = None,
) -> Rotation:
    current_vector = rotation.apply(observed_local - handle_local)
    length = float(np.linalg.norm(current_vector))
    if length < 1e-8:
        return rotation
    ray = np.asarray(
        (
            (observed_pixel[0] - camera[0, 2]) / camera[0, 0],
            (observed_pixel[1] - camera[1, 2]) / camera[1, 1],
            1.0,
        ),
        dtype=float,
    )
    a = float(ray @ ray)
    b = -2.0 * float(ray @ pivot_camera)
    c = float(pivot_camera @ pivot_camera - length**2)
    discriminant = b * b - 4.0 * a * c
    if discriminant >= 0.0:
        root = float(np.sqrt(discriminant))
        candidates = [(-b - root) / (2.0 * a), (-b + root) / (2.0 * a)]
        positive = [value for value in candidates if value > 1e-6]
        if positive:
            target_candidates = [value * ray - pivot_camera for value in positive]
            reference = current_vector if reference_vector is None else reference_vector
            reference = reference / max(np.linalg.norm(reference), 1e-12)
            current_direction = current_vector / max(np.linalg.norm(current_vector), 1e-12)

            def continuity_score(vector: np.ndarray) -> float:
                direction = vector / max(np.linalg.norm(vector), 1e-12)
                return float(direction @ reference + 0.25 * direction @ current_direction)

            target_vector = max(target_candidates, key=continuity_score)
        else:
            target_vector = current_vector
    else:
        closest_depth = max(1e-6, float(ray @ pivot_camera) / a)
        closest_vector = closest_depth * ray - pivot_camera
        if np.linalg.norm(closest_vector) < 1e-8:
            return rotation
        target_vector = length * closest_vector / np.linalg.norm(closest_vector)
    target_vector *= length / max(np.linalg.norm(target_vector), 1e-12)
    delta, _ = Rotation.align_vectors(
        target_vector[None, :], current_vector[None, :]
    )
    return delta * rotation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hypotheses", type=Path, nargs="+", required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--human-sites", type=Path)
    parser.add_argument(
        "--grasp-site-id",
        default="right_hand",
        help="One immutable observed human site used for the entire rigid grasp.",
    )
    parser.add_argument(
        "--max-rotation-step-deg",
        type=float,
        help="Optional hard validator for a persistent rigid grasp trajectory.",
    )
    parser.add_argument(
        "--silhouette-twist",
        action="store_true",
        help="Resolve rotation about the grasp axis from the observed silhouette shape.",
    )
    parser.add_argument("--silhouette-half-width", type=float)
    parser.add_argument("--silhouette-half-height", type=float)
    parser.add_argument("--silhouette-twist-step-deg", type=float, default=3.0)
    parser.add_argument("--silhouette-temporal-weight", type=float, default=2.0)
    parser.add_argument("--silhouette-max-step-deg", type=float, default=50.0)
    parser.add_argument("--frame-count", type=int, required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--fx", type=float, required=True)
    parser.add_argument("--fy", type=float, required=True)
    parser.add_argument("--cx", type=float, required=True)
    parser.add_argument("--cy", type=float, required=True)
    parser.add_argument("--visible-face-normal", type=_vector, required=True)
    parser.add_argument("--handle-local", type=_vector, required=True)
    parser.add_argument("--blade-center-local", type=_vector, required=True)
    parser.add_argument(
        "--persistent-grasp",
        action="store_true",
        help="Enforce the observed hand site and rigid handle point as the same 3D point on every frame.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    camera = np.asarray(((args.fx, 0.0, args.cx), (0.0, args.fy, args.cy), (0.0, 0.0, 1.0)))
    grouped: dict[int, list[dict[str, object]]] = {}
    for hypotheses_path in args.hypotheses:
        with hypotheses_path.open() as stream:
            for line in stream:
                row = json.loads(line)
                grouped.setdefault(int(row["frame"]), []).append(row)
    hands, hand_points = _hand_site_evidence(
        args.human_sites, camera, args.grasp_site_id
    )
    selected = _select_branch(
        grouped,
        face_normal=args.visible_face_normal,
        handle_local=args.handle_local,
        hand_pixels=hands,
        camera=camera,
    )
    anchor_frames = np.asarray([int(row["frame"]) for row in selected], dtype=float)
    anchor_translations = np.asarray([_translation(row) for row in selected])
    anchor_rotations = Rotation.concatenate([_rotation(row) for row in selected])
    rotation_interpolator = Slerp(anchor_frames, anchor_rotations)

    rotations_by_frame: list[Rotation] = []
    translations_by_frame: list[np.ndarray] = []
    for frame in range(1, args.frame_count + 1):
        query = float(np.clip(frame, anchor_frames[0], anchor_frames[-1]))
        rotation = rotation_interpolator([query])[0]
        translation = np.asarray(
            [np.interp(query, anchor_frames, anchor_translations[:, axis]) for axis in range(3)],
            dtype=float,
        )
        if frame in hand_points:
            rotated_handle = rotation.apply(args.handle_local)
            grasp_consistent_z = hand_points[frame][2] - rotated_handle[2]
            translation[2] = 0.85 * grasp_consistent_z + 0.15 * translation[2]
        blade_camera = rotation.apply(args.blade_center_local) + translation
        centroid = _mask_centroid(args.mask_dir / f"{frame:05d}_mask.png")
        if centroid is not None and blade_camera[2] > 1e-6:
            target_xy = np.asarray(
                (
                    (centroid[0] - args.cx) * blade_camera[2] / args.fx,
                    (centroid[1] - args.cy) * blade_camera[2] / args.fy,
                )
            )
            translation[:2] += target_xy - blade_camera[:2]
        # The hand is evidence for a continuously held handle, but remains
        # read-only and receives only a bounded image-plane correction.
        if frame in hands:
            handle_camera = rotation.apply(args.handle_local) + translation
            handle_uv = _project(handle_camera, camera)
            delta_uv = np.clip(hands[frame] - handle_uv, -18.0, 18.0)
            translation[0] += 0.12 * delta_uv[0] * handle_camera[2] / args.fx
            translation[1] += 0.12 * delta_uv[1] * handle_camera[2] / args.fy
        rotations_by_frame.append(rotation)
        translations_by_frame.append(translation)

    # Reject single-frame mask jumps while retaining the real forehand swing.
    translation_array = np.asarray(translations_by_frame)
    for axis in range(3):
        translation_array[:, axis] = medfilt(translation_array[:, axis], kernel_size=5)
        translation_array[:, axis] = savgol_filter(
            translation_array[:, axis], window_length=7, polyorder=2, mode="interp"
        )

    if args.persistent_grasp:
        missing_hand_frames = [
            frame for frame in range(1, args.frame_count + 1) if frame not in hand_points
        ]
        if missing_hand_frames:
            raise RuntimeError(
                "Persistent grasp requires a 3D observed hand site on every frame; "
                f"missing {missing_hand_frames[:8]}"
            )
        observed_pixels = []
        for frame in range(1, args.frame_count + 1):
            centroid = _mask_centroid(args.mask_dir / f"{frame:05d}_mask.png")
            if centroid is None:
                raise RuntimeError(
                    f"Persistent grasp reprojection requires a rigid observation at frame {frame}"
                )
            observed_pixels.append(centroid)
        observed_pixels_array = np.asarray(observed_pixels, dtype=float)
        for axis in range(2):
            observed_pixels_array[:, axis] = medfilt(
                observed_pixels_array[:, axis], kernel_size=5
            )
            observed_pixels_array[:, axis] = savgol_filter(
                observed_pixels_array[:, axis],
                window_length=7,
                polyorder=2,
                mode="interp",
            )
        reference_rotations = rotations_by_frame
        rotations_by_frame = _mask_direction_reference_twist_rotations(
            reference_rotations,
            pivots=[hand_points[frame] for frame in range(1, args.frame_count + 1)],
            handle_local=args.handle_local,
            observed_local=args.blade_center_local,
            observed_pixels=observed_pixels_array,
            camera=camera,
            face_normal_local=args.visible_face_normal,
        )
        silhouette_twist_degrees = np.zeros(args.frame_count, dtype=float)
        silhouette_emissions = np.full(args.frame_count, np.nan, dtype=float)
        if args.silhouette_twist:
            if args.silhouette_half_width is None or args.silhouette_half_height is None:
                raise ValueError(
                    "--silhouette-twist requires --silhouette-half-width and "
                    "--silhouette-half-height from the geometry descriptor"
                )
            rotations_by_frame, silhouette_twist_degrees, silhouette_emissions = (
                _silhouette_twist_sequence(
                    rotations_by_frame,
                    pivots=[
                        hand_points[frame]
                        for frame in range(1, args.frame_count + 1)
                    ],
                    handle_local=args.handle_local,
                    blade_center_local=args.blade_center_local,
                    face_normal_local=args.visible_face_normal,
                    half_width=args.silhouette_half_width,
                    half_height=args.silhouette_half_height,
                    mask_paths=[
                        args.mask_dir / f"{frame:05d}_mask.png"
                        for frame in range(1, args.frame_count + 1)
                    ],
                    camera=camera,
                    step_degrees=args.silhouette_twist_step_deg,
                    temporal_weight=args.silhouette_temporal_weight,
                    maximum_step_degrees=args.silhouette_max_step_deg,
                )
            )
        for index, (frame, rotation) in enumerate(
            zip(range(1, args.frame_count + 1), rotations_by_frame)
        ):
            translation_array[index] = (
                hand_points[frame] - rotation.apply(args.handle_local)
            )

    rotation_steps_deg = np.rad2deg(
        np.asarray(
            [
                (left.inv() * right).magnitude()
                for left, right in zip(rotations_by_frame[:-1], rotations_by_frame[1:])
            ]
        )
    )
    if (
        args.max_rotation_step_deg is not None
        and len(rotation_steps_deg)
        and float(np.max(rotation_steps_deg)) > args.max_rotation_step_deg
    ):
        raise RuntimeError(
            "Rigid grasp rotation-step validator failed: "
            f"{float(np.max(rotation_steps_deg)):.3f} > "
            f"{args.max_rotation_step_deg:.3f} deg/frame"
        )

    output_rows: list[dict[str, object]] = []
    for frame, (rotation, translation) in enumerate(
        zip(rotations_by_frame, translation_array), start=1
    ):
        qx, qy, qz, qw = rotation.as_quat()
        handle_camera = rotation.apply(args.handle_local) + translation
        grasp_gap = (
            float(np.linalg.norm(handle_camera - hand_points[frame]))
            if frame in hand_points
            else float("nan")
        )
        output_rows.append(
            {
                "frame": frame,
                "time": (frame - 1) / args.fps,
                "tx": float(translation[0]),
                "ty": float(translation[1]),
                "tz": float(translation[2]),
                "qw": float(qw),
                "qx": float(qx),
                "qy": float(qy),
                "qz": float(qz),
                "grasp_active": int(args.persistent_grasp),
                "grasp_site_id": args.grasp_site_id if args.persistent_grasp else "",
                "grasp_gap_m": grasp_gap,
                "silhouette_twist_deg": (
                    float(silhouette_twist_degrees[frame - 1])
                    if args.persistent_grasp and args.silhouette_twist
                    else ""
                ),
                "source": (
                    (
                        "megapose_sam2_silhouette_twist_persistent_grasp_sequence"
                        if args.silhouette_twist
                        else "megapose_semantic_branch_cotracker_persistent_grasp_sequence"
                    )
                    if args.persistent_grasp
                    else "megapose_semantic_branch_cotracker_grasp_sequence"
                ),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    ledger = {
        "schema_version": 1,
        "entity_id": "observed_rigid_entity",
        "state_contract": "root_se3",
        "human_state_optimized": False,
        "selected_keyframes": [
            {
                "frame": int(row["frame"]),
                "hypothesis_rank": int(row["hypothesis_rank"]),
                "render_mask_iou": float(row.get("official_render_mask_iou", 0.0)),
                "visible_face_score": _face_score(row, args.visible_face_normal),
            }
            for row in selected
        ],
        "factors": [
            "megapose_pose_hypothesis",
            "visible_face_semantic_gate",
            "temporal_rotation_continuity",
            "mask_centroid_reprojection",
            *(["mask_silhouette_twist"] if args.silhouette_twist else []),
            (
                "persistent_read_only_hand_to_handle_point_contact"
                if args.persistent_grasp
                else "read_only_human_handle_contact"
            ),
        ],
        "persistent_grasp": args.persistent_grasp,
        "grasp_site_id": args.grasp_site_id if args.persistent_grasp else None,
        "grasp_site_switching_allowed": False,
        "rotation_driver": (
            (
                "mask_shape_twist_with_megapose_prior_and_read_only_hand_pivot"
                if args.silhouette_twist
                else "mask_direction_with_megapose_face_twist_and_read_only_hand_pivot"
            )
            if args.persistent_grasp
            else "selected_pose_hypotheses"
        ),
        "silhouette_twist": args.silhouette_twist,
        "silhouette_emission": (
            {
                "mean": float(np.nanmean(silhouette_emissions)),
                "p95": float(np.nanpercentile(silhouette_emissions, 95)),
            }
            if args.persistent_grasp and args.silhouette_twist
            else None
        ),
        "rotation_step_deg": {
            "median": float(np.median(rotation_steps_deg)),
            "p95": float(np.percentile(rotation_steps_deg, 95)),
            "maximum": float(np.max(rotation_steps_deg)),
            "hard_limit": args.max_rotation_step_deg,
        },
        "grasp_gap_m": {
            "maximum": float(np.nanmax([row["grasp_gap_m"] for row in output_rows])),
            "mean": float(np.nanmean([row["grasp_gap_m"] for row in output_rows])),
        },
        "inputs": {
            "hypotheses": [str(path.resolve()) for path in args.hypotheses],
            "hypotheses_sha256": [_sha256(path) for path in args.hypotheses],
            "human_sites": str(args.human_sites.resolve()) if args.human_sites else None,
        },
        "output": str(args.output.resolve()),
        "output_sha256": _sha256(args.output),
    }
    ledger_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    ledger_path.write_text(json.dumps(ledger, indent=2) + "\n")
    print(json.dumps({**ledger, "manifest": str(ledger_path)}, indent=2))


if __name__ == "__main__":
    main()
