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


def _right_hand_evidence(
    path: Path | None, camera: np.ndarray
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    if path is None:
        return {}, {}
    pixels: dict[int, np.ndarray] = {}
    points: dict[int, np.ndarray] = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("site_id") != "right_hand":
                continue
            point = np.asarray([row["x_m"], row["y_m"], row["z_m"]], dtype=float)
            if point[2] > 1e-6:
                frame = int(row["frame"])
                points[frame] = point
                pixels[frame] = _project(point, camera)
    return pixels, points


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
    hands, hand_points = _right_hand_evidence(args.human_sites, camera)
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
        constrained_rotations: list[Rotation] = []
        previous_vector: np.ndarray | None = None
        for index, (frame, rotation) in enumerate(
            zip(range(1, args.frame_count + 1), rotations_by_frame)
        ):
            constrained = _pivot_constrained_rotation(
                rotation,
                hand_points[frame],
                args.handle_local,
                args.blade_center_local,
                observed_pixels_array[index],
                camera,
                reference_vector=previous_vector,
            )
            constrained_rotations.append(constrained)
            previous_vector = constrained.apply(
                args.blade_center_local - args.handle_local
            )
        rotations_by_frame = constrained_rotations
        for index, (frame, rotation) in enumerate(
            zip(range(1, args.frame_count + 1), rotations_by_frame)
        ):
            translation_array[index] = (
                hand_points[frame] - rotation.apply(args.handle_local)
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
                "grasp_gap_m": grasp_gap,
                "source": (
                    "megapose_semantic_branch_cotracker_persistent_grasp_sequence"
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
            (
                "persistent_read_only_hand_to_handle_point_contact"
                if args.persistent_grasp
                else "read_only_human_handle_contact"
            ),
        ],
        "persistent_grasp": args.persistent_grasp,
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
