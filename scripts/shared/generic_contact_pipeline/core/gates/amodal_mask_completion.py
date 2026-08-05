"""VLM-selected amodal rigid-mask completion for occlusion intervals."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np

from ..base.config import CaseProfile
from ..base.io import REPO, write_json
from ..base.schema import stage_paths


QUERY_TYPE = "amodal_mask_completion_check"
CHOICES = (
    "completed_from_partial_and_references",
)


def _rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _frame(row: Mapping[str, str]) -> int:
    return int(float(row["frame"]))


def completion_intervals(profile: CaseProfile, *, minimum_gap_frames: int = 3) -> list[tuple[int, int, int, str]]:
    """Return bounded gaps in a declared rigid parallel-line observation."""

    config = profile.data.get("amodal_mask_completion")
    if not isinstance(config, Mapping) or not bool(config.get("enabled", False)):
        return []
    minimum_gap_frames = int(config.get("minimum_gap_frames", minimum_gap_frames))
    observation_rows = _rows(stage_paths(profile)["object_observations"])
    line_rows = _rows(profile.result_dir / "line_observations.csv")
    if not observation_rows or not line_rows:
        return []
    all_frames = sorted({_frame(row) for row in observation_rows})
    line_frames = sorted({_frame(row) for row in line_rows if row.get("line_observation_trusted", "1") == "1"})
    if not line_frames:
        return []
    line_set = set(line_frames)
    candidates = [frame for frame in all_frames if min(line_frames) < frame < max(line_frames) and frame not in line_set]
    if not candidates:
        return []
    times = {_frame(row): row.get("time", "") for row in observation_rows}
    intervals: list[tuple[int, int]] = []
    start = previous = candidates[0]
    for frame in candidates[1:]:
        if frame != previous + 1:
            intervals.append((start, previous))
            start = frame
        previous = frame
    intervals.append((start, previous))
    windows: list[tuple[int, int, int, str]] = []
    for start, end in intervals:
        if end - start + 1 < minimum_gap_frames:
            continue
        keyframes = sorted({
            int(round(start + fraction * (end - start)))
            for fraction in (0.25, 0.5, 0.75)
        })
        windows.extend((frame, start, end, times.get(frame, "")) for frame in keyframes)
    return windows


def _pose_rows(profile: CaseProfile) -> dict[int, dict[str, str]]:
    paths = stage_paths(profile)
    for path in (
        profile.result_dir / "generic_stage4_candidate" / "generic_object_pose_candidate.csv",
        paths["object_pose"],
        paths["object_pose_init"],
    ):
        rows = _rows(path)
        if rows:
            return {_frame(row): row for row in rows}
    return {}


def _quaternion(row: Mapping[str, str]) -> np.ndarray:
    value = np.asarray([float(row[key]) for key in ("qw", "qx", "qy", "qz")], dtype=float)
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        return np.asarray((1.0, 0.0, 0.0, 0.0), dtype=float)
    return value / norm


def _nlerp(first: np.ndarray, second: np.ndarray, alpha: float) -> np.ndarray:
    if float(np.dot(first, second)) < 0.0:
        second = -second
    value = (1.0 - alpha) * first + alpha * second
    return value / max(float(np.linalg.norm(value)), 1e-12)


def _rotation(quaternion: Sequence[float]) -> np.ndarray:
    w, x, y, z = (float(value) for value in quaternion)
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=float)


def _geometry_points(profile: CaseProfile) -> np.ndarray:
    descriptor_path = Path(str(profile.data["geometry_asset_descriptor"]))
    if not descriptor_path.is_absolute():
        descriptor_path = REPO / descriptor_path
    descriptor = json.loads(descriptor_path.read_text())
    raw = descriptor.get("feature_points", {})
    selected = [
        points
        for feature_id, points in raw.items()
        if feature_id in {
            "object:body",
            "object:handle",
            "object:handle_rail_left",
            "object:handle_rail_right",
            "object:support_rear_axle",
            "object:support_front_axle",
        }
    ]
    points = np.asarray([point for group in selected for point in group], dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4:
        raise ValueError("amodal completion requires at least four declared rigid feature points")
    return points


def _boundary_frames(profile: CaseProfile, start: int, end: int) -> tuple[int, int]:
    line_frames = sorted({_frame(row) for row in _rows(profile.result_dir / "line_observations.csv")})
    before = [frame for frame in line_frames if frame < start]
    after = [frame for frame in line_frames if frame > end]
    if not before or not after:
        raise ValueError("amodal completion interval must be bounded by visible line frames")
    return before[-1], after[0]


def _candidate_quaternion(
    choice: str,
    frame: int,
    start: int,
    end: int,
    poses: Mapping[int, Mapping[str, str]],
    before: int,
    after: int,
) -> np.ndarray:
    if choice == "solver_continuation":
        return _quaternion(poses[frame])
    before_q = _quaternion(poses[before])
    if choice == "last_visible_hold":
        return before_q
    if choice == "visible_line_interpolation":
        alpha = (frame - before) / max(float(after - before), 1.0)
        return _nlerp(before_q, _quaternion(poses[after]), alpha)
    raise ValueError(f"unsupported amodal mask candidate: {choice}")


def _mask_for(
    profile: CaseProfile,
    frame: int,
    quaternion: Sequence[float],
    poses: Mapping[int, Mapping[str, str]],
    points_local: np.ndarray,
) -> np.ndarray:
    row = poses[frame]
    translation = np.asarray([float(row[key]) for key in ("tx", "ty", "tz")], dtype=float)
    world = points_local @ _rotation(quaternion).T + translation
    camera = profile.camera
    z = np.maximum(world[:, 2], 1e-6)
    pixels = np.column_stack((
        float(camera["fx"]) * world[:, 0] / z + float(camera["cx"]),
        float(camera["fy"]) * world[:, 1] / z + float(camera["cy"]),
    ))
    frame_path = profile.sample_dir / "frames" / f"{frame:05d}.png"
    image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"missing frame for amodal completion: {frame_path}")
    hull = cv2.convexHull(np.round(pixels).astype(np.int32))
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    return mask


def materialize_candidate_evidence(
    profile: CaseProfile,
    *,
    frame: int,
    start: int,
    end: int,
    out_path: Path,
) -> str:
    before, after = _boundary_frames(profile, start, end)
    panels = []
    for panel_frame, label in (
        (frame, "CURRENT: original partial mask to complete"),
        (before, "REFERENCE: last frame with two visible rails"),
        (after, "REFERENCE: next frame with two visible rails"),
    ):
        source = cv2.imread(str(profile.sample_dir / "frames" / f"{panel_frame:05d}.png"), cv2.IMREAD_COLOR)
        mask = cv2.imread(
            str(profile.sample_dir / "results" / "segmentation" / "masks" / f"{panel_frame:05d}_mask.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        if source is None or mask is None:
            return ""
        overlay = np.zeros_like(source)
        overlay[mask > 0] = (255, 255, 255)
        panel = cv2.addWeighted(source, 1.0, overlay, 0.58, 0.0)
        contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(panel, contours, -1, (40, 230, 255), 3)
        cv2.putText(panel, label, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (40, 230, 255), 2, cv2.LINE_AA)
        panels.append(panel)
    canvas = np.concatenate(panels, axis=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
    return str(out_path)


def materialize_selected_masks(profile: CaseProfile, raw_rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    out_dir = profile.result_dir / "vlm" / "stage4" / "amodal_masks"
    if out_dir.is_dir():
        for stale in out_dir.glob("*.png"):
            stale.unlink()
    written: list[int] = []
    selections: list[dict[str, object]] = []
    approved_intervals: set[tuple[int, int]] = set()
    for row in raw_rows:
        if row.get("query_type") != QUERY_TYPE:
            continue
        choice = str(row.get("label", "unclear"))
        start = int(row.get("start_frame", 0))
        end = int(row.get("end_frame", 0))
        selection = {"start_frame": start, "end_frame": end, "choice": choice, "query_id": row.get("query_id", "")}
        selections.append(selection)
        if choice == "completed_from_partial_and_references":
            approved_intervals.add((start, end))
    observations = {_frame(row): row for row in _rows(stage_paths(profile)["object_observations"])}
    for start, end in sorted(approved_intervals):
        for frame in range(start, end + 1):
            source_mask_path = profile.sample_dir / "results" / "segmentation" / "masks" / f"{frame:05d}_mask.png"
            source_mask = cv2.imread(str(source_mask_path), cv2.IMREAD_GRAYSCALE)
            observation = observations.get(frame)
            if source_mask is None or observation is None:
                continue
            body_y1 = int(float(observation["body_bbox_y1"]))
            body_y2 = int(float(observation["body_bbox_y2"]))
            body_x1 = int(float(observation["body_bbox_x1"]))
            body_x2 = int(float(observation["body_bbox_x2"]))
            body_region = np.zeros_like(source_mask)
            body_region[body_y1 : body_y2 + 1, body_x1 : body_x2 + 1] = source_mask[
                body_y1 : body_y2 + 1,
                body_x1 : body_x2 + 1,
            ]
            rows, cols = np.where(body_region > 0)
            if len(rows) < 16:
                continue
            hull = cv2.convexHull(np.column_stack((cols, rows)).astype(np.int32))
            completed = np.zeros_like(source_mask)
            cv2.fillConvexPoly(completed, hull, 255)
            # The VLM approves completion of the rigid body gap only.  The
            # original mask remains authoritative for rails, handle and wheels.
            completed = cv2.bitwise_or(completed, source_mask)
            out_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_dir / f"{frame:05d}.png"), completed)
            written.append(frame)
    pose_path = stage_paths(profile)["object_pose"]
    manifest = {
        "schema_version": 1,
        "query_type": QUERY_TYPE,
        "pose_artifact": str(pose_path),
        "pose_sha256": hashlib.sha256(pose_path.read_bytes()).hexdigest() if pose_path.is_file() else "",
        "selections": selections,
        "written_frames": sorted(set(written)),
        "continuous_pose_from_vlm": False,
        "pixel_mask_generator": "qwen_approved_body_convex_hull_union_original_partial_mask",
        "vlm_role": "discrete_occlusion_completion_approval_only",
    }
    write_json(out_dir.parent / "amodal_mask_completion_manifest.json", manifest)
    return manifest
