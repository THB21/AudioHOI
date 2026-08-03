"""Generic persistent grasp candidates for a descriptor-declared rigid feature."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ....core.base.config import CaseProfile
from ....core.base.io import REPO, read_csv, write_csv, write_json
from ....core.base.schema import stage_paths
from ....core.human_sites import extract_gvhmr_site_measurements


def build(profile: CaseProfile) -> dict[str, object]:
    paths = stage_paths(profile)
    observations = read_csv(paths["object_observations"])
    if not observations:
        raise ValueError("persistent feature grasp requires rigid observations")
    raw = profile.data.get("persistent_feature_grasp", {})
    config = dict(raw) if isinstance(raw, dict) else {}
    descriptor_path = REPO / str(profile.data["geometry_asset_descriptor"])
    descriptor = json.loads(descriptor_path.read_text())
    feature_id = str(config.get("geometry_feature_id", "object:handle"))
    points = np.asarray(descriptor["feature_points"][feature_id], dtype=float)
    if points.shape != (1, 3):
        raise ValueError("persistent grasp feature must be one descriptor point")
    frame_times = {int(row["frame"]): float(row["time"]) for row in observations}
    sites = extract_gvhmr_site_measurements(
        sample_id=profile.case_name,
        result_pkl=profile.sample_dir / "results/gvhmr/result.pkl",
        body_models_root=REPO / "third-party/GVHMR/inputs/checkpoints/body_models",
        frame_times=frame_times,
    )
    hands_by_frame: dict[int, list[object]] = {}
    for site in sites.measurements:
        if site.site.body_part == "hand":
            hands_by_frame.setdefault(site.frame, []).append(site)
    ratio_limit = float(config.get("activation_distance_min_bbox_ratio", 1.0))
    release_ratio_limit = float(config.get("release_distance_min_bbox_ratio", ratio_limit))
    acquire_min_frames = int(config.get("acquire_min_frames", 1))
    release_min_frames = int(config.get("release_min_frames", 1))
    if ratio_limit <= 0.0 or release_ratio_limit < ratio_limit:
        raise ValueError("persistent grasp distance ratios must be positive with release >= activation")
    if acquire_min_frames < 1 or release_min_frames < 1:
        raise ValueError("persistent grasp hysteresis lengths must be positive")
    hold_when_occluded = bool(config.get("hold_when_occluded", True))
    persist_until_release = bool(config.get("persist_until_release", False))
    anchor_update_visibility_states = frozenset(
        str(value) for value in config.get("anchor_update_visibility_states", ("visible",))
    )
    offset_update_alpha = float(config.get("image_offset_update_alpha", 0.2))
    if not 0.0 < offset_update_alpha <= 1.0:
        raise ValueError("image_offset_update_alpha must be within (0, 1]")
    acquired_side: str | None = None
    acquire_streak = {"left": 0, "right": 0}
    release_streak = 0
    release_start_index: int | None = None
    stable_image_offset: np.ndarray | None = None
    stable_confidence = 0.0
    output: list[dict[str, object]] = []
    for observation in observations:
        frame = int(observation["frame"])
        target = np.asarray((float(observation["handle_center_x"]), float(observation["handle_center_y"])))
        candidates = hands_by_frame.get(frame, [])
        projected: list[tuple[float, object, np.ndarray]] = []
        for site in candidates:
            x, y, z = site.xyz_m
            if z <= 1e-6:
                continue
            u = profile.camera["fx"] * x / z + profile.camera["cx"]
            v = profile.camera["fy"] * y / z + profile.camera["cy"]
            site_uv = np.asarray((u, v), dtype=float)
            projected.append((float(np.linalg.norm(site_uv - target)), site, site_uv))
        projected.sort(key=lambda item: item[0])
        nearest = projected[0] if projected else None
        extent = min(
            float(observation["body_bbox_x2"]) - float(observation["body_bbox_x1"]),
            float(observation["body_bbox_y2"]) - float(observation["body_bbox_y1"]),
        )
        feature_visible = observation.get("handle_visible", "0") == "1"
        visibility = str(observation.get("visibility", "unknown"))
        initial_active = nearest is not None and extent > 0.0 and nearest[0] <= ratio_limit * extent
        if acquired_side is None:
            distance_by_side = {
                site.site.side: distance
                for distance, site, _site_uv in projected
                if site.site.side in acquire_streak
            }
            for side in acquire_streak:
                acquire_streak[side] = (
                    acquire_streak[side] + 1
                    if extent > 0.0 and distance_by_side.get(side, float("inf")) <= ratio_limit * extent
                    else 0
                )
            ready = [side for side, streak in acquire_streak.items() if streak >= acquire_min_frames]
            if ready:
                acquired_side = min(ready, key=lambda side: distance_by_side.get(side, float("inf")))
                acquire_streak = {"left": 0, "right": 0}
        selected_pair = next(
            ((distance, site, site_uv) for distance, site, site_uv in projected if site.site.side == acquired_side),
            None,
        )
        geometrically_active = selected_pair is not None and extent > 0.0 and selected_pair[0] <= release_ratio_limit * extent
        release_evidence = acquired_side is not None and feature_visible and not geometrically_active
        if release_evidence:
            if release_streak == 0:
                release_start_index = len(output)
            release_streak += 1
        else:
            release_streak = 0
            release_start_index = None
        confirmed_release = acquired_side is not None and release_streak >= release_min_frames
        if confirmed_release:
            # Hysteresis confirms a sustained release at its first evidence frame,
            # rather than delaying the semantic transition until the final vote.
            if release_start_index is not None:
                for index in range(release_start_index, len(output)):
                    output[index]["contact_active"] = "0"
                    output[index]["human_side"] = ""
                    output[index]["anchor_update"] = "0"
                    output[index]["keep_previous"] = "0"
                    output[index]["interaction_state"] = "release" if index == release_start_index else "inactive"
                    output[index]["source"] = "generic_gvhmr_site_release_hysteresis"
            acquired_side = None
            selected_pair = None
            geometrically_active = False
            stable_image_offset = None
            stable_confidence = 0.0
            release_streak = 0
            release_start_index = None
        held_by_memory = acquired_side is not None and (
            persist_until_release or (hold_when_occluded and not feature_visible)
        )
        active = geometrically_active or held_by_memory
        selected = selected_pair[1] if selected_pair is not None else None
        selected_distance = selected_pair[0] if selected_pair is not None else None
        selected_uv = selected_pair[2] if selected_pair is not None else None
        current_confidence = (
            max(0.0, 1.0 - (selected_distance / max(extent * ratio_limit, 1e-6)))
            if selected_distance is not None
            else 0.0
        )
        anchor_update = (
            active
            and feature_visible
            and visibility in anchor_update_visibility_states
        )
        if anchor_update and selected_uv is not None:
            observed_offset = target - selected_uv
            stable_image_offset = (
                observed_offset
                if stable_image_offset is None
                else (1.0 - offset_update_alpha) * stable_image_offset + offset_update_alpha * observed_offset
            )
            observation_confidence = float(observation.get("handle_conf", current_confidence))
            stable_confidence = max(stable_confidence, observation_confidence)
        contact_target = (
            target
            if anchor_update
            else (
                selected_uv + stable_image_offset
                if active and selected_uv is not None and stable_image_offset is not None
                else target
            )
        )
        contact_confidence = (
            max(current_confidence, float(observation.get("handle_conf", 0.0)))
            if anchor_update
            else stable_confidence
        )
        side = selected.site.side if selected is not None else (acquired_side or "")
        interaction_state = (
            "occluded_hold"
            if active and not feature_visible
            else (
                "persistent"
                if active
                else ("release" if confirmed_release and release_min_frames == 1 else "inactive")
            )
        )
        output.append({
            "frame": frame,
            "time": observation["time"],
            "contact_active": "1" if active else "0",
            "human_part": "hand",
            "human_side": side,
            "object_part": str(config.get("object_part", "handle")),
            "object_local_id": feature_id,
            "geometry_feature_id": feature_id,
            "stable_local_x": float(points[0, 0]),
            "stable_local_y": float(points[0, 1]),
            "stable_local_z": float(points[0, 2]),
            "contact_u": float(contact_target[0]),
            "contact_v": float(contact_target[1]),
            "contact_conf": contact_confidence,
            "anchor_score": contact_confidence,
            "contact_depth_offset_m": "",
            "visibility": "visible" if feature_visible else "hidden",
            "anchor_update": "1" if anchor_update else "0",
            "keep_previous": "1" if active and not anchor_update else "0",
            "interaction_state": interaction_state,
            "source": "generic_gvhmr_site_with_persistent_image_contact_offset",
        })
    out = write_csv(paths["contact_candidates"], output)
    metrics = {
        "component": "persistent_feature_grasp",
        "contact_candidates": str(out),
        "rows": len(output),
        "active_rows": sum(row["contact_active"] == "1" for row in output),
        "feature_id": feature_id,
        "gvhmr_read_only": sites.read_only,
        "human_state_optimized": False,
        "persist_until_release": persist_until_release,
        "activation_distance_min_bbox_ratio": ratio_limit,
        "release_distance_min_bbox_ratio": release_ratio_limit,
        "acquire_min_frames": acquire_min_frames,
        "release_min_frames": release_min_frames,
        "image_contact_offset_calibrated": stable_image_offset is not None,
        "anchor_update_visibility_states": sorted(anchor_update_visibility_states),
        "case_dispatch_used": False,
    }
    write_json(paths["stage2_metrics"], metrics)
    return metrics
