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
    hold_when_occluded = bool(config.get("hold_when_occluded", True))
    acquired_side: str | None = None
    output: list[dict[str, object]] = []
    for observation in observations:
        frame = int(observation["frame"])
        target = np.asarray((float(observation["handle_center_x"]), float(observation["handle_center_y"])))
        candidates = hands_by_frame.get(frame, [])
        projected: list[tuple[float, object]] = []
        for site in candidates:
            x, y, z = site.xyz_m
            if z <= 1e-6:
                continue
            u = profile.camera["fx"] * x / z + profile.camera["cx"]
            v = profile.camera["fy"] * y / z + profile.camera["cy"]
            projected.append((float(np.linalg.norm(np.asarray((u, v)) - target)), site))
        projected.sort(key=lambda item: item[0])
        nearest = projected[0] if projected else None
        extent = min(
            float(observation["body_bbox_x2"]) - float(observation["body_bbox_x1"]),
            float(observation["body_bbox_y2"]) - float(observation["body_bbox_y1"]),
        )
        feature_visible = observation.get("handle_visible", "0") == "1"
        initial_active = nearest is not None and extent > 0.0 and nearest[0] <= ratio_limit * extent
        if acquired_side is None and initial_active and nearest is not None:
            acquired_side = nearest[1].site.side
        selected_pair = next(((distance, site) for distance, site in projected if site.site.side == acquired_side), None)
        geometrically_active = selected_pair is not None and extent > 0.0 and selected_pair[0] <= ratio_limit * extent
        held_by_memory = acquired_side is not None and hold_when_occluded and not feature_visible
        active = geometrically_active or held_by_memory
        selected = selected_pair[1] if selected_pair is not None else None
        selected_distance = selected_pair[0] if selected_pair is not None else None
        side = selected.site.side if selected is not None else (acquired_side or "unknown")
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
            "contact_u": float(target[0]),
            "contact_v": float(target[1]),
            "contact_conf": max(0.0, 1.0 - (selected_distance / max(extent * ratio_limit, 1e-6))) if selected_distance is not None else 0.0,
            "anchor_score": max(0.0, 1.0 - (selected_distance / max(extent * ratio_limit, 1e-6))) if selected_distance is not None else 0.0,
            "contact_depth_offset_m": "",
            "visibility": "visible" if feature_visible else "hidden",
            "anchor_update": "1" if geometrically_active and feature_visible else "0",
            "keep_previous": "1" if held_by_memory else "0",
            "source": "generic_gvhmr_feature_proximity_and_occluded_hold",
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
        "case_dispatch_used": False,
    }
    write_json(paths["stage2_metrics"], metrics)
    return metrics
