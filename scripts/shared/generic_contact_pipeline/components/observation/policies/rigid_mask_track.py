"""Generic rigid-object observations from a tracked mask, center, and depth."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ....core.base.config import CaseProfile
from ....core.base.io import read_csv, write_csv, write_json
from ....core.base.schema import stage_paths
from . import mask_track_center


def _mask_geometry(path: Path, body_width_ratio: float) -> dict[str, float | int]:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"missing rigid-object mask: {path}")
    binary = (mask > 0).astype(np.uint8)
    _count, labels, component_stats, _centroids = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )
    component_areas = [int(value) for value in component_stats[1:, cv2.CC_STAT_AREA]]
    if not component_areas or max(component_areas) < 16:
        raise ValueError(f"rigid-object mask has too few pixels: {path}")
    primary_label = int(np.argmax(np.asarray(component_areas, dtype=int))) + 1
    primary_area = int(component_stats[primary_label, cv2.CC_STAT_AREA])
    px, py, pw, ph = [
        int(component_stats[primary_label, field])
        for field in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT)
    ]
    geometry_mask = labels == primary_label
    association_margin = 0.25 * max(pw, ph)
    for label_index in range(1, _count):
        if label_index == primary_label:
            continue
        x, y, w, h, area = [int(value) for value in component_stats[label_index]]
        if area < max(16, int(round(0.01 * primary_area))):
            continue
        horizontal_gap = max(0, px - (x + w), x - (px + pw))
        vertical_gap = max(0, py - (y + h), y - (py + ph))
        if float(np.hypot(horizontal_gap, vertical_gap)) <= association_margin:
            geometry_mask |= labels == label_index
    ys, xs = np.where(geometry_mask)
    total_area = int(np.count_nonzero(binary))
    x1, x2, y1, y2 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    widths = np.asarray([np.count_nonzero(geometry_mask[y]) for y in range(y1, y2 + 1)])
    threshold = max(2.0, float(widths.max()) * body_width_ratio)
    body_rows = np.where(widths >= threshold)[0] + y1
    body_runs: list[tuple[int, int]] = []
    if len(body_rows):
        run_start = previous = int(body_rows[0])
        for value in body_rows[1:]:
            current = int(value)
            if current != previous + 1:
                body_runs.append((run_start, previous))
                run_start = current
            previous = current
        body_runs.append((run_start, previous))
    body_y1, body_y2 = (
        max(body_runs, key=lambda interval: (interval[1] - interval[0] + 1, interval[1]))
        if body_runs
        else (y1, y2)
    )
    body_ys, body_xs = np.where(geometry_mask & (np.indices(mask.shape)[0] >= body_y1))
    body_x1 = int(body_xs.min()) if len(body_xs) else x1
    body_x2 = int(body_xs.max()) if len(body_xs) else x2
    feature_region = geometry_mask & (np.indices(mask.shape)[0] < body_y1)
    feature_ys, feature_xs = np.where(feature_region)
    feature_visible = int(len(feature_xs) >= 4)
    if feature_visible:
        feature_top = int(feature_ys.min())
        grip_band_height = max(8, int(round(0.08 * max(1, body_y2 - body_y1 + 1))))
        grip_region = feature_region & (np.indices(mask.shape)[0] < feature_top + grip_band_height)
        grip_ys, grip_xs = np.where(grip_region)
        feature_u = float(np.mean(grip_xs))
        feature_v = float(np.mean(grip_ys))
    else:
        feature_u = 0.5 * (body_x1 + body_x2)
        feature_v = float(body_y1)
    bbox_area = max(1, (x2 - x1 + 1) * (y2 - y1 + 1))
    return {
        "bbox_x1": x1, "bbox_y1": y1, "bbox_x2": x2, "bbox_y2": y2,
        "body_x1": body_x1, "body_y1": body_y1, "body_x2": body_x2, "body_y2": body_y2,
        "feature_u": feature_u, "feature_v": feature_v, "feature_visible": feature_visible,
        "area": total_area,
        "bbox_fill_ratio": float(len(xs) / bbox_area),
        "largest_component_ratio": float(primary_area / max(total_area, 1)),
    }


def _parallel_line_pair(
    path: Path,
    geometry: dict[str, float | int],
    config: dict[str, object],
) -> tuple[dict[str, float], dict[str, float]] | None:
    """Recover two visible, near-parallel rigid feature lines from a mask.

    The extractor is geometry-role driven: it only assumes that the requested
    pair occupies the region between the top of the tracked mask and the top of
    the main rigid body.  Object identity is supplied by the asset descriptor,
    not by this observation policy.
    """

    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"missing rigid-object mask: {path}")
    x1 = int(geometry["bbox_x1"])
    x2 = int(geometry["bbox_x2"])
    y1 = int(geometry["bbox_y1"])
    body_y1 = int(geometry["body_y1"])
    feature_height = body_y1 - y1
    if feature_height < int(config.get("minimum_feature_height_px", 24)):
        return None
    margin = int(config.get("roi_margin_px", 15))
    left = max(0, x1 - margin)
    right = min(mask.shape[1], x2 + margin + 1)
    top = max(0, y1 - 5)
    bottom = min(mask.shape[0], body_y1 + margin)
    edges = cv2.Canny(mask[top:bottom, left:right], 20, 80)
    minimum_span = max(
        int(config.get("minimum_line_span_px", 16)),
        int(round(float(config.get("minimum_span_ratio", 0.28)) * feature_height)),
    )
    raw = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 360.0,
        threshold=max(10, minimum_span // 2),
        minLineLength=minimum_span,
        maxLineGap=int(config.get("maximum_line_gap_px", 10)),
    )
    if raw is None:
        return None

    candidates: list[dict[str, float]] = []
    maximum_dx_dy = float(config.get("maximum_dx_dy", 1.2))
    for local_x1, local_y1, local_x2, local_y2 in raw[:, 0]:
        ax, ay = float(local_x1 + left), float(local_y1 + top)
        bx, by = float(local_x2 + left), float(local_y2 + top)
        if ay > by:
            ax, ay, bx, by = bx, by, ax, ay
        dy = by - ay
        dx = bx - ax
        if dy < minimum_span or abs(dx) > maximum_dx_dy * dy:
            continue
        candidates.append({"x1": ax, "y1": ay, "x2": bx, "y2": by, "slope": dx / dy})
    if len(candidates) < 2:
        return None

    def x_at(line: dict[str, float], y: float) -> float:
        return line["x1"] + line["slope"] * (y - line["y1"])

    body_width = max(1.0, float(geometry["body_x2"]) - float(geometry["body_x1"]))
    minimum_separation = max(
        float(config.get("minimum_separation_px", 4.0)),
        float(config.get("minimum_separation_body_ratio", 0.0)) * body_width,
    )
    maximum_separation = float(config.get("maximum_separation_body_ratio", 0.45)) * body_width
    maximum_angle_delta = np.deg2rad(float(config.get("maximum_angle_delta_deg", 10.0)))
    best: tuple[float, dict[str, float], dict[str, float], float, float] | None = None
    for index, first in enumerate(candidates):
        for second in candidates[index + 1 :]:
            overlap_top = max(first["y1"], second["y1"])
            overlap_bottom = min(first["y2"], second["y2"])
            overlap = overlap_bottom - overlap_top
            if overlap < minimum_span:
                continue
            angle_delta = abs(np.arctan(first["slope"]) - np.arctan(second["slope"]))
            if angle_delta > maximum_angle_delta:
                continue
            middle_y = 0.5 * (overlap_top + overlap_bottom)
            separation = abs(x_at(first, middle_y) - x_at(second, middle_y))
            if not minimum_separation <= separation <= maximum_separation:
                continue
            # Prefer long common support and two distinct rails over duplicate
            # Hough edges belonging to the same thin rail.
            score = overlap + 0.35 * separation - 8.0 * angle_delta
            if best is None or score > best[0]:
                best = (score, first, second, overlap_top, overlap_bottom)
    if best is None:
        return None

    _score, first, second, overlap_top, overlap_bottom = best
    middle_y = 0.5 * (overlap_top + overlap_bottom)
    ordered = sorted((first, second), key=lambda line: x_at(line, middle_y))
    angle_delta = abs(np.arctan(ordered[0]["slope"]) - np.arctan(ordered[1]["slope"]))
    span_score = min(1.0, (overlap_bottom - overlap_top) / max(float(feature_height), 1.0))
    angle_score = max(0.0, 1.0 - angle_delta / max(maximum_angle_delta, 1e-6))
    confidence = float(np.clip(0.65 * span_score + 0.35 * angle_score, 0.0, 1.0))
    result = []
    for line in ordered:
        result.append({
            "physical_x1": x_at(line, overlap_top),
            "physical_y1": overlap_top,
            "physical_x2": x_at(line, overlap_bottom),
            "physical_y2": overlap_bottom,
            "endpoint_track_conf": confidence,
        })
    return result[0], result[1]


def _single_feature_axis(
    path: Path,
    geometry: dict[str, float | int],
    config: dict[str, object],
) -> dict[str, float] | None:
    """Recover one visible feature axis without assigning its part identity."""

    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"missing rigid-object mask: {path}")
    x1 = int(geometry["bbox_x1"])
    x2 = int(geometry["bbox_x2"])
    y1 = int(geometry["bbox_y1"])
    body_y1 = int(geometry["body_y1"])
    feature_height = body_y1 - y1
    minimum_feature_height = int(config.get("single_minimum_feature_height_px", 12))
    if feature_height < minimum_feature_height:
        return None
    margin = int(config.get("roi_margin_px", 15))
    left = max(0, x1 - margin)
    right = min(mask.shape[1], x2 + margin + 1)
    top = max(0, y1 - 5)
    bottom = min(mask.shape[0], body_y1 + 4)
    edges = cv2.Canny(mask[top:bottom, left:right], 20, 80)
    minimum_span = max(
        int(config.get("single_minimum_line_span_px", 12)),
        int(round(float(config.get("single_minimum_span_ratio", 0.35)) * feature_height)),
    )
    raw = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 360.0,
        threshold=max(8, minimum_span // 2),
        minLineLength=minimum_span,
        maxLineGap=int(config.get("maximum_line_gap_px", 10)),
    )
    if raw is None:
        return None
    maximum_dx_dy = float(config.get("maximum_dx_dy", 1.2))
    best: tuple[float, dict[str, float]] | None = None
    for local_x1, local_y1, local_x2, local_y2 in raw[:, 0]:
        ax, ay = float(local_x1 + left), float(local_y1 + top)
        bx, by = float(local_x2 + left), float(local_y2 + top)
        if ay > by:
            ax, ay, bx, by = bx, by, ax, ay
        dy = by - ay
        if dy <= 0.0:
            continue
        slope = (bx - ax) / dy
        clipped_y1 = max(float(y1), ay)
        clipped_y2 = min(float(body_y1), by)
        clipped_span = clipped_y2 - clipped_y1
        if clipped_span < minimum_span or abs(slope) > maximum_dx_dy:
            continue
        clipped_x1 = ax + slope * (clipped_y1 - ay)
        clipped_x2 = ax + slope * (clipped_y2 - ay)
        connection_gap = max(0.0, float(body_y1) - clipped_y2)
        score = clipped_span - 0.25 * connection_gap
        confidence = float(np.clip(clipped_span / max(float(feature_height), 1.0), 0.0, 1.0))
        candidate = {
            "physical_x1": clipped_x1,
            "physical_y1": clipped_y1,
            "physical_x2": clipped_x2,
            "physical_y2": clipped_y2,
            "endpoint_track_conf": confidence,
        }
        if best is None or score > best[0]:
            best = (score, candidate)
    return None if best is None else best[1]


def build(profile: CaseProfile) -> dict[str, object]:
    base = mask_track_center.build(profile)
    paths = stage_paths(profile)
    rows = read_csv(paths["object_observations"])
    raw_config = profile.data.get("rigid_mask_observation", {})
    config = dict(raw_config) if isinstance(raw_config, dict) else {}
    body_width_ratio = float(config.get("body_row_width_ratio", 0.45))
    mask_pattern = str(config.get("mask_pattern", "results/segmentation/masks/{frame:05d}_mask.png"))
    geometries = []
    mask_paths: list[Path] = []
    for row in rows:
        frame = int(float(row["frame"]))
        mask_path = profile.sample_dir / mask_pattern.format(frame=frame)
        mask_paths.append(mask_path)
        geometries.append(
            _mask_geometry(
                mask_path,
                body_width_ratio,
            )
        )
    output: list[dict[str, object]] = []
    visibility_counts = {"visible": 0, "partially_visible": 0, "occluded": 0}
    for row, geometry in zip(rows, geometries):
        fill_score = float(np.clip(float(geometry["bbox_fill_ratio"]) / 0.35, 0.0, 1.0))
        component_score = float(
            np.clip((float(geometry["largest_component_ratio"]) - 0.45) / 0.50, 0.0, 1.0)
        )
        mask_reliability = min(fill_score, component_score)
        visibility = (
            "occluded"
            if mask_reliability < 0.45
            else "partially_visible"
            if mask_reliability < 0.85
            else "visible"
        )
        visibility_counts[visibility] += 1
        base_confidence = float(row.get("observation_conf", "1.0") or 1.0)
        observation_confidence = base_confidence * mask_reliability
        feature_visible = int(bool(geometry["feature_visible"]) and visibility != "occluded")
        output.append({
            "frame": row["frame"],
            "time": row["time"],
            "center_x": 0.5 * (float(geometry["body_x1"]) + float(geometry["body_x2"])),
            "center_y": 0.5 * (float(geometry["body_y1"]) + float(geometry["body_y2"])),
            "ref_u": row["ref_u"],
            "ref_v": row["ref_v"],
            "lowest_visible_x": row["support_u"],
            "lowest_visible_y": geometry["bbox_y2"],
            "support_u": row["support_u"],
            "support_v": geometry["bbox_y2"],
            "bbox_x1": geometry["bbox_x1"],
            "bbox_y1": geometry["bbox_y1"],
            "bbox_x2": geometry["bbox_x2"],
            "bbox_y2": geometry["bbox_y2"],
            "body_bbox_x1": geometry["body_x1"],
            "body_bbox_y1": geometry["body_y1"],
            "body_bbox_x2": geometry["body_x2"],
            "body_bbox_y2": geometry["body_y2"],
            "mask_area_px": geometry["area"],
            "mask_conf": f"{observation_confidence:.6f}",
            "observation_conf": f"{observation_confidence:.6f}",
            "visibility": visibility,
            "handle_center_x": geometry["feature_u"],
            "handle_center_y": geometry["feature_v"],
            "handle_visible": feature_visible,
            "handle_conf": f"{0.75 * observation_confidence:.6f}" if feature_visible else "0.0",
            "object_ref_depth_m": row["object_ref_depth_m"],
            "depth_conf": row.get("depth_conf", "1.0"),
            "source": "generic_rigid_mask_track",
        })
    out = write_csv(paths["object_observations"], output)
    parallel_config_raw = config.get("parallel_line_pair")
    line_rows: list[dict[str, object]] = []
    if isinstance(parallel_config_raw, dict):
        parallel_config = dict(parallel_config_raw)
        feature_ids = tuple(str(value) for value in parallel_config.get("feature_ids", ()))
        if len(feature_ids) != 2:
            raise ValueError("parallel_line_pair requires exactly two geometry feature ids")
        semantic_role = str(parallel_config.get("semantic_role", "parallel_rigid_line"))
        for row, geometry, mask_path, object_observation in zip(rows, geometries, mask_paths, output):
            pair = _parallel_line_pair(mask_path, geometry, parallel_config)
            if pair is not None:
                for feature_id, line in zip(feature_ids, pair):
                    line_rows.append({
                        "frame": row["frame"],
                        "time": row["time"],
                        "feature_id": feature_id,
                        "candidate_feature_ids": feature_id,
                        "semantic_role": semantic_role,
                        **line,
                        "line_observation_mode": "paired",
                        "line_observation_trusted": "1",
                        "visibility": "visible_pair",
                        "source": "generic_parallel_rigid_line_pair_from_mask",
                })
                continue
            if object_observation["visibility"] != "visible":
                continue
            single = _single_feature_axis(mask_path, geometry, parallel_config)
            if single is not None:
                line_rows.append({
                    "frame": row["frame"],
                    "time": row["time"],
                    "feature_id": "",
                    "candidate_feature_ids": "|".join(feature_ids),
                    "semantic_role": semantic_role,
                    **single,
                    "line_observation_mode": "unassigned_axis",
                    "line_observation_trusted": "1",
                    "visibility": "visible_single",
                    "source": "generic_unassigned_rigid_line_axis_from_mask",
                })
        line_artifact = profile.result_dir / str(parallel_config.get("artifact", "line_observations.csv"))
        write_csv(line_artifact, line_rows)
    metrics = {
        "component": "rigid_mask_track",
        "base_observation": base,
        "object_observations": str(out),
        "rows": len(output),
        "body_row_width_ratio": body_width_ratio,
        "visibility_counts": visibility_counts,
        "mask_reliability_policy": "bbox_fill_and_largest_component_ratio",
        "parallel_line_rows": len(line_rows),
        "paired_line_rows": sum(row.get("line_observation_mode") == "paired" for row in line_rows),
        "unassigned_axis_rows": sum(row.get("line_observation_mode") == "unassigned_axis" for row in line_rows),
        "case_dispatch_used": False,
    }
    write_json(paths["stage1_metrics"], metrics)
    return metrics
