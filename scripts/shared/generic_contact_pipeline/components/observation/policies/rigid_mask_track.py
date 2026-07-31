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
    ys, xs = np.where(mask > 0)
    if len(xs) < 16:
        raise ValueError(f"rigid-object mask has too few pixels: {path}")
    x1, x2, y1, y2 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    widths = np.asarray([np.count_nonzero(mask[y] > 0) for y in range(y1, y2 + 1)])
    threshold = max(2.0, float(widths.max()) * body_width_ratio)
    body_rows = np.where(widths >= threshold)[0] + y1
    body_y1 = int(body_rows.min()) if len(body_rows) else y1
    body_y2 = int(body_rows.max()) if len(body_rows) else y2
    body_ys, body_xs = np.where((mask > 0) & (np.indices(mask.shape)[0] >= body_y1))
    body_x1 = int(body_xs.min()) if len(body_xs) else x1
    body_x2 = int(body_xs.max()) if len(body_xs) else x2
    feature_ys, feature_xs = np.where((mask > 0) & (np.indices(mask.shape)[0] < body_y1))
    feature_visible = int(len(feature_xs) >= 4)
    feature_u = float(np.mean(feature_xs)) if feature_visible else 0.5 * (body_x1 + body_x2)
    feature_v = float(np.mean(feature_ys)) if feature_visible else float(body_y1)
    return {
        "bbox_x1": x1, "bbox_y1": y1, "bbox_x2": x2, "bbox_y2": y2,
        "body_x1": body_x1, "body_y1": body_y1, "body_x2": body_x2, "body_y2": body_y2,
        "feature_u": feature_u, "feature_v": feature_v, "feature_visible": feature_visible,
        "area": int(len(xs)),
    }


def build(profile: CaseProfile) -> dict[str, object]:
    base = mask_track_center.build(profile)
    paths = stage_paths(profile)
    rows = read_csv(paths["object_observations"])
    raw_config = profile.data.get("rigid_mask_observation", {})
    config = dict(raw_config) if isinstance(raw_config, dict) else {}
    body_width_ratio = float(config.get("body_row_width_ratio", 0.45))
    mask_pattern = str(config.get("mask_pattern", "results/segmentation/masks/{frame:05d}_mask.png"))
    output: list[dict[str, object]] = []
    for row in rows:
        frame = int(float(row["frame"]))
        geometry = _mask_geometry(profile.sample_dir / mask_pattern.format(frame=frame), body_width_ratio)
        output.append({
            "frame": row["frame"],
            "time": row["time"],
            "center_x": row["ref_u"],
            "center_y": row["ref_v"],
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
            "mask_conf": row.get("observation_conf", "1.0"),
            "observation_conf": row.get("observation_conf", "1.0"),
            "handle_center_x": geometry["feature_u"],
            "handle_center_y": geometry["feature_v"],
            "handle_visible": geometry["feature_visible"],
            "handle_conf": "0.75" if geometry["feature_visible"] else "0.0",
            "object_ref_depth_m": row["object_ref_depth_m"],
            "depth_conf": row.get("depth_conf", "1.0"),
            "source": "generic_rigid_mask_track",
        })
    out = write_csv(paths["object_observations"], output)
    metrics = {
        "component": "rigid_mask_track",
        "base_observation": base,
        "object_observations": str(out),
        "rows": len(output),
        "body_row_width_ratio": body_width_ratio,
        "case_dispatch_used": False,
    }
    write_json(paths["stage1_metrics"], metrics)
    return metrics
