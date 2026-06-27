from __future__ import annotations

import math
from typing import Any

import numpy as np


def _f(row: dict[str, Any], key: str, default: float = math.nan) -> float:
    try:
        value = row.get(key, "")
        if value in {"", None}:
            return default
        return float(value)
    except Exception:
        return default


def contact_region_uv_from_obs(row: dict[str, Any]) -> tuple[np.ndarray | None, str]:
    side = str(row.get("contact_region_side", "") or row.get("hand_contact_side", "") or row.get("handle_side", "")).lower()
    body = [_f(row, k) for k in ("body_bbox_x1", "body_bbox_y1", "body_bbox_x2", "body_bbox_y2")]
    if not all(np.isfinite(v) for v in body):
        body = [_f(row, k) for k in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")]
    if not all(np.isfinite(v) for v in body):
        return None, "missing_body_bbox_contact_region"
    x1, y1, x2, y2 = body
    w = max(1.0, x2 - x1)
    if side not in {"left", "right"}:
        side = "left" if str(row.get("visual_handle_side", "")).lower() == "left" else "right"
    sign = -1.0 if side == "left" else 1.0
    u = (x1 if side == "left" else x2) + sign * 0.30 * w
    v = 0.5 * (y1 + y2) - 0.04 * max(1.0, y2 - y1)
    return np.asarray([u, v], dtype=float), "generic_body_bbox_contact_region_fallback"
