from __future__ import annotations

import math
from typing import Any

from .schemas import EvaluationPaths, MetricBlock
from .utils import f, load_json, write_json, write_rows


def tradeoff_score(contact_gap_mm: float | None, penetration_depth_mean_mm: float | None, *, sigma_gap: float = 50.0, sigma_pen: float = 10.0) -> float | None:
    if contact_gap_mm is None and penetration_depth_mean_mm is None:
        return None
    gap = max(0.0, contact_gap_mm or 0.0)
    pen = max(0.0, penetration_depth_mean_mm or 0.0)
    physical_contact_score = math.exp(-gap / sigma_gap)
    non_penetration_score = math.exp(-pen / sigma_pen)
    return math.sqrt(physical_contact_score * non_penetration_score)


def compute_penetration_floating_metrics(paths: EvaluationPaths) -> MetricBlock:
    hoi = load_json(paths.hoi_eval_json)
    pen_frame = f(hoi.get("pen_frame_ratio"))
    pen_mean = f(hoi.get("pen_depth_mean_mm"))
    pen_max = f(hoi.get("pen_depth_max_mm"))
    contact_gap = f(hoi.get("contact_gap_mm"))
    metrics: dict[str, Any] = {
        "penetration_frame_ratio": pen_frame,
        "penetration_depth_mean_mm": pen_mean,
        "penetration_depth_max_mm": pen_max,
        "non_collision_ratio": f(hoi.get("non_collision_ratio")),
        "floating_rate": f(hoi.get("floating_rate")),
        "floating_gap_mean_mm": f(hoi.get("floating_gap_mean_mm")),
        "tradeoff_score": tradeoff_score(contact_gap, pen_mean),
        "source": str(paths.hoi_eval_json) if paths.hoi_eval_json.exists() else "missing_hoi_eval",
    }
    out_json = write_json(paths.evaluation_dir / "penetration_floating_metrics.json", metrics)
    out_csv = write_rows(paths.evaluation_dir / "penetration_floating_metrics.csv", [metrics])
    write_rows(
        paths.evaluation_dir / "contact_physics_tradeoff.csv",
        [
            {
                "contact_gap_mm": contact_gap,
                "penetration_depth_mean_mm": pen_mean,
                "tradeoff_score": metrics["tradeoff_score"],
                "interpretation": "balances floating/contact gap against penetration depth",
            }
        ],
    )
    return MetricBlock("penetration_floating", metrics, {"json": str(out_json), "csv": str(out_csv)})
