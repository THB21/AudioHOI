from __future__ import annotations

from typing import Any

from .schemas import EvaluationPaths, MetricBlock
from .utils import f, load_json, write_json, write_rows


def compute_temporal_audio_metrics(paths: EvaluationPaths) -> MetricBlock:
    hoi = load_json(paths.hoi_eval_json)
    metrics: dict[str, Any] = {
        "object_jerk": f(hoi.get("object_jerk")),
        "accel_at_events": f(hoi.get("accel_at_events")),
        "accel_in_flight": f(hoi.get("accel_in_flight")),
        "contact_ratio_audio_windows": f(hoi.get("contact_ratio_audio_windows")),
        "audio_events": f(hoi.get("audio_events")),
        "source": str(paths.hoi_eval_json) if paths.hoi_eval_json.exists() else "missing_hoi_eval",
    }
    out_json = write_json(paths.evaluation_dir / "temporal_audio_metrics.json", metrics)
    out_csv = write_rows(paths.evaluation_dir / "temporal_audio_metrics.csv", [metrics])
    return MetricBlock("temporal_audio", metrics, {"json": str(out_json), "csv": str(out_csv)})
