#!/usr/bin/env python3
"""Aggregate paired metric deltas for the nine modality-ablation runs."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from nine_modality_ablation import CASES, REPO, load_profile


METRICS = {
    "contact_proxy_score": "higher",
    "non_collision_score": "higher",
    "penetration_ratio": "lower",
    "floating_ratio": "lower",
    "anchor_drift_m": "lower",
    "jump_score": "lower",
    "static_drift_m": "lower",
}


def metric_path(case: str, mode: str) -> Path:
    profile = load_profile(case)
    sample = Path(profile["sample_dir"])
    if not sample.is_absolute():
        sample = REPO / sample
    return sample / "results/scientific_audio_ablation" / mode / "vlm_trace/06_evaluation/metric_scores.json"


def main() -> None:
    out = REPO / "deliverables/nine_case_visual_vlm_audio_ablation/comparison"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    missing = []
    for case in CASES:
        left_path = metric_path(case, "visual_vlm_only")
        right_path = metric_path(case, "visual_vlm_plus_audio")
        if not left_path.exists() or not right_path.exists():
            missing.append({"case": case, "visual_vlm_only": str(left_path), "visual_vlm_plus_audio": str(right_path)})
            continue
        left = json.loads(left_path.read_text())
        right = json.loads(right_path.read_text())
        for metric, direction in METRICS.items():
            lv, rv = left.get(metric), right.get(metric)
            if not isinstance(lv, (int, float)) or not isinstance(rv, (int, float)):
                continue
            delta = float(rv) - float(lv)
            improved = delta > 0 if direction == "higher" else delta < 0
            rows.append({
                "case": case, "metric": metric, "direction": direction,
                "visual_vlm_only": lv, "visual_vlm_plus_audio": rv,
                "audio_delta": delta, "audio_improved": int(improved),
            })
    fields = list(rows[0]) if rows else ["case", "metric", "direction", "visual_vlm_only", "visual_vlm_plus_audio", "audio_delta", "audio_improved"]
    with (out / "paired_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    summary = {"complete_cases": sorted({row["case"] for row in rows}), "missing_cases": missing, "rows": rows}
    (out / "paired_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
