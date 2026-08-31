#!/usr/bin/env python3
"""Combine unary VLM, physical plausibility, and smoothness for an audio ablation.

The script consumes independently computed geometry metrics and blind unary scores.
It never uses audio detections as ground truth. Outputs contain both raw scores and
the paired change (with audio minus without audio).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


MODES = ("vlm_only", "vlm_plus_audio")
MODE_TO_UNARY = {
    "vlm_only": "visual_vlm_only",
    "vlm_plus_audio": "visual_vlm_plus_audio",
}

# Report paper-compatible metrics where available. Higher/lower defines improvement.
METRICS = {
    "vlm_overall_quality_1_to_5": ("higher", "Unary VLM overall quality"),
    "vlm_contact_timing_1_to_5": ("higher", "Unary VLM contact timing"),
    "vlm_contact_location_1_to_5": ("higher", "Unary VLM contact location"),
    "non_collision_percent": ("higher", "Non-collision"),
    "contact_percent_frames": ("higher", "Contact frames"),
    "contact_gap_mm": ("lower", "Contact gap on expected-contact frames"),
    "human_temporal_smoothness_mm": ("lower", "Human temporal smoothness"),
    "object_temporal_smoothness_mm": ("lower", "Object temporal smoothness"),
}


def load_unary(path: Path, case: str) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = json.loads(path.read_text())
    return {
        mode: next((r for r in rows if r.get("case") == case and r.get("mode") == unary_mode), {})
        for mode, unary_mode in MODE_TO_UNARY.items()
    }


def normalize_geometry(raw: dict) -> dict[str, float | None]:
    def percent(name: str) -> float | None:
        value = raw.get(name)
        return round(100.0 * float(value), 3) if isinstance(value, (int, float)) else None

    def millimetres(name: str) -> float | None:
        value = raw.get(name)
        return round(1000.0 * float(value), 3) if isinstance(value, (int, float)) else None

    return {
        "non_collision_percent": percent("hoi_paper_non_collision_score"),
        "contact_percent_frames": percent("hoi_paper_contact_score"),
        "contact_gap_mm": raw.get("contact_gap_mm"),
        "human_temporal_smoothness_mm": millimetres("hoi_paper_human_temporal_smoothness_m"),
        "object_temporal_smoothness_mm": millimetres("hoi_paper_object_temporal_smoothness_m"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", default="back_view_basketball")
    ap.add_argument("--experiment-dir", type=Path,
                    default=Path("deliverables/backward_basketball_vlm_audio_increment_ablation"))
    ap.add_argument("--unary-json", type=Path,
                    default=Path("samples_known_object/hoi_interaction_evaluation/"
                                 "perceptual_unary_audio_ablation/unary_scores.json"))
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    out = args.out_dir or args.experiment_dir / "complete_evaluation"
    out.mkdir(parents=True, exist_ok=True)

    unary = load_unary(args.unary_json, args.case)
    values: dict[str, dict] = {}
    sources = {}
    for mode in MODES:
        geometry_path = args.experiment_dir / mode / "hoi_interaction_metrics.json"
        if not geometry_path.exists():
            raise FileNotFoundError(f"missing geometry metrics: {geometry_path}")
        geometry = json.loads(geometry_path.read_text())
        u = unary.get(mode, {})
        values[mode] = {
            "vlm_overall_quality_1_to_5": u.get("overall_quality"),
            "vlm_contact_timing_1_to_5": u.get("contact_timing"),
            "vlm_contact_location_1_to_5": u.get("contact_location"),
            **normalize_geometry(geometry),
        }
        sources[mode] = {"geometry": str(geometry_path), "unary_blind_id": u.get("blind_id")}

    rows = []
    for metric, (direction, label) in METRICS.items():
        left = values["vlm_only"].get(metric)
        right = values["vlm_plus_audio"].get(metric)
        delta = right - left if isinstance(left, (int, float)) and isinstance(right, (int, float)) else None
        improvement = None
        if delta is not None:
            improvement = delta if direction == "higher" else -delta
        rows.append({
            "metric": metric,
            "label": label,
            "better": direction,
            "without_audio": left,
            "with_audio": right,
            "audio_delta": delta,
            "signed_improvement": improvement,
        })

    fields = list(rows[0])
    with (out / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "case": args.case,
        "comparison": "Visual+VLM versus Visual+VLM+Audio",
        "audio_is_only_intended_factor": True,
        "metric_definitions": {
            "unary_vlm": "blind independent integer score from 1 to 5",
            "non_collision": "mean fraction of SMPL-X vertices outside the object surface",
            "contact": "fraction of valid frames with any SMPL-X vertex on or inside the object surface",
            "contact_gap": "mean hand/object surface gap on expected-contact frames",
            "temporal_smoothness": "mean distance to the same point's two-neighbour temporal average; lower is smoother",
        },
        "warning": "Contact percentage is not a timing-accuracy score. Unary quality is always reported on the required 1-to-5 scale; confidence is retained separately and the current pilot uses one moving-camera view.",
        "sources": sources,
        "rows": rows,
    }
    (out / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
