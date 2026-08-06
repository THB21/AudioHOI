#!/usr/bin/env python3
"""Apply a forced-choice identity gate with hard temporal-track validation."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


FIELDS = [
    "frame", "time", "entity_id", "selected_candidate_id", "u", "v", "radius_px",
    "selection_source", "vlm_label", "vlm_confidence", "hard_track_distance_px",
    "hard_validation", "candidate_count", "unique_entity_count",
]


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def run(result_dir: Path, maximum_track_distance: float) -> dict[str, object]:
    candidate_rows = rows(result_dir / "sphere_identity_candidates.csv")
    grouped: dict[int, list[dict[str, str]]] = {}
    for row in candidate_rows:
        grouped.setdefault(int(float(row["frame"])), []).append(row)
    raw_path = result_dir / "vlm/stage1/qwen_raw_results.json"
    raw = json.loads(raw_path.read_text()) if raw_path.is_file() else []
    decisions = {
        int(float(row["frame"])): row
        for row in raw
        if row.get("query_type") == "single_entity_identity_check"
    }
    observed_radii = [
        float(row["radius_px"]) for row in candidate_rows
        if float(row.get("radius_px", 0.0)) > 1.0 and row.get("candidate_id") == "candidate_a"
    ]
    fallback_radius = float(np.median(observed_radii)) if observed_radii else 8.0
    selected: list[dict[str, object]] = []
    vlm_accepted = 0
    vlm_rejected = 0
    fallback_frames: list[int] = []
    for frame in sorted(grouped):
        candidates = {row["candidate_id"]: row for row in grouped[frame]}
        anchor = candidates["candidate_a"]
        decision = decisions.get(frame, {})
        label = str(decision.get("label", ""))
        confidence = float(decision.get("confidence", 0.0) or 0.0)
        proposed = candidates.get(label)
        if proposed is None:
            chosen = anchor
            validation = "no_valid_vlm_candidate_use_persistent_identity"
            source = "persistent_identity_fallback"
            if len(candidates) > 1:
                fallback_frames.append(frame)
        else:
            distance = float(proposed.get("track_distance_px", 0.0) or 0.0)
            if distance <= maximum_track_distance:
                chosen = proposed
                validation = "accepted_by_vlm_and_persistent_identity"
                source = "vlm_identity_gate"
                vlm_accepted += 1
            else:
                chosen = anchor
                validation = "vlm_copy_rejected_by_persistent_identity_continuity"
                source = "vlm_proposal_hard_rejected"
                vlm_rejected += 1
                fallback_frames.append(frame)
        radius = float(chosen.get("radius_px", 0.0) or 0.0)
        if radius <= 1.0:
            radius = fallback_radius
        selected.append(
            {
                "frame": frame,
                "time": chosen.get("time", ""),
                "entity_id": "target_sphere",
                "selected_candidate_id": chosen["candidate_id"],
                "u": chosen["u"],
                "v": chosen["v"],
                "radius_px": f"{radius:.3f}",
                "selection_source": source,
                "vlm_label": label,
                "vlm_confidence": f"{confidence:.3f}",
                "hard_track_distance_px": chosen.get("track_distance_px", "0.000"),
                "hard_validation": validation,
                "candidate_count": len(candidates),
                "unique_entity_count": 1,
            }
        )
    output = result_dir / "sphere_identity_selection.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(selected)
    manifest = {
        "schema_version": 1,
        "entity_count": 1,
        "frame_count": len(selected),
        "vlm_decision_count": len(decisions),
        "vlm_accepted_count": vlm_accepted,
        "vlm_hard_rejected_count": vlm_rejected,
        "hard_rejected_or_fallback_frames": sorted(set(fallback_frames)),
        "maximum_track_distance_px": maximum_track_distance,
        "policy": "VLM selects one discrete identity candidate; persistent-track continuity is a hard validator",
        "output": str(output),
    }
    (result_dir / "sphere_identity_selection_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--maximum-track-distance", type=float, default=24.0)
    args = parser.parse_args()
    print(json.dumps(run(args.result_dir, args.maximum_track_distance), indent=2))


if __name__ == "__main__":
    main()
