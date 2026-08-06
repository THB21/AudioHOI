#!/usr/bin/env python3
"""Build generic one-entity sphere candidates from mask components and a persistent track.

This tool does not decide which visual copy is physically real. It only exposes typed
candidates for a later forced-choice VLM gate. There remains exactly one optimized
entity regardless of the number of candidates in a frame.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


FIELDS = [
    "frame", "time", "candidate_id", "u", "v", "radius_px", "area_px",
    "source", "track_distance_px", "candidate_count", "ambiguous",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def mask_components(path: Path, minimum_area: int) -> list[tuple[float, float, float, int]]:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8))
    out: list[tuple[float, float, float, int]] = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < minimum_area:
            continue
        u, v = map(float, centroids[index])
        radius = float(np.sqrt(area / np.pi))
        out.append((u, v, radius, area))
    return out


def build(sample_dir: Path, result_dir: Path, minimum_area: int, merge_radius: float) -> dict[str, object]:
    tracking = sample_dir / "results/tracking"
    track_rows = read_csv(tracking / "object_center_trajectory.csv")
    track_by_frame = {
        int(float(row["frame"])): (float(row["center_x"]), float(row["center_y"]), row.get("time", ""))
        for row in track_rows
    }
    rows: list[dict[str, object]] = []
    ambiguous_frames: list[int] = []
    source_counts: dict[str, int] = {}
    for frame in sorted(track_by_frame):
        track_u, track_v, time = track_by_frame[frame]
        raw = mask_components(
            sample_dir / f"results/segmentation/masks/{frame:05d}_mask.png",
            minimum_area,
        )
        candidates: list[dict[str, object]] = [
            {
                "u": track_u,
                "v": track_v,
                "radius_px": 0.0,
                "area_px": 0,
                "source": "persistent_track_hypothesis",
                "track_distance_px": 0.0,
            }
        ]
        for u, v, radius, area in sorted(raw, key=lambda item: np.hypot(item[0] - track_u, item[1] - track_v)):
            distance = float(np.hypot(u - track_u, v - track_v))
            if distance <= merge_radius:
                candidates[0].update(radius_px=radius, area_px=area, source="persistent_track_plus_mask")
                continue
            candidates.append(
                {
                    "u": u,
                    "v": v,
                    "radius_px": radius,
                    "area_px": area,
                    "source": "disconnected_mask_component",
                    "track_distance_px": distance,
                }
            )
        candidate_count = len(candidates)
        ambiguous = candidate_count > 1
        if ambiguous:
            ambiguous_frames.append(frame)
        for index, candidate in enumerate(candidates):
            candidate_id = f"candidate_{chr(ord('a') + index)}"
            source = str(candidate["source"])
            source_counts[source] = source_counts.get(source, 0) + 1
            rows.append(
                {
                    "frame": frame,
                    "time": time,
                    "candidate_id": candidate_id,
                    "u": f"{float(candidate['u']):.3f}",
                    "v": f"{float(candidate['v']):.3f}",
                    "radius_px": f"{float(candidate['radius_px']):.3f}",
                    "area_px": int(candidate["area_px"]),
                    "source": source,
                    "track_distance_px": f"{float(candidate['track_distance_px']):.3f}",
                    "candidate_count": candidate_count,
                    "ambiguous": "1" if ambiguous else "0",
                }
            )
    output = result_dir / "sphere_identity_candidates.csv"
    write_csv(output, rows)
    manifest = {
        "schema_version": 1,
        "entity_count": 1,
        "entity_id": "target_sphere",
        "candidate_rows": len(rows),
        "frame_count": len(track_by_frame),
        "ambiguous_frames": ambiguous_frames,
        "ambiguous_frame_count": len(ambiguous_frames),
        "source_counts": source_counts,
        "selection_performed": False,
        "baseline_pose_read": False,
        "human_state_optimized": False,
        "policy": "multiple visual hypotheses for exactly one physical sphere entity",
        "output": str(output),
    }
    manifest_path = result_dir / "sphere_identity_candidates_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--minimum-area", type=int, default=20)
    parser.add_argument("--merge-radius", type=float, default=12.0)
    args = parser.parse_args()
    print(json.dumps(build(args.sample_dir, args.result_dir, args.minimum_area, args.merge_radius), indent=2))


if __name__ == "__main__":
    main()
