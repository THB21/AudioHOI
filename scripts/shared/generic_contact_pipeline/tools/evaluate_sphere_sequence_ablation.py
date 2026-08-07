#!/usr/bin/env python3
"""Compare sphere-sequence ablations against one shared visible-mask contract."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--variant", action="append", required=True, help="NAME=RESULT_DIR")
    parser.add_argument("--fx", type=float, required=True)
    parser.add_argument("--fy", type=float, required=True)
    parser.add_argument("--cx", type=float, required=True)
    parser.add_argument("--cy", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    visible = {
        int(row["frame"]): (float(row["ball_center_x"]), float(row["ball_center_y"]))
        for row in rows(args.sample_dir / "results/tracking/object_trajectory.csv")
        if row.get("ball_center_x") and row.get("ball_center_y")
    }
    audio_frames = sorted({
        int(float(row.get("peak_frame") or row.get("frame") or row.get("start_frame") or 0))
        for row in rows(args.sample_dir / "results/events/audio_events.csv")
        if row.get("event_type") in {"impact", "hand_impact", "seam_click"}
    })
    output_rows: list[dict[str, object]] = []
    provenance: dict[str, object] = {}
    for value in args.variant:
        name, raw_dir = value.split("=", 1)
        result_dir = Path(raw_dir)
        pose_path = result_dir / "ablation_pose.csv"
        pose_rows = rows(pose_path)
        xyz = np.asarray([[float(row[key]) for key in ("tx", "ty", "tz")] for row in pose_rows])
        errors: list[float] = []
        identity_interval_errors: list[float] = []
        for row, point in zip(pose_rows, xyz):
            frame = int(row["frame"])
            if frame not in visible or point[2] <= 1e-6:
                continue
            u = args.fx * point[0] / point[2] + args.cx
            v = args.fy * point[1] / point[2] + args.cy
            error = float(np.hypot(u - visible[frame][0], v - visible[frame][1]))
            errors.append(error)
            if 145 <= frame <= 180:
                identity_interval_errors.append(error)
        acceleration = np.diff(xyz, n=2, axis=0)
        jerk = np.diff(xyz, n=3, axis=0)
        attempt_dirs = sorted(
            (result_dir / "generic_sequence_solver_attempts").glob("*"),
            key=lambda path: path.stat().st_mtime,
        )
        hard = json.loads((attempt_dirs[-1] / "hard_metrics.json").read_text())["metrics"]
        publication = json.loads((result_dir / "generic_object_publication.json").read_text())
        vlm_results = result_dir / "vlm/stage4/vlm_results.csv"
        contacts = [
            int(float(row["frame"]))
            for row in rows(vlm_results) if vlm_results.exists()
            if row.get("query_type") == "contact_relation_check"
            and row.get("pass_gate") == "pass"
            and row.get("normalized_label") not in {"no_contact", "unclear", ""}
        ]
        timing = [min(abs(frame - event) for event in audio_frames) for frame in contacts] if contacts and audio_frames else []
        output_rows.append({
            "variant": name,
            "publication_status": publication.get("status"),
            "visible_mask_error_mean_px": float(np.mean(errors)),
            "visible_mask_error_p95_px": percentile(errors, 95),
            "identity_interval_145_180_error_p95_px": percentile(identity_interval_errors, 95),
            "contact_gap_p95_mm": float(hard["contact_gap_p95_m"]) * 1000.0,
            "contact_gap_max_mm": float(hard["contact_gap_max_m"]) * 1000.0,
            "trajectory_acceleration_p95_m_per_frame2": percentile(np.linalg.norm(acceleration, axis=1).tolist(), 95),
            "trajectory_jerk_p95_m_per_frame3": percentile(np.linalg.norm(jerk, axis=1).tolist(), 95),
            "vlm_contact_event_count": len(contacts),
            "contact_audio_nearest_error_mean_frames": float(np.mean(timing)) if timing else None,
        })
        provenance[name] = {
            "pose": str(pose_path.resolve()),
            "pose_sha256": sha256(pose_path),
            "hard_metrics": str((attempt_dirs[-1] / "hard_metrics.json").resolve()),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "ablation_metrics.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    json_path = args.output_dir / "ablation_metrics.json"
    json_path.write_text(json.dumps({
        "schema_version": 1,
        "metric_contract": "shared_visible_sam2_mask_and_generic_stage4_hard_metrics",
        "rows": output_rows,
        "provenance": provenance,
    }, indent=2) + "\n")
    print(json.dumps({"csv": str(csv_path), "json": str(json_path), "rows": output_rows}, indent=2))


if __name__ == "__main__":
    main()
