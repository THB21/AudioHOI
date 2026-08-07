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

    trajectory_rows = rows(args.sample_dir / "results/tracking/object_trajectory.csv")
    visible = {
        int(row["frame"]): (float(row["ball_center_x"]), float(row["ball_center_y"]))
        for row in trajectory_rows
        if row.get("ball_center_x") and row.get("ball_center_y")
    }
    missing_frames = {
        int(row["frame"])
        for row in trajectory_rows
        if not row.get("ball_center_x") or not row.get("ball_center_y")
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
        publication = json.loads((result_dir / "generic_object_publication.json").read_text())
        pose_path = (
            result_dir / "object_pose.csv"
            if publication.get("status") == "accepted"
            else result_dir / "ablation_pose.csv"
        )
        pose_rows = rows(pose_path)
        xyz = np.asarray([[float(row[key]) for key in ("tx", "ty", "tz")] for row in pose_rows])
        pose_index_by_frame = {int(row["frame"]): index for index, row in enumerate(pose_rows)}
        errors: list[float] = []
        identity_interval_errors: list[float] = []
        missing_projected_outside = 0
        missing_boundary_sticking = 0
        image_width = 2.0 * args.cx
        image_height = 2.0 * args.cy
        for row, point in zip(pose_rows, xyz):
            frame = int(row["frame"])
            if point[2] <= 1e-6:
                continue
            u = args.fx * point[0] / point[2] + args.cx
            v = args.fy * point[1] / point[2] + args.cy
            if frame in missing_frames:
                outside = u < 0.0 or u >= image_width or v < 0.0 or v >= image_height
                missing_projected_outside += int(outside)
                if not outside and min(u, image_width - u, v, image_height - v) <= 12.0:
                    missing_boundary_sticking += 1
            if frame not in visible:
                continue
            error = float(np.hypot(u - visible[frame][0], v - visible[frame][1]))
            errors.append(error)
            if 145 <= frame <= 180:
                identity_interval_errors.append(error)
        acceleration = np.diff(xyz, n=2, axis=0)
        jerk = np.diff(xyz, n=3, axis=0)
        projected_v = args.fy * xyz[:, 1] / xyz[:, 2] + args.cy
        projected_radius = args.fx * float(pose_rows[0].get("radius_m") or 0.0) / xyz[:, 2]
        support_path = result_dir / "support_geometry.json"
        floor_v = None
        if support_path.exists():
            floor_v = float(json.loads(support_path.read_text()).get("floor_v", "nan"))
            if not np.isfinite(floor_v):
                floor_v = None
        contact_events_path = result_dir / "contact_events.csv"
        floor_contact_frames = []
        if contact_events_path.exists():
            floor_contact_frames = sorted({
                int(float(row["frame"]))
                for row in rows(contact_events_path)
                if row.get("target") in {"floor", "unknown_plane"}
                or row.get("contact_type") in {"plane_support_contact_event", "floor_contact_event"}
            })
        offscreen_floor_contacts = [frame for frame in floor_contact_frames if frame in missing_frames]
        floor_bottom_errors = [
            abs(float(
                projected_v[pose_index_by_frame[frame]]
                + projected_radius[pose_index_by_frame[frame]]
                - floor_v
            ))
            for frame in offscreen_floor_contacts
            if floor_v is not None and frame in pose_index_by_frame
        ]
        vertical_velocity = np.diff(projected_v)
        reversal_frames = {
            int(pose_rows[index]["frame"])
            for index in range(1, len(vertical_velocity))
            if vertical_velocity[index - 1] > 0.0 and vertical_velocity[index] < 0.0
            and int(pose_rows[index]["frame"]) in missing_frames
        }
        unsupported_reversals = [
            frame for frame in reversal_frames
            if not any(abs(frame - event) <= 1 for event in floor_contact_frames)
        ]
        attempt_dirs = sorted(
            (result_dir / "generic_sequence_solver_attempts").glob("*"),
            key=lambda path: path.stat().st_mtime,
        )
        hard = json.loads((attempt_dirs[-1] / "hard_metrics.json").read_text())["metrics"]
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
            "missing_mask_frame_count": len(missing_frames),
            "missing_mask_projected_center_outside_count": missing_projected_outside,
            "missing_mask_boundary_sticking_count": missing_boundary_sticking,
            "contact_gap_p95_mm": float(hard["contact_gap_p95_m"]) * 1000.0,
            "contact_gap_max_mm": float(hard["contact_gap_max_m"]) * 1000.0,
            "trajectory_acceleration_p95_m_per_frame2": percentile(np.linalg.norm(acceleration, axis=1).tolist(), 95),
            "trajectory_jerk_p95_m_per_frame3": percentile(np.linalg.norm(jerk, axis=1).tolist(), 95),
            "vlm_contact_event_count": len(contacts),
            "contact_audio_nearest_error_mean_frames": float(np.mean(timing)) if timing else None,
            "offscreen_floor_contact_frames": ";".join(map(str, offscreen_floor_contacts)),
            "offscreen_floor_bottom_error_px": percentile(floor_bottom_errors, 95),
            "offscreen_unexplained_down_to_up_reversal_count": len(unsupported_reversals),
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
