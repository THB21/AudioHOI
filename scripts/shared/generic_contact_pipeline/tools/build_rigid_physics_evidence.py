#!/usr/bin/env python3
"""Build solver-independent evidence for a known rigid object.

The builder deliberately has no pose input.  It adapts observation providers,
audits feature tracks, preserves all pose hypotheses, and emits a hard pre-solve
manifest instead of manufacturing a trajectory from incomplete evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from math import isfinite, log
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.shared.generic_contact_pipeline.core.measurements import (
    RelativeDepthEvidence,
    RigidFeatureTrackEvidence,
    RigidPhysicsEvidenceManifest,
    RigidPoseHypothesisEvidence,
    RigidSilhouetteEvidence,
)


_BOUNDARY_LIMIT_PX = {
    "body_corner": 12.0,
    "support_point": 12.0,
    "wheel_center": 32.0,
    "line_endpoint": 10.0,
    "grasp_point": 18.0,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_columns(table: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"{label} lacks required columns: {missing}")


def _parse_intervals(raw: str) -> tuple[tuple[int, int], ...]:
    intervals = []
    for item in raw.split(","):
        fields = item.strip().split("-", maxsplit=1)
        if len(fields) != 2:
            raise ValueError(f"invalid trusted interval {item!r}")
        start, end = (int(value) for value in fields)
        if start < 1 or end < start:
            raise ValueError(f"invalid trusted interval {item!r}")
        intervals.append((start, end))
    if not intervals:
        raise ValueError("at least one trusted anchor interval is required")
    return tuple(intervals)


def _inside(frame: int, intervals: tuple[tuple[int, int], ...]) -> bool:
    return any(start <= frame <= end for start, end in intervals)


def _complement_intervals(
    first_frame: int,
    last_frame: int,
    trusted: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...]:
    trusted_frames = {frame for frame in range(first_frame, last_frame + 1) if _inside(frame, trusted)}
    missing = [frame for frame in range(first_frame, last_frame + 1) if frame not in trusted_frames]
    intervals: list[list[int]] = []
    for frame in missing:
        if not intervals or frame != intervals[-1][-1] + 1:
            intervals.append([frame])
        else:
            intervals[-1].append(frame)
    return tuple((values[0], values[-1]) for values in intervals)


def _maximum_consecutive_gap(frames: list[int]) -> int:
    maximum = current = 0
    previous = None
    for frame in sorted(frames):
        current = current + 1 if previous is not None and frame == previous + 1 else 1
        maximum = max(maximum, current)
        previous = frame
    return maximum


def _finite_or_large(value: object) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 1e9
    return parsed if isfinite(parsed) else 1e9


def _write_csv(records: list[object], path: Path) -> None:
    rows = [asdict(record) for record in records]
    for row in rows:
        if "candidate_feature_ids" in row:
            row["candidate_feature_ids"] = "|".join(row["candidate_feature_ids"])
    pd.DataFrame(rows).to_csv(path, index=False)


def _silhouette_and_depth(
    sample_id: str,
    path: Path,
) -> tuple[list[RigidSilhouetteEvidence], list[RelativeDepthEvidence]]:
    table = pd.read_csv(path).sort_values("frame")
    _require_columns(
        table,
        {
            "frame",
            "time",
            "center_x",
            "center_y",
            "body_bbox_x1",
            "body_bbox_y1",
            "body_bbox_x2",
            "body_bbox_y2",
            "mask_area_px",
            "observation_conf",
            "visibility",
            "object_ref_depth_m",
            "depth_conf",
        },
        "object observations",
    )
    silhouettes: list[RigidSilhouetteEvidence] = []
    depths: list[RelativeDepthEvidence] = []
    source = str(path)
    for row in table.itertuples(index=False):
        bbox = (
            float(row.body_bbox_x1),
            float(row.body_bbox_y1),
            float(row.body_bbox_x2),
            float(row.body_bbox_y2),
        )
        width = bbox[2] - bbox[0] + 1.0
        height = bbox[3] - bbox[1] + 1.0
        area = float(row.mask_area_px)
        visibility = str(row.visibility)
        silhouettes.append(
            RigidSilhouetteEvidence(
                sample_id=sample_id,
                frame=int(row.frame),
                time=float(row.time),
                visibility=visibility,
                centroid_uv=(float(row.center_x), float(row.center_y)),
                body_bbox_xyxy=bbox,
                mask_area_px=area,
                log_body_width_px=log(width),
                log_body_height_px=log(height),
                log_mask_area_px=log(area),
                log_aspect_ratio=log(width / height),
                scale_reliable=(visibility == "visible" and float(row.observation_conf) > 0.5),
                source_artifact=source,
            )
        )
        depth = float(row.object_ref_depth_m)
        confidence = float(row.depth_conf)
        if isfinite(depth) and depth > 0.0 and isfinite(confidence) and confidence > 0.0:
            depths.append(
                RelativeDepthEvidence(
                    sample_id=sample_id,
                    frame=int(row.frame),
                    time=float(row.time),
                    depth_m=depth,
                    confidence=min(confidence, 1.0),
                    log_depth=log(depth),
                    source_artifact=source,
                )
            )
    return silhouettes, depths


def _trend_rows(
    silhouettes: list[RigidSilhouetteEvidence],
    depths: list[RelativeDepthEvidence],
) -> list[dict[str, object]]:
    silhouette_by_frame = {row.frame: row for row in silhouettes if row.scale_reliable}
    depth_by_frame = {row.frame: row for row in depths}
    rows = []
    for frame in sorted(set(silhouette_by_frame) & set(depth_by_frame)):
        previous = frame - 1
        if previous not in silhouette_by_frame or previous not in depth_by_frame:
            continue
        current_s, previous_s = silhouette_by_frame[frame], silhouette_by_frame[previous]
        current_d, previous_d = depth_by_frame[frame], depth_by_frame[previous]
        delta_depth = current_d.log_depth - previous_d.log_depth
        delta_height = current_s.log_body_height_px - previous_s.log_body_height_px
        delta_area = current_s.log_mask_area_px - previous_s.log_mask_area_px
        rows.append(
            {
                "frame": frame,
                "previous_frame": previous,
                "delta_log_depth": delta_depth,
                "delta_log_body_height": delta_height,
                "delta_log_mask_area": delta_area,
                "depth_height_opposite_sign": int(delta_depth * delta_height <= 0.0),
                "depth_area_opposite_sign": int(delta_depth * delta_area <= 0.0),
            }
        )
    return rows


def _feature_evidence(
    sample_id: str,
    path: Path,
    trusted_intervals: tuple[tuple[int, int], ...],
) -> tuple[list[RigidFeatureTrackEvidence], int, int]:
    table = pd.read_csv(path)
    _require_columns(
        table,
        {
            "frame",
            "query_id",
            "feature_id",
            "feature_kind",
            "anchor_frame",
            "x",
            "y",
            "cotracker_visibility",
            "mask_compatible",
            "mask_edge_distance_px",
            "cross_bank_error_px",
            "usable",
        },
        "feature tracks",
    )
    records: list[RigidFeatureTrackEvidence] = []
    contaminated_input = 0
    untrusted_anchor_input = 0
    for row in table.itertuples(index=False):
        kind = str(row.feature_kind)
        boundary_limit = _BOUNDARY_LIMIT_PX.get(kind)
        if boundary_limit is None:
            raise ValueError(f"unsupported rigid feature role {kind!r}")
        anchor_frame = int(row.anchor_frame)
        anchor_trusted = _inside(anchor_frame, trusted_intervals)
        boundary_distance = _finite_or_large(row.mask_edge_distance_px)
        cross_bank_error = _finite_or_large(row.cross_bank_error_px)
        role_compatible = boundary_distance <= boundary_limit
        reasons = []
        if not anchor_trusted:
            reasons.append("untrusted_anchor")
        if float(row.cotracker_visibility) < 0.65:
            reasons.append("low_tracker_visibility")
        if cross_bank_error > 18.0:
            reasons.append("cross_bank_disagreement")
        if int(row.mask_compatible) != 1:
            reasons.append("outside_object_mask")
        if not role_compatible:
            reasons.append("role_boundary_mismatch")
        usable = not reasons
        if int(row.usable) == 1 and not role_compatible:
            contaminated_input += 1
        if int(row.usable) == 1 and not anchor_trusted:
            untrusted_anchor_input += 1
        feature_id = str(row.feature_id)
        records.append(
            RigidFeatureTrackEvidence(
                sample_id=sample_id,
                frame=int(row.frame),
                query_id=str(row.query_id),
                feature_kind=kind,
                candidate_feature_ids=(feature_id,),
                u=float(row.x),
                v=float(row.y),
                tracker_visibility=float(np.clip(float(row.cotracker_visibility), 0.0, 1.0)),
                boundary_distance_px=boundary_distance,
                cross_bank_error_px=cross_bank_error,
                anchor_frame=anchor_frame,
                anchor_trusted=anchor_trusted,
                role_compatible=role_compatible,
                usable=usable,
                rejection_reason=None if usable else "|".join(reasons),
            )
        )
    return records, contaminated_input, untrusted_anchor_input


def _pose_hypotheses(
    sample_id: str,
    path: Path,
) -> tuple[list[RigidPoseHypothesisEvidence], set[int]]:
    raw_rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    records = []
    for row in raw_rows:
        records.append(
            RigidPoseHypothesisEvidence(
                sample_id=sample_id,
                frame=int(row["frame"]),
                rank=int(row["hypothesis_rank"]),
                score=float(row["score"]),
                mask_iou=float(row["official_render_mask_iou"]),
                translation_m=(float(row["tx_m"]), float(row["ty_m"]), float(row["tz_m"])),
                quaternion_xyzw=(float(row["qx"]), float(row["qy"]), float(row["qz"]), float(row["qw"])),
                selected_by_provider=bool(row.get("selected_by_visual_geometry", False)),
                provider_status=str(row.get("provider_status", "unknown")),
                source_artifact=str(path),
            )
        )
    ambiguous = set()
    by_frame: dict[int, list[RigidPoseHypothesisEvidence]] = {}
    for record in records:
        by_frame.setdefault(record.frame, []).append(record)
    for frame, frame_rows in by_frame.items():
        ordered = sorted(frame_rows, key=lambda value: value.mask_iou, reverse=True)
        close_iou = len(ordered) > 1 and ordered[0].mask_iou - ordered[1].mask_iou <= 0.08
        blocked = any(value.provider_status == "blocked_visual_evidence" for value in ordered)
        if close_iou or blocked:
            ambiguous.add(frame)
    return records, ambiguous


def _trusted_line_frames(path: Path) -> tuple[pd.DataFrame, set[int]]:
    """Audit provider-native rigid line evidence without assigning a pose.

    A paired rail observation or an unassigned rail axis both constrain rigid
    orientation.  Physical left/right identity is intentionally left to the
    sequence factor compiler; this adapter only establishes that a visible,
    trusted image line exists.
    """

    table = pd.read_csv(path)
    _require_columns(
        table,
        {
            "frame",
            "physical_x1",
            "physical_y1",
            "physical_x2",
            "physical_y2",
            "endpoint_track_conf",
            "line_observation_trusted",
            "visibility",
        },
        "line observations",
    )
    finite = np.isfinite(
        table[["physical_x1", "physical_y1", "physical_x2", "physical_y2"]].to_numpy(float)
    ).all(axis=1)
    length = np.hypot(
        table.physical_x2.to_numpy(float) - table.physical_x1.to_numpy(float),
        table.physical_y2.to_numpy(float) - table.physical_y1.to_numpy(float),
    )
    trusted = (
        finite
        & (length >= 8.0)
        & (table.endpoint_track_conf.to_numpy(float) >= 0.45)
        & (table.line_observation_trusted.to_numpy(int) == 1)
        & table.visibility.astype(str).str.startswith("visible").to_numpy()
    )
    audited = table.copy()
    audited["line_length_px"] = length
    audited["usable_orientation_evidence"] = trusted.astype(int)
    audited["rejection_reason"] = np.where(
        trusted,
        "",
        np.where(~finite, "nonfinite_endpoint", np.where(length < 8.0, "line_too_short", "untrusted_or_low_confidence")),
    )
    return audited, set(audited.loc[trusted, "frame"].astype(int))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--object-observations", type=Path, required=True)
    parser.add_argument("--feature-tracks", type=Path, required=True)
    parser.add_argument("--line-observations", type=Path)
    parser.add_argument("--megapose-hypotheses", type=Path, required=True)
    parser.add_argument("--trusted-anchor-intervals", required=True)
    parser.add_argument("--max-visible-orientation-gap", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    observations = args.object_observations.resolve()
    feature_tracks = args.feature_tracks.resolve()
    megapose = args.megapose_hypotheses.resolve()
    line_observations = None if args.line_observations is None else args.line_observations.resolve()
    for path in (observations, feature_tracks, megapose, line_observations):
        if path is None:
            continue
        if not path.is_file():
            raise FileNotFoundError(path)
    trusted_intervals = _parse_intervals(args.trusted_anchor_intervals)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    silhouettes, depths = _silhouette_and_depth(args.sample_id, observations)
    solve_intervals = _complement_intervals(
        min(row.frame for row in silhouettes),
        max(row.frame for row in silhouettes),
        trusted_intervals,
    )
    if not solve_intervals:
        raise ValueError("trusted intervals cover the full sequence; no solve interval remains")
    feature_rows, contaminated_input, untrusted_anchor_input = _feature_evidence(
        args.sample_id, feature_tracks, trusted_intervals
    )
    pose_rows, ambiguous_frames = _pose_hypotheses(args.sample_id, megapose)
    line_rows, provider_line_frames = (
        (pd.DataFrame(), set())
        if line_observations is None
        else _trusted_line_frames(line_observations)
    )
    trends = _trend_rows(silhouettes, depths)

    _write_csv(silhouettes, output / "rigid_silhouette_evidence.csv")
    _write_csv(depths, output / "relative_depth_evidence.csv")
    _write_csv(feature_rows, output / "rigid_feature_track_evidence.csv")
    pd.DataFrame(trends).to_csv(output / "scale_depth_trend.csv", index=False)
    if line_observations is not None:
        line_rows.to_csv(output / "rigid_line_evidence.csv", index=False)
    with (output / "rigid_pose_hypotheses_evidence.jsonl").open("w") as handle:
        for record in pose_rows:
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")

    in_solve_interval = lambda frame: _inside(frame, solve_intervals)
    usable_rows = [row for row in feature_rows if row.usable and in_solve_interval(row.frame)]
    usable_feature_frames = {row.frame for row in usable_rows}
    trusted_rail_frames = {
        row.frame for row in usable_rows if row.feature_kind == "line_endpoint"
    } | {frame for frame in provider_line_frames if in_solve_interval(frame)}
    body_queries_by_frame: dict[int, set[str]] = {}
    for row in usable_rows:
        if row.feature_kind == "body_corner":
            body_queries_by_frame.setdefault(row.frame, set()).add(row.query_id)
    orientation_frames = trusted_rail_frames | {
        frame for frame, query_ids in body_queries_by_frame.items() if len(query_ids) >= 2
    }
    required_visible_frames = [
        row.frame
        for row in silhouettes
        if in_solve_interval(row.frame) and row.visibility == "visible" and row.frame not in orientation_frames
    ]
    maximum_orientation_gap = _maximum_consecutive_gap(required_visible_frames)
    contaminated_usable = sum(
        row.usable and not row.role_compatible and in_solve_interval(row.frame) for row in feature_rows
    )
    untrusted_anchor_usable = sum(
        row.usable and not row.anchor_trusted and in_solve_interval(row.frame) for row in feature_rows
    )
    clear_scale_count = sum(row.scale_reliable and in_solve_interval(row.frame) for row in silhouettes)
    solve_depths = [row for row in depths if in_solve_interval(row.frame)]
    gates = {
        "clear_scale_coverage": clear_scale_count >= 2,
        "relative_depth_coverage": len(solve_depths) >= 2,
        "no_contaminated_usable_tracks": contaminated_usable == 0,
        "no_untrusted_anchor_tracks": untrusted_anchor_usable == 0,
        "trusted_rigid_feature_coverage": len(usable_feature_frames) >= 2,
        "trusted_rail_coverage": len(trusted_rail_frames) >= 2,
        "bounded_visible_orientation_gap": maximum_orientation_gap <= args.max_visible_orientation_gap,
    }
    manifest = RigidPhysicsEvidenceManifest(
        schema_version=1,
        sample_id=args.sample_id,
        frame_count=len(silhouettes),
        clear_scale_frame_count=clear_scale_count,
        relative_depth_frame_count=len(solve_depths),
        usable_feature_frame_count=len(usable_feature_frames),
        trusted_rail_frame_count=len(trusted_rail_frames),
        contaminated_track_count=contaminated_input,
        untrusted_anchor_track_count=untrusted_anchor_input,
        megapose_ambiguous_frame_count=len(ambiguous_frames),
        source_hashes={
            "object_observations": _sha256(observations),
            "feature_tracks": _sha256(feature_tracks),
            "megapose_hypotheses": _sha256(megapose),
            **({"line_observations": _sha256(line_observations)} if line_observations is not None else {}),
        },
        gates=gates,
        ready_for_solver=all(gates.values()),
    )
    payload = asdict(manifest)
    payload.update(
        {
            "trusted_anchor_intervals": [list(value) for value in trusted_intervals],
            "solve_intervals": [list(value) for value in solve_intervals],
            "trusted_orientation_frames": sorted(orientation_frames),
            "maximum_visible_orientation_gap_frames": maximum_orientation_gap,
            "maximum_allowed_visible_orientation_gap_frames": args.max_visible_orientation_gap,
            "megapose_ambiguous_frames": sorted(ambiguous_frames),
            "contaminated_input_track_count": contaminated_input,
            "untrusted_anchor_input_track_count": untrusted_anchor_input,
            "publication_status": "solver_blocked" if not manifest.ready_for_solver else "evidence_ready",
            "accepted_pose_read": False,
            "accepted_pose_written": False,
            "case_dispatch_used": False,
            "human_state_optimized": False,
        }
    )
    (output / "rigid_physics_evidence_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
