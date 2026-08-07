from __future__ import annotations

import json
import numpy as np

from ....core.base.config import CaseProfile
from ....core.base.io import read_csv, write_csv, write_json
from ....core.base.schema import stage_paths


def _first_existing(*paths):
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def _by_frame(rows: list[dict[str, str]]) -> dict[int, dict[str, str]]:
    out: dict[int, dict[str, str]] = {}
    for row in rows:
        try:
            out[int(float(row["frame"]))] = row
        except Exception:
            continue
    return out


def _pick(*values: object, default: float = 0.0) -> float:
    for value in values:
        if value in {"", None}:
            continue
        try:
            return float(value)  # type: ignore[arg-type]
        except Exception:
            continue
    return default


def _accepted_offscreen_intervals(profile: CaseProfile) -> list[tuple[int, int]]:
    """Read only VLM-confirmed out-of-frame visibility windows."""

    stage_dir = profile.result_dir / "vlm" / "stage1"
    queries_path = stage_dir / "vlm_queries.csv"
    results_path = stage_dir / "vlm_results.csv"
    if not queries_path.exists() or not results_path.exists():
        return []
    accepted = {
        int(float(row.get("frame", "-1") or -1))
        for row in read_csv(results_path)
        if row.get("query_type") == "keypart_visibility_check"
        and row.get("normalized_label") == "out_of_frame"
        and row.get("pass_gate") == "pass"
    }
    intervals: list[tuple[int, int]] = []
    for row in read_csv(queries_path):
        try:
            representative = int(float(row.get("frame", "-1") or -1))
            start = int(float(row.get("start_frame", representative) or representative))
            end = int(float(row.get("end_frame", representative) or representative))
        except Exception:
            continue
        if row.get("query_type") == "keypart_visibility_check" and representative in accepted:
            intervals.append((start, end))
    return intervals


def _projected_ballistic_gap_centers(
    visible_centers: dict[int, tuple[float, float]],
    intervals: list[tuple[int, int]],
    *,
    support_frames: int,
) -> dict[int, tuple[float, float]]:
    """Fit constant-acceleration image trajectories without image clipping."""

    out: dict[int, tuple[float, float]] = {}
    for start, end in intervals:
        before = sorted((frame for frame in visible_centers if frame < start), reverse=True)[:support_frames]
        after = sorted(frame for frame in visible_centers if frame > end)[:support_frames]
        support = sorted(before + after)
        if len(before) < 2 or len(after) < 2 or len(support) < 5:
            continue
        degree = 2
        frame_values = np.asarray(support, dtype=np.float64)
        u_values = np.asarray([visible_centers[frame][0] for frame in support], dtype=np.float64)
        v_values = np.asarray([visible_centers[frame][1] for frame in support], dtype=np.float64)
        u_coeff = np.polyfit(frame_values, u_values, degree)
        v_coeff = np.polyfit(frame_values, v_values, degree)
        for frame in range(start, end + 1):
            out[frame] = (float(np.polyval(u_coeff, frame)), float(np.polyval(v_coeff, frame)))
    return out


def refine_offscreen_floor_impacts(profile: CaseProfile) -> dict[str, object]:
    """Anchor VLM-confirmed off-screen reversals to audio-confirmed floor impacts."""

    paths = stage_paths(profile)
    config = dict(dict(profile.data.get("vlm", {})).get("single_identity", {}))
    if not config.get("offscreen_floor_impact_enabled", False):
        return {"status": "not_enabled", "changed_frames": []}
    if "disable_audio_events" in set(profile.data.get("ablation_flags", ())):
        return {"status": "audio_disabled", "changed_frames": []}
    observation_path = paths["object_observations"]
    support_path = paths["support_geometry"]
    contact_path = paths["contact_events"]
    audio_path = profile.sample_dir / "results" / "events" / "audio_events.csv"
    if not all(path.exists() for path in (observation_path, support_path, contact_path, audio_path)):
        return {"status": "missing_inputs", "changed_frames": []}

    intervals = _accepted_offscreen_intervals(profile)
    observations = read_csv(observation_path)
    by_frame = _by_frame(observations)
    support = json.loads(support_path.read_text())
    floor_v = float(support.get("floor_v", float("nan")))
    if not np.isfinite(floor_v):
        return {"status": "invalid_support_plane", "changed_frames": []}
    floor_events = [
        int(float(row.get("frame", "-1") or -1))
        for row in read_csv(contact_path)
        if row.get("contact_type") in {"plane_support_contact_event", "floor_contact_event"}
        or row.get("target") in {"floor", "unknown_plane"}
    ]
    audio_events: list[tuple[int, float]] = []
    for row in read_csv(audio_path):
        if row.get("event_type") not in {"impact", "hand_impact", "seam_click"}:
            continue
        frame = int(float(row.get("audio_frame") or row.get("peak_frame") or row.get("frame") or -1))
        score = _pick(row.get("audio_score"), row.get("snr"), default=0.0)
        audio_events.append((frame, score))

    changed: list[int] = []
    impacts: list[dict[str, object]] = []
    for start, end in intervals:
        if not any(start - 3 <= frame <= end + 3 for frame in floor_events):
            continue
        candidates = [(frame, score) for frame, score in audio_events if start - 2 <= frame <= end + 2]
        if not candidates:
            continue
        impact_frame, audio_score = max(candidates, key=lambda value: value[1])
        previous_frame = start - 1
        following_frame = end + 1
        if previous_frame not in by_frame or following_frame not in by_frame:
            continue
        previous_v = _pick(by_frame[previous_frame].get("ref_v"))
        following_v = _pick(by_frame[following_frame].get("ref_v"))
        previous_samples = [
            _pick(by_frame[frame].get("ref_v"))
            for frame in range(max(1, previous_frame - 2), previous_frame + 1)
            if frame in by_frame
        ]
        following_samples = [
            _pick(by_frame[frame].get("ref_v"))
            for frame in range(following_frame, following_frame + 3)
            if frame in by_frame
        ]
        if len(previous_samples) < 2 or len(following_samples) < 2:
            continue
        incoming_velocity = float(np.median(np.diff(np.asarray(previous_samples, dtype=float))))
        outgoing_velocity = float(np.median(np.diff(np.asarray(following_samples, dtype=float))))
        previous_radius = max(0.0, _pick(by_frame[previous_frame].get("radius_px"), default=0.0))
        following_radius = max(0.0, _pick(by_frame[following_frame].get("radius_px"), default=0.0))
        if previous_radius <= 0.0 and following_radius <= 0.0:
            continue
        impact_alpha = (impact_frame - previous_frame) / float(following_frame - previous_frame)
        radius_px = (1.0 - impact_alpha) * previous_radius + impact_alpha * following_radius
        radius_m = float(dict(profile.data.get("sphere", {})).get("radius_m", 0.0) or 0.0)
        focal_y = float(dict(profile.data.get("camera", {})).get("fy", 0.0) or 0.0)
        impact_depth = _pick(
            by_frame.get(impact_frame, {}).get("object_ref_depth_m"),
            by_frame.get(impact_frame, {}).get("object_depth_smooth"),
            default=0.0,
        )
        # A boundary-clipped mask underestimates the radius precisely when the
        # sphere leaves the frame.  Use the known sphere geometry for contact.
        if radius_m > 0.0 and focal_y > 0.0 and impact_depth > 0.0:
            radius_px = focal_y * radius_m / impact_depth
        impact_center_v = floor_v - radius_px
        incoming_dt = impact_frame - previous_frame
        outgoing_dt = impact_frame - following_frame
        if incoming_dt <= 0 or outgoing_dt >= 0:
            continue
        incoming_acceleration = 2.0 * (
            impact_center_v - previous_v - incoming_velocity * incoming_dt
        ) / float(incoming_dt**2)
        outgoing_acceleration = 2.0 * (
            impact_center_v - following_v - outgoing_velocity * outgoing_dt
        ) / float(outgoing_dt**2)
        for frame in range(start, end + 1):
            row = by_frame.get(frame)
            if row is None:
                continue
            if frame <= impact_frame:
                dt = frame - previous_frame
                center_v = previous_v + incoming_velocity * dt + 0.5 * incoming_acceleration * dt**2
            else:
                dt = frame - following_frame
                center_v = following_v + outgoing_velocity * dt + 0.5 * outgoing_acceleration * dt**2
            if frame == impact_frame:
                center_v = impact_center_v
            radius_alpha = (frame - previous_frame) / float(following_frame - previous_frame)
            radius = (1.0 - radius_alpha) * previous_radius + radius_alpha * following_radius
            frame_depth = _pick(
                row.get("object_ref_depth_m"),
                row.get("object_depth_smooth"),
                default=0.0,
            )
            if radius_m > 0.0 and focal_y > 0.0 and frame_depth > 0.0:
                radius = focal_y * radius_m / frame_depth
            for field in ("ref_v", "ref_v_smooth", "ref_v_fit"):
                row[field] = f"{center_v:.3f}"
            row["support_v"] = f"{center_v + radius:.3f}"
            row["support_v_raw"] = f"{center_v + radius:.3f}"
            row["support_dv"] = f"{radius:.3f}"
            row["support_dv_smooth"] = f"{radius:.3f}"
            row["radius_px"] = f"{radius:.3f}"
            row["ref_source"] = "vlm_audio_out_of_frame_floor_impact"
            changed.append(frame)
        impacts.append(
            {
                "interval": [start, end],
                "impact_frame": impact_frame,
                "audio_score": audio_score,
                "floor_center_v": impact_center_v,
                "source": "vlm_out_of_frame+audio_impulse+floor_contact_candidate",
            }
        )
    if changed:
        write_csv(observation_path, observations)
    result = {
        "status": "refined" if changed else "no_joint_floor_impact_evidence",
        "changed_frames": sorted(set(changed)),
        "impacts": impacts,
        "human_state_optimized": False,
        "case_dispatch_used": False,
    }
    write_json(profile.result_dir / "offscreen_floor_impact_refinement.json", result)
    return result


def build(profile: CaseProfile) -> dict[str, object]:
    paths = stage_paths(profile)
    tracking_dir = profile.sample_dir / "results/tracking"
    ball_traj_csv = _first_existing(tracking_dir / "object_trajectory.csv", tracking_dir / "ball_trajectory.csv")
    center_traj_csv = _first_existing(
        tracking_dir / "object_center_trajectory.csv",
        tracking_dir / "cotracker_center_trajectory.csv",
    )
    points_csv = _first_existing(tracking_dir / "object_points.csv", tracking_dir / "cotracker_points.csv")
    depth_csv = _first_existing(
        profile.sample_dir / "results/da3/priors/object_depth_prior.csv",
        profile.sample_dir / "results/da3/priors/ball_depth_prior.csv",
    )
    if not ball_traj_csv.exists():
        raise FileNotFoundError(f"Missing SAM2/object trajectory: {ball_traj_csv}")
    if not center_traj_csv.exists():
        raise FileNotFoundError(f"Missing CoTracker/object center trajectory: {center_traj_csv}")
    if not depth_csv.exists():
        raise FileNotFoundError(f"Missing DA3 object depth prior: {depth_csv}")

    ball_by_frame = _by_frame(read_csv(ball_traj_csv))
    center_by_frame = _by_frame(read_csv(center_traj_csv))
    points_by_frame = _by_frame(read_csv(points_csv)) if points_csv.exists() else {}
    depth_by_frame = _by_frame(read_csv(depth_csv))
    identity_selection_path = profile.result_dir / "sphere_identity_selection.csv"
    identity_by_frame = _by_frame(read_csv(identity_selection_path)) if identity_selection_path.exists() else {}
    identity_config = dict(dict(profile.data.get("vlm", {})).get("single_identity", {}))
    single_identity_enabled = bool(identity_config.get("enabled", False)) and (
        "disable_vlm_semantic_evidence" not in set(profile.data.get("ablation_flags", ()))
    )
    prefer_visible_mask = single_identity_enabled and bool(identity_config.get("prefer_visible_mask", False))
    maximum_gap = max(0, int(identity_config.get("interpolate_missing_mask_max_frames", 0)))
    offscreen_intervals = _accepted_offscreen_intervals(profile) if single_identity_enabled else []
    ballistic_support_frames = max(2, int(identity_config.get("projected_ballistic_support_frames", 5)))
    visible_mask_centers = {
        frame: (_pick(row.get("ball_center_x")), _pick(row.get("ball_center_y")))
        for frame, row in ball_by_frame.items()
        if row.get("ball_center_x") not in {"", None} and row.get("ball_center_y") not in {"", None}
    }
    projected_ballistic_centers = _projected_ballistic_gap_centers(
        visible_mask_centers,
        offscreen_intervals,
        support_frames=ballistic_support_frames,
    )

    def interpolated_mask_center(frame: int) -> tuple[float, float] | None:
        if frame in visible_mask_centers:
            return visible_mask_centers[frame]
        if maximum_gap <= 0:
            return None
        previous = max((value for value in visible_mask_centers if value < frame), default=None)
        following = min((value for value in visible_mask_centers if value > frame), default=None)
        if previous is None or following is None or following - previous - 1 > maximum_gap:
            return None
        alpha = (frame - previous) / (following - previous)
        p0 = visible_mask_centers[previous]
        p1 = visible_mask_centers[following]
        return (
            (1.0 - alpha) * p0[0] + alpha * p1[0],
            (1.0 - alpha) * p0[1] + alpha * p1[1],
        )
    rows = []
    for fr in sorted(ball_by_frame):
        ball = ball_by_frame.get(fr, {})
        center = center_by_frame.get(fr, {})
        points = points_by_frame.get(fr, {})
        depth = depth_by_frame.get(fr, {})
        identity = identity_by_frame.get(fr, {})
        ballistic_center = projected_ballistic_centers.get(fr)
        mask_center = (
            ballistic_center
            if ballistic_center is not None
            else interpolated_mask_center(fr)
            if prefer_visible_mask and not identity_by_frame
            else None
        )
        if mask_center is not None:
            center_u, center_v = mask_center
        elif single_identity_enabled and not identity_by_frame:
            # Without the discrete identity gate, ambiguous multi-component
            # masks remain the vision-only observation. Persistent tracking is
            # an alternative hypothesis, not an implicit oracle.
            center_u = _pick(ball.get("ball_center_x"), center.get("ball_center_x"), center.get("center_x"), depth.get("u"))
            center_v = _pick(ball.get("ball_center_y"), center.get("ball_center_y"), center.get("center_y"), depth.get("v"))
        else:
            center_u = _pick(identity.get("u"), center.get("ball_center_x"), center.get("center_x"), ball.get("ball_center_x"), depth.get("u"))
            center_v = _pick(identity.get("v"), center.get("ball_center_y"), center.get("center_y"), ball.get("ball_center_y"), depth.get("v"))
        radius_px = _pick(identity.get("radius_px"), depth.get("radius_px"), ball.get("radius"), default=0.0)
        bottom_v = _pick(points.get("bottom_y"), default=center_v + radius_px)
        z = _pick(depth.get("da3_depth_smooth"), depth.get("da3_depth_raw"), default=1.0)
        is_offscreen = ballistic_center is not None
        obs_conf = "0.650000" if is_offscreen else "1.000000" if ball else "0.000000"
        proxy_conf = "1.000000" if center.get("source") else "0.750000"
        rows.append(
            {
                "frame": str(fr),
                "time": ball.get("time", center.get("time", depth.get("time", ""))),
                "ref_u": f"{center_u:.3f}",
                "ref_v": f"{center_v:.3f}",
                "ref_u_smooth": f"{center_u:.3f}",
                "ref_v_smooth": f"{center_v:.3f}",
                "ref_u_fit": f"{center_u:.3f}",
                "ref_v_fit": f"{center_v:.3f}",
                "ref_source": (
                    "vlm_out_of_frame_projected_ballistic"
                    if is_offscreen
                    else "vlm_single_identity_visible_mask"
                    if mask_center is not None and fr in visible_mask_centers
                    else "vlm_single_identity_mask_gap_interpolation"
                    if mask_center is not None
                    else identity.get("selection_source", center.get("source", "cotracker_center_trajectory"))
                ),
                "ref_type": "center",
                "support_u": f"{center_u:.3f}",
                "support_v": f"{bottom_v:.3f}",
                "support_v_raw": f"{bottom_v:.3f}",
                "support_dv": f"{bottom_v - center_v:.3f}",
                "support_dv_smooth": f"{bottom_v - center_v:.3f}",
                "support_source": "cotracker_bottom" if points.get("bottom_y") else "sam2_center_plus_radius",
                "contact_u": "",
                "contact_v": "",
                "contact_source": "stage2_contact_policy",
                "object_ref_depth_m": f"{z:.6f}",
                "contact_proxy_depth_m": "",
                "contact_depth_offset_m": "",
                "ref_conf": "0.650000" if is_offscreen else "1.000000",
                "support_conf": "0.000000",
                "contact_conf": "0.000000",
                "depth_conf": "1.000000",
                "observation_conf": obs_conf,
                "visible_fraction": "0.000000" if is_offscreen else "1.000000",
                "occlusion_state": "absent" if is_offscreen else "visible",
                "vlm_visibility": "out_of_frame" if is_offscreen else "visible",
                "proxy_conf": proxy_conf,
                "proxy_jitter_px": "0.000000",
                "support_jitter_px": "0.000000",
                "proxy_sigma_px": "5.000000",
                "support_sigma_px": "8.000000",
                "active_label": "",
                "active_label_conf": "",
                "active_part_u": "",
                "active_part_v": "",
                "active_part_z": "",
                "contact_proxy_name": "",
                "da3_depth_raw": depth.get("da3_depth_raw", ""),
                "da3_depth_smooth": depth.get("da3_depth_smooth", ""),
                "object_depth_raw": f"{z:.6f}",
                "object_depth_smooth": f"{z:.6f}",
                "object_depth_confidence": "1.000000",
                "support_gap_px": "0.000000",
                "object_motion_score": "",
                "audio_score": "",
                "radius_px": f"{radius_px:.3f}",
            }
        )
    out = write_csv(paths["object_observations"], rows)
    write_csv(paths["object_local_points"], [], fields=["point_id", "part", "role", "local_x", "local_y", "local_z", "source"])
    metrics = {
        "component": "mask_track_center",
        "source": str(ball_traj_csv),
        "center_source": str(center_traj_csv),
        "points_source": str(points_csv) if points_csv.exists() else "",
        "depth_source": str(depth_csv),
        "identity_selection_source": str(identity_selection_path) if identity_selection_path.exists() else "",
        "object_observations": str(out),
        "local_points": "none_for_center_proxy",
        "rows": len(rows),
        "vlm_confirmed_offscreen_intervals": [list(value) for value in offscreen_intervals],
        "projected_ballistic_rows": len(projected_ballistic_centers),
        "policy": "generic one-entity identity selection when present; VLM-confirmed out-of-frame gaps use unclipped projected-ballistic continuation; other gaps retain mask/track fallback; contact fields are owned by Stage2",
    }
    write_json(paths["stage1_metrics"], metrics)
    return metrics
