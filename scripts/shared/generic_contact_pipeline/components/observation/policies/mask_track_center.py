from __future__ import annotations

from ....core.base.config import CaseProfile
from ....core.base.io import read_csv, repo_path, write_csv, write_json
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
    # Held-out cases can provide the already materialized *visual* Stage-1 table
    # even when the original SAM2/DA3 intermediates were not retained.  Reusing
    # that table is preferable to silently using a solved pose.  In the strict
    # no-audio arm, all audio-valued columns are zeroed before contact extraction.
    baseline_obs_raw = profile.baseline.get("object_observations_csv", "")
    baseline_obs = repo_path(baseline_obs_raw) if baseline_obs_raw else None
    raw_visual_inputs_complete = (
        ball_traj_csv.exists() and center_traj_csv.exists() and depth_csv.exists()
    )
    if not raw_visual_inputs_complete and baseline_obs is not None and baseline_obs.exists():
        rows = read_csv(baseline_obs)
        audio_disabled = "disable_audio_events" in set(profile.data.get("ablation_flags", []))
        if audio_disabled:
            for row in rows:
                for key in list(row):
                    if key.lower() in {"audio_score", "audio_support", "audio_confidence"}:
                        row[key] = "0.000000"
                    elif key.lower() in {"audio_contact_frame", "audio_event", "audio_active"}:
                        row[key] = "0"
        out = write_csv(paths["object_observations"], rows)
        write_csv(paths["object_local_points"], [], fields=["point_id", "part", "role", "local_x", "local_y", "local_z", "source"])
        metrics = {
            "component": "mask_track_center",
            "source": str(baseline_obs),
            "source_kind": "configured_visual_observation_fallback",
            "audio_columns_zeroed": audio_disabled,
            "object_observations": str(out),
            "local_points": "none_for_center_proxy",
            "rows": len(rows),
            "policy": "reuse materialized visual Stage-1 observations only; solved object pose is not read",
        }
        write_json(paths["stage1_metrics"], metrics)
        return metrics
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
    rows = []
    for fr in sorted(ball_by_frame):
        ball = ball_by_frame.get(fr, {})
        center = center_by_frame.get(fr, {})
        points = points_by_frame.get(fr, {})
        depth = depth_by_frame.get(fr, {})
        center_u = _pick(center.get("ball_center_x"), center.get("center_x"), ball.get("ball_center_x"), depth.get("u"))
        center_v = _pick(center.get("ball_center_y"), center.get("center_y"), ball.get("ball_center_y"), depth.get("v"))
        radius_px = _pick(depth.get("radius_px"), ball.get("radius"), default=0.0)
        bottom_v = _pick(points.get("bottom_y"), default=center_v + radius_px)
        z = _pick(depth.get("da3_depth_smooth"), depth.get("da3_depth_raw"), default=1.0)
        obs_conf = "1.000000" if ball else "0.000000"
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
                "ref_source": center.get("source", "cotracker_center_trajectory"),
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
                "ref_conf": "1.000000",
                "support_conf": "0.000000",
                "contact_conf": "0.000000",
                "depth_conf": "1.000000",
                "observation_conf": obs_conf,
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
        "object_observations": str(out),
        "local_points": "none_for_center_proxy",
        "rows": len(rows),
        "policy": "generic mask/object center/radius + optional CoTracker/object bottom + DA3 depth only; contact fields are owned by Stage2",
    }
    write_json(paths["stage1_metrics"], metrics)
    return metrics
