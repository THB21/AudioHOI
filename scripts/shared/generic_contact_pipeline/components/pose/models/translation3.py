from __future__ import annotations

import math

from ....core.base.config import CaseProfile
from ....core.evaluation.ball_residuals import write_ball_residual_report
from ....core.base.camera import backproject_uvz
from ....core.base.io import float_or_none, read_csv, write_csv, write_json
from ....core.base.schema import stage_paths


def _line_angles_by_frame(profile: CaseProfile) -> dict[int, float]:
    path = profile.sample_dir / "results" / "tracking" / "object_mesh_tracks_test.csv"
    if not path.exists():
        return {}
    grouped: dict[int, dict[str, tuple[float, float]]] = {}
    for row in read_csv(path):
        pid = row.get("point_id", "")
        if pid not in {"tip_a", "tip_b"}:
            continue
        try:
            fr = int(float(row["frame"]))
            x = float(row["x"])
            y = float(row["y"])
        except Exception:
            continue
        grouped.setdefault(fr, {})[pid] = (x, y)
    out: dict[int, float] = {}
    for fr, pts in grouped.items():
        if "tip_a" not in pts or "tip_b" not in pts:
            continue
        ax, ay = pts["tip_a"]
        bx, by = pts["tip_b"]
        out[fr] = math.atan2(by - ay, bx - ax)
    return out


def _quat_z(angle: float | None) -> tuple[float, float, float, float]:
    if angle is None:
        return 1.0, 0.0, 0.0, 0.0
    half = 0.5 * angle
    return math.cos(half), 0.0, 0.0, math.sin(half)


def build(profile: CaseProfile) -> dict[str, object]:
    paths = stage_paths(profile)
    obs_rows = read_csv(paths["object_observations"])
    line_angles = _line_angles_by_frame(profile)
    rows = []
    for row in obs_rows:
        # Use the raw stable observation for the camera ray. Smoothing belongs
        # in the optimizer, not in the definition of the 2D anchor, otherwise
        # later backprojection breaks exact video overlay consistency.
        u = float_or_none(row.get("ref_u")) or float_or_none(row.get("ref_u_smooth")) or float_or_none(row.get("ref_u_fit")) or 0.0
        v = float_or_none(row.get("ref_v")) or float_or_none(row.get("ref_v_smooth")) or float_or_none(row.get("ref_v_fit")) or 0.0
        z = (
            float_or_none(row.get("object_depth_smooth"))
            or float_or_none(row.get("object_ref_depth_m"))
            or float_or_none(row.get("da3_depth_smooth"))
            or 1.0
        )
        tx, ty, tz = backproject_uvz(float(u), float(v), float(z), profile.camera)
        support_dv = float_or_none(row.get("support_dv_smooth")) or float_or_none(row.get("support_dv")) or 0.0
        fr = int(float(row.get("frame", "0") or 0))
        qw, qx, qy, qz = _quat_z(line_angles.get(fr))
        rows.append(
            {
                "frame": row.get("frame", ""),
                "time": row.get("time", ""),
                "tx": f"{tx:.6f}",
                "ty": f"{ty:.6f}",
                "tz": f"{tz:.6f}",
                "qw": f"{qw:.6f}",
                "qx": f"{qx:.6f}",
                "qy": f"{qy:.6f}",
                "qz": f"{qz:.6f}",
                "radius_m": "0.120000",
                "coord_frame": "gvhmr_incam",
                "u_obs": f"{float(u):.3f}",
                "v_obs": f"{float(v):.3f}",
                "u_proj": f"{float(u):.3f}",
                "v_proj": f"{float(v):.3f}",
                "bottom_proj_v": f"{float(v) + float(support_dv):.3f}",
                "floor_v": row.get("support_v", ""),
                "u_ref_obs": f"{float(u):.3f}",
                "v_ref_obs": f"{float(v):.3f}",
                "support_v_obs": row.get("support_v", ""),
                "depth_prior_m": row.get("object_depth_smooth", row.get("object_ref_depth_m", "")),
                "contact_depth_offset_m": row.get("contact_depth_offset_m", ""),
                "contact_frame": "0",
                "audio_contact_frame": "0",
                "source": "translation3_from_stage1_observation_backproject_uvz",
            }
        )
    out = write_csv(paths["object_pose_init"], rows)
    residual_report = write_ball_residual_report(
        profile,
        out,
        stage_label="stage3_translation3_init",
        out_name="ball_stage3_translation3_residuals.csv",
    )
    metrics = {
        "component": "translation3",
        "source": str(paths["object_observations"]),
        "object_pose_init": str(out),
        "rows": len(rows),
        "line_orientation_frames": len(line_angles),
        "policy": "clean backprojection from Stage1 object observation using K_fullimg; optional tip_a/tip_b tracks define camera-plane object orientation",
        "optimizer_style_residual_report": residual_report,
    }
    write_json(paths["stage3_metrics"], metrics)
    return metrics
