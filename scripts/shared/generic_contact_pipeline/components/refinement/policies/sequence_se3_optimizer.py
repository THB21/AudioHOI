from __future__ import annotations

from ....core.base.config import CaseProfile
from ....core.base.io import read_csv, write_csv, write_json
from ....core.base.schema import stage_paths
from ..sequence_se3_optimizer import smooth_quaternion_pose_sequence


def apply(profile: CaseProfile) -> dict[str, object]:
    paths = stage_paths(profile)
    pose_path = paths["object_pose"]
    if not pose_path.exists():
        metrics = {"component": "sequence_se3_optimizer", "enabled": False, "reason": "missing object_pose.csv"}
        write_json(profile.result_dir / "sequence_se3_optimizer_metrics.json", metrics)
        return metrics
    rows = read_csv(pose_path)
    if not rows or not {"tx", "ty", "tz", "qw", "qx", "qy", "qz"}.issubset(rows[0].keys()):
        metrics = {"component": "sequence_se3_optimizer", "enabled": False, "reason": "object_pose is not quaternion SE3"}
        write_json(profile.result_dir / "sequence_se3_optimizer_metrics.json", metrics)
        return metrics
    config = profile.data.get("sequence_se3_optimizer", {}) if isinstance(profile.data.get("sequence_se3_optimizer", {}), dict) else {}
    out_rows, audit_rows = smooth_quaternion_pose_sequence(
        rows,
        prior_w=float(config.get("prior_w", 0.34)),
        smooth_w=float(config.get("smooth_w", 0.95)),
        accel_w=float(config.get("accel_w", 4.2)),
    )
    pose_out = write_csv(pose_path, out_rows)
    audit_out = write_csv(profile.result_dir / "sequence_se3_audit.csv", audit_rows)
    metrics = {
        "component": "sequence_se3_optimizer",
        "enabled": True,
        "object_pose": str(pose_out),
        "sequence_se3_audit": str(audit_out),
        "rows": len(out_rows),
        "smoothed_rows": sum(1 for row in audit_rows if row.get("se3_smoothed") == "1"),
        "velocity_spikes": sum(1 for row in audit_rows if row.get("se3_velocity_spike") == "1"),
        "accel_spikes": sum(1 for row in audit_rows if row.get("se3_accel_spike") == "1"),
        "policy": "generic quaternion SE3 sequence optimizer with velocity and acceleration residuals",
    }
    write_json(profile.result_dir / "sequence_se3_optimizer_metrics.json", metrics)
    return metrics
