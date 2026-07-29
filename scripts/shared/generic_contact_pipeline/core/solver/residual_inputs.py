from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def _read_csv_by_frame(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="") as handle:
        return {int(float(row["frame"])): row for row in csv.DictReader(handle)}


def _finite_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key)
    if value in {None, ""}:
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    return out


def _pose_xyz(row: dict[str, str]) -> list[float] | None:
    values = [_finite_float(row, key) for key in ("tx", "ty", "tz")]
    if any(value is None for value in values):
        return None
    return [float(value) for value in values]


def _execution_records(residual_execution_plan: dict[str, object] | object) -> list[dict[str, object]]:
    if isinstance(residual_execution_plan, dict):
        records = residual_execution_plan.get("records", [])
        return [record for record in records if isinstance(record, dict)]
    records = getattr(residual_execution_plan, "records", ())
    return [
        {
            "factor_id": record.factor_id,
            "residual_fn_ref": record.residual_fn_ref,
            "evaluator_ref": record.evaluator_ref,
        }
        for record in records
    ]


def build_legacy_ball_residual_input_bundle(
    result_dir: Path,
    residual_execution_plan: dict[str, object] | object,
) -> dict[str, dict[str, Any]]:
    """Build explicit generic residual inputs from current ball-case CSV traces.

    This is an adapter for residual parity work only. It does not solve,
    write accepted outputs, or infer object-specific behavior. The contact
    residual is a legacy 2.5D proxy using image-space contact coordinates plus
    depth offset; the production path should replace it with GeometryProvider
    3D site distances.
    """

    pose_by_frame = _read_csv_by_frame(result_dir / "object_pose.csv")
    obs_by_frame = _read_csv_by_frame(result_dir / "object_observations.csv")
    contact_by_frame = _read_csv_by_frame(result_dir / "object_contact_points.csv")
    frames = sorted(set(pose_by_frame) & set(obs_by_frame))
    bundle: dict[str, dict[str, Any]] = {}

    for record in _execution_records(residual_execution_plan):
        factor_id = str(record.get("factor_id", ""))
        residual_ref = str(record.get("residual_fn_ref", ""))

        if residual_ref == "shadow_residual::metric_depth":
            predicted: list[float] = []
            target: list[float] = []
            for frame in frames:
                pose = pose_by_frame[frame]
                obs = obs_by_frame[frame]
                pred = _finite_float(pose, "tz")
                tgt = _finite_float(obs, "object_ref_depth_m")
                if pred is None or tgt is None:
                    continue
                predicted.append(pred)
                target.append(tgt)
            if predicted:
                bundle[factor_id] = {
                    "predicted_depth_m": predicted,
                    "target_depth_m": target,
                    "weight": 1.0,
                    "sigma_m": 1.0,
                }

        elif residual_ref == "shadow_residual::contact_distance":
            anchors: list[list[float]] = []
            targets: list[list[float]] = []
            for frame, contact in sorted(contact_by_frame.items()):
                if contact.get("contact_active") != "1":
                    continue
                pose = pose_by_frame.get(frame)
                if pose is None:
                    continue
                cu = _finite_float(contact, "contact_u")
                cv = _finite_float(contact, "contact_v")
                dz = _finite_float(contact, "contact_depth_offset_m") or 0.0
                pu = _finite_float(pose, "u_proj")
                pv = _finite_float(pose, "v_proj")
                if cu is None or cv is None or pu is None or pv is None:
                    continue
                anchors.append([cu, cv, dz])
                targets.append([pu, pv, 0.0])
            if anchors:
                bundle[factor_id] = {
                    "anchors": anchors,
                    "targets": targets,
                    "weight": 1.0,
                    "sigma_m": 1.0,
                }

        elif residual_ref in {"shadow_residual::temporal_velocity", "shadow_residual::temporal_acceleration"}:
            ordered_xyz = [(frame, _pose_xyz(pose_by_frame[frame])) for frame in sorted(pose_by_frame)]
            ordered_xyz = [(frame, xyz) for frame, xyz in ordered_xyz if xyz is not None]
            if len(ordered_xyz) >= 2:
                if residual_ref == "shadow_residual::temporal_velocity":
                    x = ordered_xyz[1][1]
                    prev = ordered_xyz[0][1]
                else:
                    x = ordered_xyz[2][1] if len(ordered_xyz) >= 3 else ordered_xyz[1][1]
                    prev = ordered_xyz[1][1] if len(ordered_xyz) >= 3 else ordered_xyz[0][1]
                bundle[factor_id] = {
                    "x": x,
                    "prev": prev,
                    "weight": 1.0,
                    "scales": [1.0, 1.0, 1.0],
                }

        elif residual_ref == "shadow_residual::pose_prior":
            first = pose_by_frame[min(pose_by_frame)]
            xyz = _pose_xyz(first)
            if xyz is not None:
                state = [0.0, 0.0, 0.0, *xyz]
                bundle[factor_id] = {
                    "x": state,
                    "ref": state,
                    "init": state,
                    "rot_bound": 1.0,
                    "xy_bound": 1.0,
                    "z_bound": 1.0,
                    "w_prior_rot": 1.0,
                    "w_prior_xy": 1.0,
                    "w_prior_z": 1.0,
                }

    return bundle
