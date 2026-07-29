"""Legacy basketball/football CSV adapter for residual parity evaluation.

This module is deliberately outside ``core.solver``. It translates historical
ball-case artifacts into the case-independent residual input provider boundary;
it is not a production solver input contract.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from ..solver.residual_inputs import (
    ResidualInputRequest,
    build_residual_input_bundle,
    build_state_regularization_residual_inputs,
)


def _read_csv_by_frame(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="") as handle:
        return {int(float(row["frame"])): row for row in csv.DictReader(handle)}


def _finite_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key)
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _pose_xyz(row: dict[str, str]) -> list[float] | None:
    values = [_finite_float(row, key) for key in ("tx", "ty", "tz")]
    if any(value is None for value in values):
        return None
    return [float(value) for value in values]


def _state_values(row: dict[str, str], fields: tuple[str, ...]) -> list[float] | None:
    values = [_finite_float(row, field) for field in fields]
    if any(value is None for value in values):
        return None
    return [float(value) for value in values]


def build_legacy_ball_residual_input_bundle(
    result_dir: Path,
    residual_execution_plan: dict[str, object] | object,
) -> dict[str, dict[str, Any]]:
    """Translate historical ball CSV traces for residual parity only."""

    pose_by_frame = _read_csv_by_frame(result_dir / "object_pose.csv")
    init_by_frame = _read_csv_by_frame(result_dir / "object_pose_init.csv")
    obs_by_frame = _read_csv_by_frame(result_dir / "object_observations.csv")
    contact_by_frame = _read_csv_by_frame(result_dir / "object_contact_points.csv")
    frames = sorted(set(pose_by_frame) & set(obs_by_frame))

    def metric_depth(_: ResidualInputRequest) -> dict[str, Any] | None:
        predicted: list[float] = []
        target: list[float] = []
        for frame in frames:
            pred = _finite_float(pose_by_frame[frame], "tz")
            tgt = _finite_float(obs_by_frame[frame], "object_ref_depth_m")
            if pred is None or tgt is None:
                continue
            predicted.append(pred)
            target.append(tgt)
        if not predicted:
            return None
        return {
            "predicted_depth_m": predicted,
            "target_depth_m": target,
            "weight": 1.0,
            "sigma_m": 1.0,
        }

    def contact_distance(_: ResidualInputRequest) -> dict[str, Any] | None:
        # Historical 2.5D proxy only. Production contact must use GeometryProvider
        # world-space entity sites instead of these image/depth-offset columns.
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
        if not anchors:
            return None
        return {"anchors": anchors, "targets": targets, "weight": 1.0, "sigma_m": 1.0}

    def temporal(request: ResidualInputRequest) -> dict[str, Any] | None:
        ordered_xyz = [(frame, _pose_xyz(pose_by_frame[frame])) for frame in sorted(pose_by_frame)]
        ordered_xyz = [(frame, xyz) for frame, xyz in ordered_xyz if xyz is not None]
        if len(ordered_xyz) < 2:
            return None
        if request.residual_fn_ref == "shadow_residual::temporal_velocity":
            x = ordered_xyz[1][1]
            prev = ordered_xyz[0][1]
        else:
            x = ordered_xyz[2][1] if len(ordered_xyz) >= 3 else ordered_xyz[1][1]
            prev = ordered_xyz[1][1] if len(ordered_xyz) >= 3 else ordered_xyz[0][1]
        return {"x": x, "prev": prev, "weight": 1.0, "scales": [1.0, 1.0, 1.0]}

    def pose_prior(_: ResidualInputRequest) -> dict[str, Any] | None:
        first = pose_by_frame[min(pose_by_frame)]
        xyz = _pose_xyz(first)
        if xyz is None:
            return None
        state = [0.0, 0.0, 0.0, *xyz]
        return {
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

    def regularization(request: ResidualInputRequest) -> dict[str, Any] | None:
        values: list[list[float]] = []
        target: list[list[float]] = []
        for frame in sorted(set(pose_by_frame) & set(init_by_frame)):
            value = _state_values(pose_by_frame[frame], ("tx", "ty", "tz"))
            reference = _state_values(init_by_frame[frame], ("tx", "ty", "tz"))
            if value is None or reference is None:
                continue
            values.append(value)
            target.append(reference)
        payload = build_state_regularization_residual_inputs(
            factor_id=request.factor_id,
            values=values,
            target=target,
            scales=(1.0, 1.0, 1.0),
            weight=1.0,
        )
        return payload.get(request.factor_id)

    return build_residual_input_bundle(
        residual_execution_plan,
        {
            "shadow_residual::metric_depth": metric_depth,
            "shadow_residual::contact_distance": contact_distance,
            "shadow_residual::temporal_velocity": temporal,
            "shadow_residual::temporal_acceleration": temporal,
            "shadow_residual::pose_prior": pose_prior,
            "shadow_residual::regularization": regularization,
        },
    )
