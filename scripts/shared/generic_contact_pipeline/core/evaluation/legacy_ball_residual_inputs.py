"""Legacy basketball/football CSV adapter for residual parity evaluation.

This module is deliberately outside ``core.solver``. It translates historical
ball-case artifacts into the case-independent residual input provider boundary;
it is not a production solver input contract.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from ..contact_constraints import adapt_contact_state_rows
from ..human_sites import adapt_human_site_rows
from ..solver.residual_inputs import (
    ResidualInputRequest,
    build_residual_input_bundle,
    build_state_regularization_residual_inputs,
    build_world_space_contact_residual_inputs,
)
from ..state import SphereGeometryProvider


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _read_csv_by_frame(path: Path) -> dict[int, dict[str, str]]:
    return {int(float(row["frame"])): row for row in _read_csv(path)}


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


def _site_id(body_part: str, side: str) -> str:
    return f"{side}_{body_part}" if side in {"left", "right"} else body_part


def build_legacy_ball_residual_input_bundle(
    result_dir: Path,
    residual_execution_plan: dict[str, object] | object,
) -> dict[str, dict[str, Any]]:
    """Translate historical ball CSV traces for residual parity only."""

    pose_by_frame = _read_csv_by_frame(result_dir / "object_pose.csv")
    init_by_frame = _read_csv_by_frame(result_dir / "object_pose_init.csv")
    obs_by_frame = _read_csv_by_frame(result_dir / "object_observations.csv")
    frames = sorted(set(pose_by_frame) & set(obs_by_frame))
    sample_id = "legacy_ball_residual_parity"
    contact_states = adapt_contact_state_rows(
        sample_id,
        _read_csv(result_dir / "contact_state_frames.csv"),
        str(result_dir / "contact_state_frames.csv"),
    )
    human_sites = adapt_human_site_rows(
        sample_id,
        _read_csv(result_dir / "human_sites.csv"),
        str(result_dir / "human_sites.csv"),
    ).measurements
    human_sites_by_key = {
        (site.frame, _site_id(site.site.body_part, site.site.side)): site.xyz_m for site in human_sites
    }
    object_states = {
        frame: xyz for frame, row in pose_by_frame.items() if (xyz := _pose_xyz(row)) is not None
    }
    radius_m = _finite_float(pose_by_frame[min(pose_by_frame)], "radius_m")
    if radius_m is None or radius_m <= 0.0:
        raise ValueError("legacy sphere parity input requires a positive radius_m")
    sphere_geometry = SphereGeometryProvider(radius_m)
    active_frames: list[int] = []
    source_sites: dict[int, tuple[float, float, float]] = {}
    for state in contact_states:
        if not state.human_active:
            continue
        site = human_sites_by_key.get((state.frame, _site_id(state.human_site.body_part, state.human_site.side)))
        if site is None:
            continue
        active_frames.append(state.frame)
        source_sites[state.frame] = site

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

    def contact_distance(request: ResidualInputRequest) -> dict[str, Any] | None:
        payload = build_world_space_contact_residual_inputs(
            factor_id=request.factor_id,
            geometry_provider=sphere_geometry,
            object_states=object_states,
            source_sites=source_sites,
            active_frames=active_frames,
            object_feature_id="object:surface",
            weight=1.0,
            sigma_m=1.0,
        )
        return payload.get(request.factor_id)

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
