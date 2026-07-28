from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..base.camera import backproject_uvz
from ..base.config import CaseProfile
from ..base.io import read_csv, repo_relative_value, write_json
from ..contact_constraints import ContactMode, adapt_contact_event_rows, adapt_contact_state_rows
from ..human_sites import adapt_human_site_rows
from ..measurements import MetricDepthMeasurement, Point2DMeasurement, adapt_legacy_observation_rows


SPHERE_CANDIDATE_NAME = "generic_sphere_sequence_candidate.csv"
SPHERE_RESIDUAL_NAME = "generic_sphere_sequence_residuals.csv"
SPHERE_ATTEMPT_NAME = "generic_sphere_sequence_attempt.json"

POSE_FIELDS = [
    "frame", "time", "tx", "ty", "tz", "qw", "qx", "qy", "qz", "radius_m", "coord_frame",
    "u_obs", "v_obs", "radius_obs_px", "u_proj", "v_proj", "radius_proj_px", "bottom_proj_v",
    "floor_v", "support_type", "support_source", "support_confidence", "residual_px", "contact_frame",
    "audio_contact_frame", "human_contact_event", "floor_contact_event", "human_contact_state",
    "floor_contact_state", "contact_part", "contact_side", "contact_label", "active_part", "active_part_y",
    "active_part_z", "u_ref_obs", "v_ref_obs", "contact_u", "contact_v", "contact_depth_offset_m",
    "contact_depth_offset_used_m", "z_contact_final", "global_z_ref", "z_ref_global_shift",
    "z_ref_anchor_segment", "contact_depth_gap",
]


@dataclass(frozen=True)
class SphereSequenceParameters:
    minimum_depth_m: float = 0.20
    maximum_contact_depth_offset_m: float = 1.0
    reference_weight: float = 0.7
    temporal_weight: float = 5.0
    physical_xz_weight: float = 1.25
    physical_y_weight: float = 1.5
    gravity_mps2: float = 9.81
    robust_loss: str = "soft_l1"
    robust_scale: float = 1.0
    max_function_evaluations: int = 400


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _site_id(body_part: str, side: str) -> str:
    return f"{side}_{body_part}" if side in {"left", "right"} else body_part


def _write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fields} for row in rows])


def _build_anchor_segment_reference(
    z_init: np.ndarray,
    anchor_mask: np.ndarray,
    anchor_values: np.ndarray,
    minimum_depth_m: float,
) -> np.ndarray:
    anchor_idx = np.flatnonzero(anchor_mask)
    if len(anchor_idx) == 0:
        raise RuntimeError("No human contact events found; cannot build anchor reference")
    z_ref = np.asarray(z_init, dtype=np.float64).copy()
    first = int(anchor_idx[0])
    last = int(anchor_idx[-1])
    z_ref[: first + 1] = float(anchor_values[first])
    for left, right in zip(anchor_idx[:-1], anchor_idx[1:]):
        left = int(left)
        right = int(right)
        if right <= left:
            continue
        alpha = np.linspace(0.0, 1.0, right - left + 1, dtype=np.float64)
        z_ref[left : right + 1] = (1.0 - alpha) * float(anchor_values[left]) + alpha * float(anchor_values[right])
    z_ref[last:] = float(anchor_values[last])
    return np.maximum(z_ref, minimum_depth_m)


def _reconstruct_xyz(u: np.ndarray, v: np.ndarray, z: np.ndarray, camera: dict[str, float]) -> np.ndarray:
    x = (u - float(camera["cx"])) * z / float(camera["fx"])
    y = (v - float(camera["cy"])) * z / float(camera["fy"])
    return np.stack([x, y, z], axis=1)


def _solve_anchor_interpolation(
    z_ref: np.ndarray,
    anchor_mask: np.ndarray,
    anchor_values: np.ndarray,
    u_obs: np.ndarray,
    v_obs: np.ndarray,
    times: np.ndarray,
    flight_mask: np.ndarray,
    camera: dict[str, float],
    parameters: SphereSequenceParameters,
) -> np.ndarray:
    try:
        from scipy.optimize import least_squares
    except ImportError as exc:  # pragma: no cover - exercised in the audiohoi runtime
        raise RuntimeError("sphere sequence solve requires scipy; run with the audiohoi environment") from exc

    free_idx = np.flatnonzero(~anchor_mask)
    anchor_idx = np.flatnonzero(anchor_mask)
    if len(anchor_idx) == 0:
        raise RuntimeError("No anchors available for sphere sequence solve")
    dt = np.diff(times)
    dt_mean = float(np.mean(dt)) if len(dt) else 1.0 / 30.0
    gravity_step = parameters.gravity_mps2 * dt_mean**2
    flight_triplet = np.zeros(len(z_ref), dtype=bool)
    if len(z_ref) >= 3:
        flight_triplet[1:-1] = flight_mask[:-2] & flight_mask[1:-1] & flight_mask[2:]

    def unpack(free_values: np.ndarray) -> np.ndarray:
        z = np.asarray(z_ref, dtype=np.float64).copy()
        z[anchor_idx] = anchor_values[anchor_idx]
        z[free_idx] = free_values
        return np.maximum(z, parameters.minimum_depth_m)

    def residual(free_values: np.ndarray) -> np.ndarray:
        z = unpack(free_values)
        xyz = _reconstruct_xyz(u_obs, v_obs, z, camera)
        values = [parameters.reference_weight * (z[free_idx] - z_ref[free_idx])]
        if len(z) >= 3:
            second_z = z[2:] - 2.0 * z[1:-1] + z[:-2]
            smooth_mask = ~anchor_mask[1:-1]
            if np.any(smooth_mask):
                values.append(parameters.temporal_weight * second_z[smooth_mask])
            physical_mask = flight_triplet[1:-1]
            if np.any(physical_mask):
                second_x = xyz[2:, 0] - 2.0 * xyz[1:-1, 0] + xyz[:-2, 0]
                second_y = xyz[2:, 1] - 2.0 * xyz[1:-1, 1] + xyz[:-2, 1]
                if parameters.physical_xz_weight > 0.0:
                    values.append(parameters.physical_xz_weight * second_x[physical_mask])
                    values.append(parameters.physical_xz_weight * second_z[physical_mask])
                if parameters.physical_y_weight > 0.0:
                    values.append(parameters.physical_y_weight * (second_y[physical_mask] - gravity_step))
        return np.concatenate([np.ravel(item) for item in values]).astype(np.float64)

    result = least_squares(
        residual,
        x0=z_ref[free_idx].copy(),
        method="trf",
        loss=parameters.robust_loss,
        f_scale=parameters.robust_scale,
        max_nfev=parameters.max_function_evaluations,
    )
    return unpack(result.x)


def solve_sphere_sequence_candidate(
    profile: CaseProfile,
    result_dir: Path,
    *,
    contact_events_csv: Path,
    human_sites_csv: Path,
    support_geometry_json: Path,
    candidate_dir: Path,
    parameters: SphereSequenceParameters = SphereSequenceParameters(),
) -> dict[str, object]:
    if profile.component("pose_model") != "translation3" or profile.component("geometry_model") != "sphere_proxy":
        raise ValueError("sphere sequence candidate requires translation3 + sphere_proxy")
    if candidate_dir.resolve() == result_dir.resolve():
        raise ValueError("candidate directory must not equal the canonical result directory")

    observation_csv = result_dir / "object_observations.csv"
    state_csv = result_dir / "contact_state_frames.csv"
    input_paths = {
        "object_measurements": observation_csv,
        "contact_events": contact_events_csv,
        "contact_timeline": state_csv,
        "human_sites": human_sites_csv,
        "support_geometry": support_geometry_json,
    }
    for name, path in input_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"missing {name}: {path}")

    observation_rows = read_csv(observation_csv)
    measurement_result = adapt_legacy_observation_rows(profile.case_name, observation_rows, str(repo_relative_value(observation_csv)))
    events = adapt_contact_event_rows(profile.case_name, read_csv(contact_events_csv), str(repo_relative_value(contact_events_csv)))
    states = adapt_contact_state_rows(profile.case_name, read_csv(state_csv), str(repo_relative_value(state_csv)))
    human_sites = adapt_human_site_rows(profile.case_name, read_csv(human_sites_csv), str(repo_relative_value(human_sites_csv))).measurements
    support_geometry = json.loads(support_geometry_json.read_text())
    support_floor_v = float(support_geometry["floor_v"])
    support_type = str(support_geometry.get("support_type", "floor"))
    support_source = str(support_geometry.get("source", "support_geometry_observation"))
    support_confidence = float(support_geometry.get("confidence", 0.0))

    raw_centers: dict[int, Point2DMeasurement] = {}
    depths: dict[int, MetricDepthMeasurement] = {}
    for measurement in measurement_result.measurements:
        if isinstance(measurement, Point2DMeasurement) and measurement.meta.feature.semantic_role == "object_center":
            raw_centers[measurement.meta.frame] = measurement
        elif isinstance(measurement, MetricDepthMeasurement) and measurement.meta.feature.semantic_role == "object_center_depth":
            depths[measurement.meta.frame] = measurement

    state_by_frame = {state.frame: state for state in states}
    site_by_key = {(site.frame, _site_id(site.site.body_part, site.site.side)): site for site in human_sites}
    frames = sorted(set(raw_centers) & set(depths) & set(state_by_frame))
    if len(frames) != len(observation_rows):
        raise ValueError(f"typed sphere inputs are not frame-complete: {len(frames)} != {len(observation_rows)}")

    human_event_frames = {event.peak_frame for event in events if event.mode != ContactMode.SUPPORT}
    floor_event_frames = {event.peak_frame for event in events if event.mode == ContactMode.SUPPORT}
    if not human_event_frames:
        raise RuntimeError("No human contact events found; cannot solve sphere sequence")

    fallback_site = next(
        (_site_id(state.human_site.body_part, state.human_site.side) for state in states if state.human_site.body_part not in {"environment", "unknown"}),
        "right_hand",
    )
    labels: list[str] = []
    site_xyz: list[tuple[float, float, float]] = []
    for frame in frames:
        state = state_by_frame[frame]
        label = _site_id(state.human_site.body_part, state.human_site.side)
        if (frame, label) not in site_by_key:
            label = fallback_site
        site = site_by_key.get((frame, label))
        if site is None:
            raise ValueError(f"missing human-site measurement for frame={frame} site={label}")
        labels.append(label)
        site_xyz.append(site.xyz_m)

    u_obs = np.asarray([raw_centers[frame].u for frame in frames], dtype=np.float64)
    v_obs = np.asarray([raw_centers[frame].v for frame in frames], dtype=np.float64)
    times = np.asarray([raw_centers[frame].meta.time for frame in frames], dtype=np.float64)
    z_init = np.asarray([depths[frame].depth_m for frame in frames], dtype=np.float64)
    human_event_mask = np.asarray([frame in human_event_frames for frame in frames], dtype=bool)
    floor_event_mask = np.asarray([frame in floor_event_frames for frame in frames], dtype=bool)
    human_state_mask = np.asarray([state_by_frame[frame].human_active for frame in frames], dtype=bool)
    floor_state_mask = np.asarray([state_by_frame[frame].support_active for frame in frames], dtype=bool)
    flight_mask = ~(human_state_mask | floor_state_mask)
    site_xyz_array = np.asarray(site_xyz, dtype=np.float64)
    contact_offset_raw = np.asarray([state_by_frame[frame].contact_depth_offset_m for frame in frames], dtype=np.float64)
    valid_offset = np.isfinite(contact_offset_raw) & (np.abs(contact_offset_raw) <= parameters.maximum_contact_depth_offset_m)
    contact_offset_used = np.zeros_like(contact_offset_raw)
    contact_offset_used[valid_offset] = contact_offset_raw[valid_offset]
    anchor_values = site_xyz_array[:, 2] - contact_offset_used
    global_shift = float(np.median(anchor_values[human_event_mask] - z_init[human_event_mask]))
    z_ref_global = np.maximum(z_init + global_shift, parameters.minimum_depth_m)
    z_ref_segment = _build_anchor_segment_reference(z_init, human_event_mask, anchor_values, parameters.minimum_depth_m)
    z_final = _solve_anchor_interpolation(
        z_ref_segment,
        human_event_mask,
        anchor_values,
        u_obs,
        v_obs,
        times,
        flight_mask,
        profile.camera,
        parameters,
    )
    anchor_indices = np.flatnonzero(human_event_mask)
    first_anchor, last_anchor = int(anchor_indices[0]), int(anchor_indices[-1])
    if first_anchor > 0:
        z_final[:first_anchor] = max(parameters.minimum_depth_m, float(z_final[first_anchor]))
    if last_anchor + 1 < len(z_final):
        z_final[last_anchor + 1 :] = max(parameters.minimum_depth_m, float(z_final[last_anchor]))

    xyz_final = _reconstruct_xyz(u_obs, v_obs, z_final, profile.camera)
    z_contact_final = z_final + contact_offset_used
    sphere = profile.data.get("sphere", {})
    radius_m = float(sphere.get("radius_m", 0.0)) if isinstance(sphere, dict) else 0.0
    if radius_m <= 0.0:
        raise ValueError("sphere.radius_m must be declared in the case profile")

    pose_rows: list[dict[str, object]] = []
    residual_rows: list[dict[str, object]] = []
    for index, frame in enumerate(frames):
        state = state_by_frame[frame]
        label = labels[index]
        if floor_event_mask[index] and not human_event_mask[index]:
            contact_part, contact_side, contact_label = "floor", "", "floor"
        else:
            contact_side, contact_part = label.split("_", 1) if "_" in label else ("", label)
            contact_label = label
        pose_rows.append(
            {
                "frame": frame,
                "time": f"{times[index]:.6f}",
                "tx": f"{xyz_final[index, 0]:.6f}",
                "ty": f"{xyz_final[index, 1]:.6f}",
                "tz": f"{xyz_final[index, 2]:.6f}",
                "qw": "1.000000", "qx": "0.000000", "qy": "0.000000", "qz": "0.000000",
                "radius_m": f"{radius_m:.6f}", "coord_frame": "gvhmr_incam",
                "u_obs": f"{u_obs[index]:.3f}", "v_obs": f"{v_obs[index]:.3f}",
                "radius_obs_px": "", "u_proj": f"{u_obs[index]:.3f}", "v_proj": f"{v_obs[index]:.3f}",
                "radius_proj_px": "", "bottom_proj_v": "", "floor_v": f"{support_floor_v:.3f}",
                "support_type": support_type, "support_source": support_source,
                "support_confidence": f"{support_confidence:.6f}",
                "residual_px": "0.000000", "contact_frame": int(human_event_mask[index]), "audio_contact_frame": 0,
                "human_contact_event": int(human_event_mask[index]), "floor_contact_event": int(floor_event_mask[index]),
                "human_contact_state": int(human_state_mask[index]), "floor_contact_state": int(floor_state_mask[index]),
                "contact_part": contact_part, "contact_side": contact_side, "contact_label": contact_label,
                "active_part": label, "active_part_y": f"{site_xyz_array[index, 1]:.6f}",
                "active_part_z": f"{site_xyz_array[index, 2]:.6f}",
                "u_ref_obs": f"{u_obs[index]:.3f}", "v_ref_obs": f"{v_obs[index]:.3f}",
                "contact_u": f"{state.object_u:.3f}" if state.object_u is not None else "",
                "contact_v": f"{state.object_v:.3f}" if state.object_v is not None else "",
                "contact_depth_offset_m": f"{contact_offset_raw[index]:.6f}",
                "contact_depth_offset_used_m": f"{contact_offset_used[index]:.6f}",
                "z_contact_final": f"{z_contact_final[index]:.6f}",
                "global_z_ref": f"{z_ref_segment[index]:.6f}",
                "z_ref_global_shift": f"{z_ref_global[index]:.6f}",
                "z_ref_anchor_segment": f"{z_ref_segment[index]:.6f}",
                "contact_depth_gap": f"{z_contact_final[index] - site_xyz_array[index, 2]:.6f}",
            }
        )
        residual_rows.append(
            {
                "frame": frame,
                "measurement_depth_m": f"{z_init[index]:.6f}",
                "candidate_depth_m": f"{z_final[index]:.6f}",
                "depth_delta_m": f"{z_final[index] - z_init[index]:.6f}",
                "contact_event_active": int(human_event_mask[index]),
                "support_event_active": int(floor_event_mask[index]),
                "flight_active": int(flight_mask[index]),
                "contact_depth_gap_m": f"{z_contact_final[index] - site_xyz_array[index, 2]:.6f}",
            }
        )

    candidate_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = candidate_dir / SPHERE_CANDIDATE_NAME
    residual_path = candidate_dir / SPHERE_RESIDUAL_NAME
    attempt_path = candidate_dir / SPHERE_ATTEMPT_NAME
    _write_csv(candidate_path, pose_rows, POSE_FIELDS)
    _write_csv(residual_path, residual_rows, list(residual_rows[0]))
    input_records = {
        name: {"path": str(repo_relative_value(path)), "sha256": _sha256(path)}
        for name, path in input_paths.items()
    }
    attempt_core = {
        "sample_id": profile.case_name,
        "state_spec": "translation3:sphere",
        "geometry_provider": "sphere_proxy",
        "parameters": asdict(parameters),
        "inputs": input_records,
        "candidate_sha256": _sha256(candidate_path),
        "residual_sha256": _sha256(residual_path),
    }
    attempt = {
        "schema_version": 1,
        "mode": "generic_sphere_sequence_candidate",
        "attempt_id": f"sphere-{_canonical_hash(attempt_core)[:12]}",
        "solver_executed": True,
        "accepted_outputs_written": False,
        "baseline_pose_read": False,
        "executor_scope": "isolated_candidate_dir",
        "canonical_result_dir": str(repo_relative_value(result_dir)),
        "candidate_dir": str(repo_relative_value(candidate_dir)),
        "inputs": input_records,
        "state_spec": attempt_core["state_spec"],
        "geometry_provider": attempt_core["geometry_provider"],
        "parameters": attempt_core["parameters"],
        "frames": len(frames),
        "human_contact_events": int(np.count_nonzero(human_event_mask)),
        "support_events": int(np.count_nonzero(floor_event_mask)),
        "global_depth_shift_m": global_shift,
        "candidate_artifact": SPHERE_CANDIDATE_NAME,
        "candidate_sha256": attempt_core["candidate_sha256"],
        "residual_artifact": SPHERE_RESIDUAL_NAME,
        "residual_sha256": attempt_core["residual_sha256"],
        "accepted_output_policy": "sandbox_only_until_all_ball_regression_gates_pass",
        "canonical_sha256": _canonical_hash(attempt_core),
    }
    write_json(attempt_path, attempt)
    return attempt
