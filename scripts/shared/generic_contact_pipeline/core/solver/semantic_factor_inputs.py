"""Typed, object-agnostic inputs for semantic and interval-audio factors."""
from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
from typing import Mapping, Sequence

import numpy as np


_TURN_LABELS = {"counterclockwise": 1.0, "clockwise": -1.0}
_AUDIO_MODES = {
    "silence": 0.0,
    "sustained_motion": 1.0,
    "short_tug": 1.0,
    "motion_onset": 2.0,
    "motion_offset": 2.0,
    "seam_click": 3.0,
}


def _unit(values: Sequence[float], field: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all() or np.linalg.norm(vector) <= 1e-8:
        raise ValueError(f"{field} must be a finite nonzero 3-vector")
    return vector / np.linalg.norm(vector)


def _rotate_wxyz(quaternion: Sequence[float], vector: Sequence[float]) -> np.ndarray:
    q = np.asarray(quaternion, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) <= 1e-8:
        raise ValueError("semantic factors require a finite root quaternion")
    q /= np.linalg.norm(q)
    w, x, y, z = q
    rotation = np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=float,
    )
    return rotation @ np.asarray(vector, dtype=float)


def _relative_rotvec(previous: Sequence[float], current: Sequence[float]) -> np.ndarray:
    a = np.asarray(previous[3:7], dtype=float)
    b = np.asarray(current[3:7], dtype=float)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    inverse = a * np.asarray((1.0, -1.0, -1.0, -1.0))
    aw, ax, ay, az = b
    bw, bx, by, bz = inverse
    q = np.asarray(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        )
    )
    q /= np.linalg.norm(q)
    if q[0] < 0:
        q *= -1
    norm = float(np.linalg.norm(q[1:]))
    if norm <= 1e-12:
        return np.zeros(3)
    return 2.0 * np.arctan2(norm, float(q[0])) * q[1:] / norm


@dataclass(frozen=True)
class FaceVisibilityFactorInput:
    selected_face_normal_local_by_frame: Mapping[int, tuple[float, float, float]]
    incompatible_face_normals_local_by_frame: Mapping[int, tuple[tuple[float, float, float], ...]]
    camera_center_world_by_frame: Mapping[int, tuple[float, float, float]]
    active_frames: tuple[int, ...]
    confidence_by_frame: Mapping[int, float]
    evidence_ids_by_frame: Mapping[int, tuple[str, ...]]
    margin: float = 0.05
    weight: float = 1.0
    sigma: float = 1.0

    def __post_init__(self) -> None:
        _validate_common(self.active_frames, self.confidence_by_frame, self.evidence_ids_by_frame, self.weight, self.sigma)
        for frame in self.active_frames:
            _unit(self.selected_face_normal_local_by_frame[frame], "selected face normal")
            incompatible = self.incompatible_face_normals_local_by_frame.get(frame, ())
            if not incompatible:
                raise ValueError("face visibility requires incompatible face normals")
            for normal in incompatible:
                _unit(normal, "incompatible face normal")


@dataclass(frozen=True)
class FacingRelationFactorInput:
    local_facing_axis: tuple[float, float, float]
    human_reference_by_frame: Mapping[int, tuple[float, float, float]]
    active_frames: tuple[int, ...]
    target_label_by_frame: Mapping[int, str]
    confidence_by_frame: Mapping[int, float]
    evidence_ids_by_frame: Mapping[int, tuple[str, ...]]
    support_normal_world: tuple[float, float, float] = (0.0, 1.0, 0.0)
    margin: float = 0.1
    weight: float = 1.0
    sigma: float = 1.0

    def __post_init__(self) -> None:
        _unit(self.local_facing_axis, "local facing axis")
        _unit(self.support_normal_world, "support normal")
        _validate_common(self.active_frames, self.confidence_by_frame, self.evidence_ids_by_frame, self.weight, self.sigma)
        if any(
            self.target_label_by_frame.get(frame)
            not in {"grasp_side_toward_human", "grasp_side_away", "side_on"}
            for frame in self.active_frames
        ):
            raise ValueError("unsupported facing relation label")


@dataclass(frozen=True)
class HeadingTopologyInterval:
    start_frame: int
    end_frame: int
    label: str
    confidence: float
    evidence_ids: tuple[str, ...]
    geometry_consistent: bool = True

    def __post_init__(self) -> None:
        if self.start_frame < 1 or self.end_frame < self.start_frame:
            raise ValueError("heading topology interval bounds are invalid")
        if self.label not in {*_TURN_LABELS, "stationary", "unclear"}:
            raise ValueError("unsupported heading topology label")
        if not 0.0 <= self.confidence <= 1.0 or not self.evidence_ids:
            raise ValueError("heading topology requires confidence and evidence IDs")


@dataclass(frozen=True)
class HeadingTopologyFactorInput:
    intervals: tuple[HeadingTopologyInterval, ...]
    support_normal_world: tuple[float, float, float] = (0.0, 1.0, 0.0)
    minimum_increment_rad: float = 0.002
    weight: float = 1.0
    sigma_rad: float = 1.0

    def __post_init__(self) -> None:
        _unit(self.support_normal_world, "support normal")
        if self.minimum_increment_rad < 0 or self.weight < 0 or self.sigma_rad <= 0:
            raise ValueError("invalid heading topology factor scale")


@dataclass(frozen=True)
class AudioMotionInterval:
    start_frame: int
    end_frame: int
    event_type: str
    confidence: float
    evidence_id: str
    visual_speed_is_low: bool = False

    def __post_init__(self) -> None:
        if self.start_frame < 1 or self.end_frame < self.start_frame:
            raise ValueError("audio motion interval bounds are invalid")
        if self.event_type not in _AUDIO_MODES:
            raise ValueError("unsupported audio motion event")
        if not 0.0 <= self.confidence <= 1.0 or not self.evidence_id:
            raise ValueError("audio motion interval requires confidence and evidence ID")


@dataclass(frozen=True)
class AudioMotionEnvelopeFactorInput:
    intervals: tuple[AudioMotionInterval, ...]
    support_normal_world: tuple[float, float, float] = (0.0, 1.0, 0.0)
    minimum_motion_m_per_frame: float = 0.002
    weight: float = 1.0
    sigma_m_per_frame: float = 1.0

    def __post_init__(self) -> None:
        _unit(self.support_normal_world, "support normal")
        if self.minimum_motion_m_per_frame < 0 or self.weight < 0 or self.sigma_m_per_frame <= 0:
            raise ValueError("invalid audio motion factor scale")


def arbitrate_heading_topology(
    intervals: Sequence[HeadingTopologyInterval],
    states: Mapping[int, Sequence[float]],
    *,
    support_normal_world: Sequence[float],
    reliable_frames: Sequence[int],
    minimum_observable_increment_rad: float = 0.01,
) -> tuple[HeadingTopologyInterval, ...]:
    """Reject a VLM turn label only when reliable geometry disproves its sign.

    The function never changes a label or writes a pose.  Sparse reliable
    rail/wheel/face frames merely provide a sign-consistency veto; an
    unobservable interval remains available as semantic evidence.
    """

    normal = _unit(support_normal_world, "support normal")
    reliable = {int(frame) for frame in reliable_frames}
    output: list[HeadingTopologyInterval] = []
    for interval in intervals:
        expected = _TURN_LABELS.get(interval.label)
        measured: list[float] = []
        if expected is not None:
            for frame in range(interval.start_frame + 1, interval.end_frame + 1):
                if frame - 1 not in reliable or frame not in reliable:
                    continue
                if frame - 1 not in states or frame not in states:
                    continue
                increment = float(_relative_rotvec(states[frame - 1], states[frame]) @ normal)
                if abs(increment) >= minimum_observable_increment_rad:
                    measured.append(increment)
        consistent = interval.geometry_consistent
        if expected is not None and measured:
            consistent = consistent and float(np.median(measured)) * expected > 0.0
        output.append(replace(interval, geometry_consistent=consistent))
    return tuple(output)


def _validate_common(
    frames: tuple[int, ...],
    confidence: Mapping[int, float],
    evidence: Mapping[int, tuple[str, ...]],
    weight: float,
    sigma: float,
) -> None:
    if not frames or tuple(sorted(set(frames))) != frames:
        raise ValueError("semantic active frames must be sorted and unique")
    if weight < 0 or sigma <= 0 or not isfinite(weight) or not isfinite(sigma):
        raise ValueError("semantic factor scales are invalid")
    for frame in frames:
        if not 0.0 <= float(confidence.get(frame, -1.0)) <= 1.0 or not evidence.get(frame):
            raise ValueError("each semantic frame requires confidence and evidence IDs")


def build_face_visibility_inputs(
    states: Mapping[int, Sequence[float]], factor: FaceVisibilityFactorInput, weight_by_frame: Mapping[int, float] | None = None
) -> dict[str, object]:
    selected, incompatible, weights = [], [], []
    for frame in factor.active_frames:
        if frame not in states or frame not in factor.camera_center_world_by_frame:
            continue
        view = _unit(
            np.asarray(factor.camera_center_world_by_frame[frame], dtype=float)
            - np.asarray(states[frame][:3], dtype=float),
            "camera view direction",
        )
        selected_local = _unit(factor.selected_face_normal_local_by_frame[frame], "selected face normal")
        incompatible_local = tuple(
            _unit(n, "incompatible face normal")
            for n in factor.incompatible_face_normals_local_by_frame[frame]
        )
        selected.append(float(_rotate_wxyz(states[frame][3:7], selected_local) @ view))
        incompatible.append(max(float(_rotate_wxyz(states[frame][3:7], n) @ view) for n in incompatible_local))
        weights.append(float((weight_by_frame or {}).get(frame, factor.weight)) * float(factor.confidence_by_frame[frame]))
    return {"selected_rank": selected, "incompatible_rank": incompatible, "weight": weights, "margin": factor.margin, "sigma": factor.sigma}


def build_facing_relation_inputs(
    states: Mapping[int, Sequence[float]], factor: FacingRelationFactorInput, weight_by_frame: Mapping[int, float] | None = None
) -> dict[str, object]:
    agreements, weights = [], []
    normal = _unit(factor.support_normal_world, "support normal")
    local_axis = _unit(factor.local_facing_axis, "local facing axis")
    for frame in factor.active_frames:
        if frame not in states or frame not in factor.human_reference_by_frame:
            continue
        state = np.asarray(states[frame], dtype=float)
        predicted = _rotate_wxyz(state[3:7], local_axis)
        desired = np.asarray(factor.human_reference_by_frame[frame], dtype=float) - state[:3]
        predicted -= normal * float(predicted @ normal)
        desired -= normal * float(desired @ normal)
        if np.linalg.norm(predicted) <= 1e-8 or np.linalg.norm(desired) <= 1e-8:
            continue
        alignment = float((predicted / np.linalg.norm(predicted)) @ (desired / np.linalg.norm(desired)))
        target_label = factor.target_label_by_frame[frame]
        agreements.append(-alignment if target_label == "grasp_side_away" else (1.0 - abs(alignment) if target_label == "side_on" else alignment))
        weights.append(float((weight_by_frame or {}).get(frame, factor.weight)) * float(factor.confidence_by_frame[frame]))
    return {"agreement": agreements, "weight": weights, "margin": factor.margin, "sigma": factor.sigma}


def build_heading_topology_inputs(
    states: Mapping[int, Sequence[float]],
    factor: HeadingTopologyFactorInput,
    weight_by_frame: Mapping[int, float] | None = None,
) -> dict[str, object]:
    normal = _unit(factor.support_normal_world, "support normal")
    increments, signs, weights = [], [], []
    for interval in factor.intervals:
        if interval.label not in _TURN_LABELS or not interval.geometry_consistent:
            continue
        sign = _TURN_LABELS[interval.label]
        for frame in range(interval.start_frame + 1, interval.end_frame + 1):
            if frame - 1 not in states or frame not in states:
                continue
            increments.append(float(_relative_rotvec(states[frame - 1], states[frame]) @ normal))
            signs.append(sign)
            weights.append(float((weight_by_frame or {}).get(frame, factor.weight)) * interval.confidence)
    return {"signed_increment_rad": increments, "target_sign": signs, "weight": weights, "minimum_increment_rad": factor.minimum_increment_rad, "sigma_rad": factor.sigma_rad}


def build_audio_motion_inputs(
    states: Mapping[int, Sequence[float]],
    factor: AudioMotionEnvelopeFactorInput,
    weight_by_frame: Mapping[int, float] | None = None,
) -> dict[str, object]:
    normal = _unit(factor.support_normal_world, "support normal")
    speeds, modes, weights = [], [], []
    for interval in factor.intervals:
        if interval.event_type == "silence" and not interval.visual_speed_is_low:
            continue
        for frame in range(interval.start_frame + 1, interval.end_frame + 1):
            if frame - 1 not in states or frame not in states:
                continue
            delta = np.asarray(states[frame][:3], dtype=float) - np.asarray(states[frame - 1][:3], dtype=float)
            tangent = delta - normal * float(delta @ normal)
            speeds.append(float(np.linalg.norm(tangent)))
            modes.append(_AUDIO_MODES[interval.event_type])
            weights.append(float((weight_by_frame or {}).get(frame, factor.weight)) * interval.confidence)
    return {"tangential_speed_m_per_frame": speeds, "event_mode": modes, "weight": weights, "minimum_motion_m_per_frame": factor.minimum_motion_m_per_frame, "sigma_m_per_frame": factor.sigma_m_per_frame}
