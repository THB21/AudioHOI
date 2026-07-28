from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


def _axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-12:
        raise ValueError("joint axis must be nonzero")
    unit = axis / axis_norm
    x, y, z = unit
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


@dataclass(frozen=True)
class SegmentJointRule:
    rule_id: str
    joint_id: str
    parts: tuple[str, ...]
    segment_ids: tuple[str, ...]
    origin: np.ndarray
    axis: np.ndarray
    angle_sign: float = 1.0
    endpoint_selector: str = "all"

    def __post_init__(self) -> None:
        if not self.rule_id or not self.joint_id:
            raise ValueError("SegmentJointRule requires rule_id and joint_id")
        if not self.parts and not self.segment_ids:
            raise ValueError("SegmentJointRule requires at least one part or segment id")
        if self.origin.shape != (3,) or self.axis.shape != (3,):
            raise ValueError("SegmentJointRule origin and axis must have shape (3,)")
        if self.endpoint_selector not in {"all", "max_y"}:
            raise ValueError("unsupported endpoint selector")

    def matches(self, segment_id: str, part: str) -> bool:
        return part in self.parts or segment_id in self.segment_ids or any(token in segment_id for token in self.segment_ids)


@dataclass(frozen=True)
class ArticulatedKinematicProvider:
    rules: tuple[SegmentJointRule, ...]

    def articulate_segment(
        self,
        segment_id: str,
        part: str,
        points: np.ndarray,
        joint_values: dict[str, float],
    ) -> np.ndarray:
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("segment points must have shape (N, 3)")
        out = points.copy()
        for rule in self.rules:
            if not rule.matches(segment_id, part):
                continue
            angle = float(joint_values.get(rule.joint_id, 0.0)) * float(rule.angle_sign)
            rotated = self._rotate(out, rule.origin, rule.axis, angle)
            if rule.endpoint_selector == "all":
                out = rotated
            elif rule.endpoint_selector == "max_y":
                index = int(np.argmax(out[:, 1]))
                out[index : index + 1] = rotated[index : index + 1]
        return out

    @staticmethod
    def _rotate(points: np.ndarray, origin: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
        rot = _axis_angle_matrix(axis, angle)
        return (points - origin) @ rot.T + origin
