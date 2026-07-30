from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin
from typing import Sequence

import numpy as np

from ..state.types import DofKind, DofSpec, StateSpec


def _quaternion_indices(dof: DofSpec) -> tuple[int, int, int, int]:
    names = tuple(field.rsplit(".", 1)[-1] for field in dof.source_fields)
    if len(names) != 4 or set(names) != {"qx", "qy", "qz", "qw"}:
        raise ValueError(f"rotation dof {dof.dof_id} must identify qx/qy/qz/qw source fields")
    return tuple(names.index(name) for name in ("qx", "qy", "qz", "qw"))


def _normalize_quaternion_xyzw(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("rotation state requires a finite nonzero quaternion")
    return value / norm


def _rotvec_quaternion_xyzw(rotvec: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rotvec))
    if angle <= 1e-12:
        scale = 0.5 - angle * angle / 48.0
        return _normalize_quaternion_xyzw(np.array((*(scale * rotvec), 1.0), dtype=float))
    half = 0.5 * angle
    return np.array((*(sin(half) * rotvec / angle), cos(half)), dtype=float)


def _multiply_quaternion_xyzw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return _normalize_quaternion_xyzw(
        np.array(
            (
                lw * rx + lx * rw + ly * rz - lz * ry,
                lw * ry - lx * rz + ly * rw + lz * rx,
                lw * rz + lx * ry - ly * rx + lz * rw,
                lw * rw - lx * rx - ly * ry - lz * rz,
            ),
            dtype=float,
        )
    )


def _wrap_periodic(value: float, lower: float, upper: float) -> float:
    period = upper - lower
    if period <= 0.0:
        raise ValueError("periodic bounds must have positive width")
    return float((value - lower) % period + lower)


@dataclass(frozen=True)
class _DofLayout:
    dof: DofSpec
    state_start: int
    state_stop: int
    parameter_start: int
    parameter_stop: int


@dataclass(frozen=True)
class StateSpecParameterization:
    state_spec: StateSpec
    layouts: tuple[_DofLayout, ...]
    state_width: int
    parameter_width_per_frame: int

    @classmethod
    def from_state_spec(cls, state_spec: StateSpec) -> StateSpecParameterization:
        layouts: list[_DofLayout] = []
        state_offset = 0
        parameter_offset = 0
        for dof in state_spec.dofs:
            if not dof.observable:
                if dof.kind == DofKind.ROTATION_SO3:
                    _quaternion_indices(dof)
                parameter_dimension = 0
            elif dof.kind == DofKind.ROTATION_SO3:
                _quaternion_indices(dof)
                parameter_dimension = 3
            else:
                parameter_dimension = dof.dimension
            layouts.append(
                _DofLayout(
                    dof=dof,
                    state_start=state_offset,
                    state_stop=state_offset + dof.dimension,
                    parameter_start=parameter_offset,
                    parameter_stop=parameter_offset + parameter_dimension,
                )
            )
            state_offset += dof.dimension
            parameter_offset += parameter_dimension
        if parameter_offset <= 0:
            raise ValueError("StateSpec must expose at least one optimizable degree of freedom")
        return cls(state_spec, tuple(layouts), state_offset, parameter_offset)

    def _initial_matrix(self, initial_states: Sequence[Sequence[float]]) -> np.ndarray:
        initial = np.asarray(initial_states, dtype=float)
        if initial.ndim != 2 or initial.shape[1] != self.state_width or not np.isfinite(initial).all():
            raise ValueError("initial states must be a finite matrix matching StateSpec width")
        for layout in self.layouts:
            if layout.dof.kind != DofKind.ROTATION_SO3:
                continue
            indices = _quaternion_indices(layout.dof)
            for row in initial:
                _normalize_quaternion_xyzw(row[layout.state_start : layout.state_stop][list(indices)])
        return initial

    def _raw_initial_parameters(self, initial_states: Sequence[Sequence[float]]) -> np.ndarray:
        initial = self._initial_matrix(initial_states)
        parameters = np.zeros((initial.shape[0], self.parameter_width_per_frame), dtype=float)
        for layout in self.layouts:
            if layout.parameter_start == layout.parameter_stop:
                continue
            if layout.dof.kind == DofKind.ROTATION_SO3:
                continue
            parameters[:, layout.parameter_start : layout.parameter_stop] = initial[
                :, layout.state_start : layout.state_stop
            ]
        return parameters.reshape(-1)

    def initial_parameters(self, initial_states: Sequence[Sequence[float]]) -> np.ndarray:
        raw = self._raw_initial_parameters(initial_states)
        lower, upper = self.parameter_bounds(len(initial_states))
        return np.clip(raw, lower, upper)

    def initial_bound_projection_count(self, initial_states: Sequence[Sequence[float]]) -> int:
        raw = self._raw_initial_parameters(initial_states)
        projected = self.initial_parameters(initial_states)
        return int(np.count_nonzero(raw != projected))

    def parameter_bounds(self, frame_count: int) -> tuple[np.ndarray, np.ndarray]:
        lower = np.full((frame_count, self.parameter_width_per_frame), -np.inf, dtype=float)
        upper = np.full((frame_count, self.parameter_width_per_frame), np.inf, dtype=float)
        for layout in self.layouts:
            bound = layout.dof.bound
            if bound is None or layout.dof.kind in {DofKind.ROTATION_SO3, DofKind.PERIODIC}:
                continue
            width = layout.parameter_stop - layout.parameter_start
            lower_values = bound.lower if isinstance(bound.lower, tuple) else (bound.lower,) * width
            upper_values = bound.upper if isinstance(bound.upper, tuple) else (bound.upper,) * width
            for component, value in enumerate(lower_values):
                if value is not None:
                    lower[:, layout.parameter_start + component] = value
            for component, value in enumerate(upper_values):
                if value is not None:
                    upper[:, layout.parameter_start + component] = value
        return lower.reshape(-1), upper.reshape(-1)

    def decode(
        self,
        parameters: Sequence[float],
        initial_states: Sequence[Sequence[float]],
    ) -> np.ndarray:
        initial = self._initial_matrix(initial_states)
        parameter_matrix = np.asarray(parameters, dtype=float).reshape(
            initial.shape[0], self.parameter_width_per_frame
        )
        if not np.isfinite(parameter_matrix).all():
            raise ValueError("state parameters must be finite")
        decoded = initial.copy()
        for layout in self.layouts:
            dof = layout.dof
            state_slice = slice(layout.state_start, layout.state_stop)
            parameter_slice = slice(layout.parameter_start, layout.parameter_stop)
            if dof.kind == DofKind.ROTATION_SO3:
                indices = _quaternion_indices(dof)
                for index, row in enumerate(initial):
                    base = _normalize_quaternion_xyzw(row[state_slice][list(indices)])
                    values = decoded[index, state_slice]
                    for component, source_index in enumerate(indices):
                        values[source_index] = base[component]
                    if not dof.observable:
                        continue
                    delta = _rotvec_quaternion_xyzw(parameter_matrix[index, parameter_slice])
                    rotated = _multiply_quaternion_xyzw(delta, base)
                    for component, source_index in enumerate(indices):
                        values[source_index] = rotated[component]
                continue
            if layout.parameter_start == layout.parameter_stop:
                continue
            values = parameter_matrix[:, parameter_slice]
            if dof.kind == DofKind.PERIODIC:
                if (
                    dof.bound is None
                    or not isinstance(dof.bound.lower, (float, int))
                    or not isinstance(dof.bound.upper, (float, int))
                ):
                    raise ValueError(f"periodic dof {dof.dof_id} requires finite wrapping bounds")
                values = np.vectorize(
                    lambda value: _wrap_periodic(value, dof.bound.lower, dof.bound.upper),
                    otypes=[float],
                )(values)
            decoded[:, state_slice] = values
        return decoded
