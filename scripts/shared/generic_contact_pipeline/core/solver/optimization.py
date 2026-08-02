from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .factor_residuals import FactorResidualEvaluator
from .parameterization import StateSpecParameterization
from .residual_inputs import ContactFacingFactorInput
from .sparsity import ResidualRowDependency, build_factor_frame_jacobian_sparsity


StateMapping = Mapping[int, Sequence[float]]
ResidualInputBuilder = Callable[[StateMapping], dict[str, dict[str, Any]]]


@dataclass(frozen=True)
class SequenceOptimizationProblem:
    attempt_id: str
    sequence_contract_sha256: str
    frames: tuple[int, ...]
    initial_states: tuple[tuple[float, ...], ...]
    factor_ids: tuple[str, ...]
    residual_execution_plan: dict[str, object] | object
    residual_input_builder: ResidualInputBuilder
    state_parameterization: StateSpecParameterization | None = None
    residual_dependencies: tuple[ResidualRowDependency, ...] = ()
    parent_solve_attempt_id: str | None = None
    lower_bounds: tuple[tuple[float, ...], ...] | None = None
    upper_bounds: tuple[tuple[float, ...], ...] | None = None
    initialization_kind: str | None = None
    initialization_ledger_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.attempt_id.startswith("generic-attempt-") or not self.sequence_contract_sha256:
            raise ValueError("sequence optimization must bind a generic attempt and sequence contract")
        if not self.frames or len(self.frames) != len(self.initial_states):
            raise ValueError("sequence optimization requires frame-aligned initial states")
        if len(set(self.frames)) != len(self.frames) or tuple(sorted(self.frames)) != self.frames:
            raise ValueError("sequence optimization frames must be unique and sorted")
        widths = {len(state) for state in self.initial_states}
        if len(widths) != 1 or not next(iter(widths)):
            raise ValueError("sequence optimization states must have one nonzero width")
        if not self.factor_ids or len(set(self.factor_ids)) != len(self.factor_ids):
            raise ValueError("sequence optimization requires unique factor ids")
        if (self.lower_bounds is None) != (self.upper_bounds is None):
            raise ValueError("sequence optimization bounds must provide both lower and upper values")
        if (self.initialization_kind is None) != (self.initialization_ledger_sha256 is None):
            raise ValueError("sequence optimization initialization provenance requires kind and ledger hash")
        if self.initialization_kind is not None:
            if not self.initialization_kind:
                raise ValueError("sequence optimization initialization kind must be nonempty")
            digest = str(self.initialization_ledger_sha256)
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise ValueError("sequence optimization initialization ledger must be a lowercase SHA-256")
        if self.state_parameterization is not None:
            if self.state_parameterization.state_width != next(iter(widths)):
                raise ValueError("StateSpec parameterization width must match sequence states")
            if self.lower_bounds is not None:
                raise ValueError("explicit state bounds cannot be combined with StateSpec parameterization")
        selected_factors = set(self.factor_ids)
        if any(dependency.factor_id not in selected_factors for dependency in self.residual_dependencies):
            raise ValueError("residual dependencies must reference selected factors")
        if any(frame not in set(self.frames) for dependency in self.residual_dependencies for frame in dependency.frames):
            raise ValueError("residual dependencies must reference sequence frames")
        if self.parent_solve_attempt_id is not None and not self.parent_solve_attempt_id.startswith("generic-solve-"):
            raise ValueError("parent solve attempt must use a generic-solve id")
        for bounds in (self.lower_bounds, self.upper_bounds):
            if bounds is not None and (len(bounds) != len(self.frames) or any(len(row) not in widths for row in bounds)):
                raise ValueError("sequence optimization bounds must match state shape")
        if self.lower_bounds is not None:
            initial = np.asarray(self.initial_states, dtype=float)
            lower = np.asarray(self.lower_bounds, dtype=float)
            upper = np.asarray(self.upper_bounds, dtype=float)
            if np.any(lower >= upper) or np.any(initial < lower) or np.any(initial > upper):
                raise ValueError("sequence optimization bounds must contain initial states and have positive width")


@dataclass(frozen=True)
class SequenceOptimizationParameters:
    robust_loss: str = "soft_l1"
    robust_scale: float = 1.0
    max_function_evaluations: int = 200

    def __post_init__(self) -> None:
        if not self.robust_loss or self.robust_scale <= 0.0 or self.max_function_evaluations <= 0:
            raise ValueError("sequence optimization parameters must be positive and name a robust loss")


@dataclass(frozen=True)
class GenericSequenceSolveResult:
    attempt_id: str
    solve_attempt_id: str
    parent_solve_attempt_id: str | None
    sequence_contract_sha256: str
    frames: tuple[int, ...]
    states: tuple[tuple[float, ...], ...]
    factor_ids: tuple[str, ...]
    residual_program_sha256: str
    state_spec_id: str | None
    parameterization: str
    state_dimension: int
    parameter_dimension: int
    initial_bound_projection_count: int
    jacobian_sparsity_used: bool
    jacobian_nonzero_count: int
    jacobian_density: float
    initial_residual_count: int
    final_residual_count: int
    initial_squared_error: float
    final_squared_error: float
    function_evaluations: int
    success: bool
    message: str
    canonical_sha256: str
    termination_status: int = 0
    optimality: float = float("inf")
    active_bound_count: int = 0
    case_dispatch_used: bool = False
    residuals_executed: bool = True
    solver_executed: bool = True
    accepted_outputs_written: bool = False


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _jsonable(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _plan_records(plan: dict[str, object] | object) -> list[dict[str, object]]:
    if isinstance(plan, dict):
        return [record for record in plan.get("records", []) if isinstance(record, dict)]
    return [
        {
            "factor_id": record.factor_id,
            "evaluator_ref": record.evaluator_ref,
            "status": record.status,
        }
        for record in getattr(plan, "records", ())
    ]


def _array_kwargs(raw_kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        key: np.asarray(value, dtype=float) if isinstance(value, list) else value
        for key, value in raw_kwargs.items()
    }


def build_runtime_residual_vector(
    residual_execution_plan: dict[str, object] | object,
    residual_inputs: dict[str, dict[str, Any]],
    factor_ids: Sequence[str],
) -> np.ndarray:
    blocks = build_runtime_residual_blocks(residual_execution_plan, residual_inputs, factor_ids)
    result = np.concatenate([values for _, values in blocks])
    if not np.isfinite(result).all():
        raise ValueError("sequence optimization residuals must be finite")
    return result


def build_runtime_residual_blocks(
    residual_execution_plan: dict[str, object] | object,
    residual_inputs: dict[str, dict[str, Any]],
    factor_ids: Sequence[str],
) -> tuple[tuple[str, np.ndarray], ...]:
    selected = set(factor_ids)
    evaluator = FactorResidualEvaluator()
    residuals: list[tuple[str, np.ndarray]] = []
    executed: set[str] = set()
    for record in _plan_records(residual_execution_plan):
        factor_id = str(record.get("factor_id", ""))
        if factor_id not in selected:
            continue
        if str(record.get("status", "")) != "ready_not_executed":
            raise ValueError(f"selected factor is not executable: {factor_id}")
        raw_kwargs = residual_inputs.get(factor_id)
        if raw_kwargs is None:
            raise ValueError(f"missing runtime residual inputs for selected factor: {factor_id}")
        evaluator_ref = str(record.get("evaluator_ref", ""))
        method_name = evaluator_ref.removeprefix("FactorResidualEvaluator.")
        method = getattr(evaluator, method_name)
        values = np.asarray(method(**_array_kwargs(raw_kwargs)), dtype=float).reshape(-1)
        if not values.size or not np.isfinite(values).all():
            raise ValueError(f"runtime residual block must be finite and nonempty: {factor_id}")
        residuals.append((factor_id, values))
        executed.add(factor_id)
    missing = selected - executed
    if missing:
        raise ValueError("selected factors are absent from residual execution plan: " + ",".join(sorted(missing)))
    if not residuals:
        raise ValueError("sequence optimization produced no residual blocks")
    return tuple(residuals)


def solve_sequence_optimization(
    problem: SequenceOptimizationProblem,
    parameters: SequenceOptimizationParameters = SequenceOptimizationParameters(),
) -> GenericSequenceSolveResult:
    try:
        from scipy.optimize import least_squares
    except ImportError as exc:  # pragma: no cover - depends on declared runtime
        raise RuntimeError("generic sequence solve requires scipy") from exc

    state_shape = (len(problem.frames), len(problem.initial_states[0]))
    initial = np.asarray(problem.initial_states, dtype=float)
    if not np.isfinite(initial).all():
        raise ValueError("sequence optimization initial states must be finite")

    def states_from_vector(vector: np.ndarray) -> dict[int, tuple[float, ...]]:
        if problem.state_parameterization is None:
            matrix = np.asarray(vector, dtype=float).reshape(state_shape)
        else:
            matrix = problem.state_parameterization.decode(vector, problem.initial_states)
        return {
            frame: tuple(float(value) for value in matrix[index])
            for index, frame in enumerate(problem.frames)
        }

    def residual(vector: np.ndarray) -> np.ndarray:
        states = states_from_vector(vector)
        inputs = problem.residual_input_builder(states)
        return build_runtime_residual_vector(
            problem.residual_execution_plan,
            inputs,
            problem.factor_ids,
        )

    if problem.state_parameterization is None:
        x0 = initial.reshape(-1)
    else:
        x0 = problem.state_parameterization.initial_parameters(problem.initial_states)
    initial_states = states_from_vector(x0)
    initial_inputs = problem.residual_input_builder(initial_states)
    initial_blocks = build_runtime_residual_blocks(
        problem.residual_execution_plan,
        initial_inputs,
        problem.factor_ids,
    )
    residual_program_sha256 = _canonical_hash(_jsonable(initial_inputs))
    initial_residual = np.concatenate([values for _, values in initial_blocks])
    jacobian_sparsity = None
    if problem.residual_dependencies:
        parameter_width_per_frame = int(x0.size // len(problem.frames))
        jacobian_sparsity = build_factor_frame_jacobian_sparsity(
            factor_block_sizes=tuple((factor_id, int(values.size)) for factor_id, values in initial_blocks),
            frames=problem.frames,
            parameter_width_per_frame=parameter_width_per_frame,
            dependencies=problem.residual_dependencies,
        )
    solve_attempt_payload = {
        "contract_attempt_id": problem.attempt_id,
        "parent_solve_attempt_id": problem.parent_solve_attempt_id,
        "sequence_contract_sha256": problem.sequence_contract_sha256,
        "frames": problem.frames,
        "initial_states_sha256": _canonical_hash(problem.initial_states),
        "factor_ids": problem.factor_ids,
        "residual_program_sha256": residual_program_sha256,
        "state_spec_id": (
            problem.state_parameterization.state_spec.spec_id
            if problem.state_parameterization is not None
            else None
        ),
        "residual_dependencies": [asdict(dependency) for dependency in problem.residual_dependencies],
        "parameters": asdict(parameters),
    }
    if problem.initialization_kind is not None:
        solve_attempt_payload["initialization_kind"] = problem.initialization_kind
        solve_attempt_payload["initialization_ledger_sha256"] = problem.initialization_ledger_sha256
    solve_attempt_id = f"generic-solve-{_canonical_hash(solve_attempt_payload)[:12]}"
    if problem.state_parameterization is not None:
        bounds = problem.state_parameterization.parameter_bounds(len(problem.frames))
    elif problem.lower_bounds is None:
        bounds = (-np.inf, np.inf)
    else:
        bounds = (
            np.asarray(problem.lower_bounds, dtype=float).reshape(-1),
            np.asarray(problem.upper_bounds, dtype=float).reshape(-1),
        )
    result = least_squares(
        residual,
        x0=x0,
        bounds=bounds,
        method="trf",
        loss=parameters.robust_loss,
        f_scale=float(parameters.robust_scale),
        max_nfev=int(parameters.max_function_evaluations),
        jac_sparsity=jacobian_sparsity,
    )
    final_residual = residual(result.x)
    solved_states = states_from_vector(result.x)
    payload = {
        "attempt_id": problem.attempt_id,
        "solve_attempt_id": solve_attempt_id,
        "parent_solve_attempt_id": problem.parent_solve_attempt_id,
        "sequence_contract_sha256": problem.sequence_contract_sha256,
        "frames": problem.frames,
        "states": tuple(solved_states[frame] for frame in problem.frames),
        "factor_ids": problem.factor_ids,
        "residual_program_sha256": residual_program_sha256,
        "state_spec_id": (
            problem.state_parameterization.state_spec.spec_id
            if problem.state_parameterization is not None
            else None
        ),
        "parameterization": (
            "state_spec_tangent" if problem.state_parameterization is not None else "direct_state_matrix"
        ),
        "state_dimension": int(initial.size),
        "parameter_dimension": int(x0.size),
        "initial_bound_projection_count": (
            problem.state_parameterization.initial_bound_projection_count(problem.initial_states)
            if problem.state_parameterization is not None
            else 0
        ),
        "jacobian_sparsity_used": jacobian_sparsity is not None,
        "jacobian_nonzero_count": int(jacobian_sparsity.nnz) if jacobian_sparsity is not None else 0,
        "jacobian_density": (
            float(jacobian_sparsity.nnz / (jacobian_sparsity.shape[0] * jacobian_sparsity.shape[1]))
            if jacobian_sparsity is not None
            else 1.0
        ),
        "initial_residual_count": int(initial_residual.size),
        "final_residual_count": int(final_residual.size),
        "initial_squared_error": float(np.dot(initial_residual, initial_residual)),
        "final_squared_error": float(np.dot(final_residual, final_residual)),
        "function_evaluations": int(result.nfev),
        "termination_status": int(result.status),
        "optimality": float(result.optimality),
        "active_bound_count": int(np.count_nonzero(result.active_mask)),
        "success": bool(result.success),
        "message": str(result.message),
        "case_dispatch_used": False,
        "residuals_executed": True,
        "solver_executed": True,
        "accepted_outputs_written": False,
    }
    return GenericSequenceSolveResult(
        **payload,
        canonical_sha256=_canonical_hash(payload),
    )


def project_bounded_gap_states(
    problem: SequenceOptimizationProblem,
    result: GenericSequenceSolveResult,
    gap_frames: Sequence[int],
    *,
    rotation_recovery_frames: int = 0,
    project_rotation: bool = True,
) -> GenericSequenceSolveResult:
    """Remove bounded occlusion spikes using solved boundary states.

    This is deliberately a post-solve projection: unlike an initializer prior,
    its endpoints are the optimized visible states. Non-rotational state uses
    linear interpolation. Quaternion DOFs hold the last trusted orientation
    through the unsupported part of a gap, then use shortest-path normalized
    interpolation across the end of the gap and a stable visible recovery
    window. This avoids leaking a future turn backwards through a long
    occlusion. No case/object identity participates in the policy.
    """

    selected = sorted(set(int(frame) for frame in gap_frames if int(frame) in result.frames))
    if not selected:
        return result
    states = {
        frame: np.asarray(state, dtype=float).copy()
        for frame, state in zip(result.frames, result.states)
    }
    quaternion_groups: list[tuple[int, int, int, int]] = []
    if problem.state_parameterization is not None:
        offset = 0
        for dof in problem.state_parameterization.state_spec.dofs:
            if dof.kind.value == "rotation_so3":
                names = tuple(field.rsplit(".", 1)[-1] for field in dof.source_fields)
                quaternion_groups.append(
                    tuple(offset + names.index(name) for name in ("qw", "qx", "qy", "qz"))
                )
            offset += dof.dimension
    # Quaternion coordinates are never ordinary linear state dimensions.  If
    # rotation projection is disabled, preserve the solved quaternion instead
    # of accidentally interpolating its four scalar components.  If enabled,
    # the dedicated normalized shortest-path interpolation below owns them.
    quaternion_indices = {index for group in quaternion_groups for index in group}
    projected_quaternion_groups = quaternion_groups if project_rotation else []
    runs: list[tuple[int, int]] = []
    start = previous = selected[0]
    for frame in selected[1:]:
        if frame != previous + 1:
            runs.append((start, previous))
            start = frame
        previous = frame
    runs.append((start, previous))
    projected_frames: list[int] = []
    for start, end in runs:
        left, right = start - 1, end + 1
        if left not in states or right not in states:
            continue
        span = float(right - left)
        for frame in range(start, end + 1):
            if frame not in states:
                continue
            alpha = (frame - left) / span
            for index in range(len(states[frame])):
                if index not in quaternion_indices:
                    states[frame][index] = (1.0 - alpha) * states[left][index] + alpha * states[right][index]
            projected_frames.append(frame)
        rotation_right = min(
            int(result.frames[-1]),
            right + max(0, int(rotation_recovery_frames)),
        )
        if rotation_right not in states:
            rotation_right = right
        recovery = max(0, int(rotation_recovery_frames))
        rotation_start = max(start, right - recovery)
        rotation_span = float(max(1, rotation_right - rotation_start))
        for group in projected_quaternion_groups:
            q0 = states[left][list(group)].copy()
            q1 = states[rotation_right][list(group)].copy()
            q0 /= np.linalg.norm(q0)
            q1 /= np.linalg.norm(q1)
            if float(q0 @ q1) < 0.0:
                q1 *= -1.0
            for frame in range(start, rotation_right):
                if frame not in states:
                    continue
                if frame < rotation_start:
                    quaternion = q0.copy()
                else:
                    alpha = (frame - rotation_start) / rotation_span
                    quaternion = (1.0 - alpha) * q0 + alpha * q1
                quaternion /= np.linalg.norm(quaternion)
                states[frame][list(group)] = quaternion
                if frame not in projected_frames:
                    projected_frames.append(frame)
    if not projected_frames:
        return result
    projected_states = tuple(tuple(float(value) for value in states[frame]) for frame in result.frames)
    state_mapping = {frame: state for frame, state in zip(result.frames, projected_states)}
    final_residual = build_runtime_residual_vector(
        problem.residual_execution_plan,
        problem.residual_input_builder(state_mapping),
        problem.factor_ids,
    )
    derived_id = "generic-solve-" + _canonical_hash(
        {
            "parent": result.solve_attempt_id,
            "policy": "bounded_gap_postsolve_interpolation",
            "frames": projected_frames,
            "rotation_recovery_frames": int(rotation_recovery_frames),
            "project_rotation": bool(project_rotation),
            "rotation_policy": "hold_then_stable_visible_recovery",
            "states": projected_states,
        }
    )[:12]
    derived = replace(
        result,
        solve_attempt_id=derived_id,
        parent_solve_attempt_id=result.solve_attempt_id,
        states=projected_states,
        final_squared_error=float(final_residual @ final_residual),
        message=result.message + "; bounded-gap post-solve interpolation applied",
        canonical_sha256="",
    )
    payload = asdict(derived)
    payload.pop("canonical_sha256")
    return replace(derived, canonical_sha256=_canonical_hash(payload))


def reevaluate_solve_result(
    problem: SequenceOptimizationProblem,
    result: GenericSequenceSolveResult,
    *,
    policy: str,
) -> GenericSequenceSolveResult:
    """Recompute residual evidence after a discrete state composition policy."""

    state_mapping = {frame: state for frame, state in zip(result.frames, result.states)}
    final_residual = build_runtime_residual_vector(
        problem.residual_execution_plan,
        problem.residual_input_builder(state_mapping),
        problem.factor_ids,
    )
    derived_id = "generic-solve-" + _canonical_hash(
        {
            "parent": result.solve_attempt_id,
            "policy": policy,
            "states": result.states,
            "residual_sha256": hashlib.sha256(final_residual.tobytes()).hexdigest(),
        }
    )[:12]
    derived = replace(
        result,
        solve_attempt_id=derived_id,
        parent_solve_attempt_id=result.solve_attempt_id,
        final_residual_count=int(final_residual.size),
        final_squared_error=float(final_residual @ final_residual),
        message=result.message + f"; residuals reevaluated after {policy}",
        canonical_sha256="",
    )
    payload = asdict(derived)
    payload.pop("canonical_sha256")
    return replace(derived, canonical_sha256=_canonical_hash(payload))


def project_contact_facing_states(
    problem: SequenceOptimizationProblem,
    result: GenericSequenceSolveResult,
    factor: ContactFacingFactorInput | None,
    *,
    smoothing_passes: int = 0,
    turn_trigger_half_window_frames: int = 0,
    turn_trigger_span_deg: float = 0.0,
    latch_exact_alignment_after_turn: bool = False,
    turn_alignment_ramp_frames: int = 0,
) -> GenericSequenceSolveResult:
    """Hard-project only root yaw for an active persistent grasp-face relation.

    Translation and non-yaw rotation are preserved.  The desired horizontal
    direction is locally averaged on the unit circle before projection, so
    noisy depth/position observations cannot create alternating yaw jitter.
    """

    if factor is None or problem.state_parameterization is None:
        return result
    quaternion_group: tuple[int, int, int, int] | None = None
    offset = 0
    for dof in problem.state_parameterization.state_spec.dofs:
        if dof.kind.value == "rotation_so3":
            names = tuple(field.rsplit(".", 1)[-1] for field in dof.source_fields)
            quaternion_group = tuple(offset + names.index(name) for name in ("qw", "qx", "qy", "qz"))
            break
        offset += dof.dimension
    if quaternion_group is None:
        return result
    states = {
        frame: np.asarray(state, dtype=float).copy()
        for frame, state in zip(result.frames, result.states)
    }
    normal = np.asarray(factor.support_normal_world, dtype=float)
    normal /= np.linalg.norm(normal)
    depth_axis = np.asarray(factor.camera_depth_axis_world, dtype=float)
    depth_axis /= np.linalg.norm(depth_axis)
    local_axis = np.asarray(factor.local_facing_axis, dtype=float)
    local_axis /= np.linalg.norm(local_axis)
    active = tuple(
        frame
        for frame in factor.active_frames
        if frame in states and frame in factor.human_reference_by_frame
    )
    if not active:
        return result
    desired_by_frame: dict[int, np.ndarray] = {}
    for frame in active:
        desired = np.asarray(factor.human_reference_by_frame[frame], dtype=float) - states[frame][:3]
        if factor.depth_relation == "object_behind_human":
            component = float(desired @ depth_axis)
            maximum = -float(factor.minimum_depth_separation_m)
            if component > maximum:
                desired += (maximum - component) * depth_axis
        desired -= normal * float(desired @ normal)
        length = float(np.linalg.norm(desired))
        if length > 1e-8:
            desired_by_frame[frame] = desired / length
    kernel = np.asarray((1.0, 4.0, 6.0, 4.0, 1.0), dtype=float) / 16.0
    runs: list[list[int]] = []
    for frame in active:
        if frame not in desired_by_frame:
            continue
        if not runs or frame != runs[-1][-1] + 1:
            runs.append([frame])
        else:
            runs[-1].append(frame)
    for run in runs:
        directions = np.stack([desired_by_frame[frame] for frame in run])
        exact_alignment_from_frame: int | None = None
        if (
            latch_exact_alignment_after_turn
            and turn_trigger_half_window_frames > 0
            and turn_trigger_span_deg > 0.0
            and len(run) >= 2
        ):
            reference = depth_axis - normal * float(depth_axis @ normal)
            if np.linalg.norm(reference) <= 1e-8:
                reference = directions[0]
            reference /= np.linalg.norm(reference)
            orthogonal = np.cross(normal, reference)
            bearings = np.unwrap(np.asarray([
                np.arctan2(float(direction @ orthogonal), float(direction @ reference))
                for direction in directions
            ]))
            half_window = int(turn_trigger_half_window_frames)
            threshold = np.deg2rad(float(turn_trigger_span_deg))
            for index, frame in enumerate(run):
                start = max(0, index - half_window)
                stop = min(len(run), index + half_window + 1)
                if abs(float(bearings[stop - 1] - bearings[start])) >= threshold:
                    exact_alignment_from_frame = frame
                    break
        for _ in range(max(0, int(smoothing_passes))):
            padded = np.pad(directions, ((2, 2), (0, 0)), mode="edge")
            directions = np.stack(
                [np.sum(padded[index:index + 5] * kernel[:, None], axis=0) for index in range(len(run))]
            )
            directions -= (directions @ normal)[:, None] * normal[None, :]
            directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        reference = depth_axis - normal * float(depth_axis @ normal)
        if np.linalg.norm(reference) <= 1e-8:
            reference = directions[0]
        reference /= np.linalg.norm(reference)
        orthogonal = np.cross(normal, reference)
        desired_bearings = np.unwrap(np.asarray([
            np.arctan2(float(direction @ orthogonal), float(direction @ reference))
            for direction in directions
        ]))
        predicted_directions: list[np.ndarray] = []
        for frame in run:
            q = states[frame][list(quaternion_group)].copy()
            q /= np.linalg.norm(q)
            w, x, y, z = q
            rotation = np.asarray(
                (
                    (1.0 - 2.0 * (y*y + z*z), 2.0 * (x*y - z*w), 2.0 * (x*z + y*w)),
                    (2.0 * (x*y + z*w), 1.0 - 2.0 * (x*x + z*z), 2.0 * (y*z - x*w)),
                    (2.0 * (x*z - y*w), 2.0 * (y*z + x*w), 1.0 - 2.0 * (x*x + y*y)),
                ),
                dtype=float,
            )
            predicted = rotation @ local_axis
            predicted -= normal * float(predicted @ normal)
            predicted /= np.linalg.norm(predicted)
            predicted_directions.append(predicted)
        predicted_bearings = np.unwrap(np.asarray([
            np.arctan2(float(direction @ orthogonal), float(direction @ reference))
            for direction in predicted_directions
        ]))
        angle_errors = desired_bearings - predicted_bearings
        angle_errors -= round(float(angle_errors[0]) / (2.0 * np.pi)) * (2.0 * np.pi)
        for index, (frame, desired) in enumerate(zip(run, directions)):
            q = states[frame][list(quaternion_group)].copy()
            q /= np.linalg.norm(q)
            w, x, y, z = q
            rotation = np.asarray(
                (
                    (1.0 - 2.0 * (y*y + z*z), 2.0 * (x*y - z*w), 2.0 * (x*z + y*w)),
                    (2.0 * (x*y + z*w), 1.0 - 2.0 * (x*x + z*z), 2.0 * (y*z - x*w)),
                    (2.0 * (x*z - y*w), 2.0 * (y*z + x*w), 1.0 - 2.0 * (x*x + y*y)),
                ),
                dtype=float,
            )
            predicted = rotation @ local_axis
            predicted -= normal * float(predicted @ normal)
            predicted /= np.linalg.norm(predicted)
            angle = float(angle_errors[index])
            maximum_angle = float(factor.maximum_facing_angle_rad)
            if exact_alignment_from_frame is not None and frame >= exact_alignment_from_frame:
                progress = (
                    1.0
                    if turn_alignment_ramp_frames <= 0
                    else min(
                        1.0,
                        (frame - exact_alignment_from_frame)
                        / float(turn_alignment_ramp_frames),
                    )
                )
                maximum_angle *= 1.0 - progress
            if abs(angle) <= maximum_angle:
                continue
            angle -= np.sign(angle) * maximum_angle
            half = 0.5 * angle
            delta = np.concatenate(([np.cos(half)], normal * np.sin(half)))
            aw, ax, ay, az = delta
            bw, bx, by, bz = q
            projected = np.asarray(
                (
                    aw*bw - ax*bx - ay*by - az*bz,
                    aw*bx + ax*bw + ay*bz - az*by,
                    aw*by - ax*bz + ay*bw + az*bx,
                    aw*bz + ax*by - ay*bx + az*bw,
                ),
                dtype=float,
            )
            projected /= np.linalg.norm(projected)
            states[frame][list(quaternion_group)] = projected
    projected_states = tuple(tuple(float(value) for value in states[frame]) for frame in result.frames)
    state_mapping = {frame: state for frame, state in zip(result.frames, projected_states)}
    final_residual = build_runtime_residual_vector(
        problem.residual_execution_plan,
        problem.residual_input_builder(state_mapping),
        problem.factor_ids,
    )
    derived = replace(
        result,
        solve_attempt_id="generic-solve-" + _canonical_hash({
            "parent": result.solve_attempt_id,
            "policy": "persistent_contact_facing_state_projection",
            "smoothing_passes": int(smoothing_passes),
            "turn_trigger_half_window_frames": int(turn_trigger_half_window_frames),
            "turn_trigger_span_deg": float(turn_trigger_span_deg),
            "latch_exact_alignment_after_turn": bool(latch_exact_alignment_after_turn),
            "turn_alignment_ramp_frames": int(turn_alignment_ramp_frames),
            "states": projected_states,
        })[:12],
        parent_solve_attempt_id=result.solve_attempt_id,
        states=projected_states,
        final_squared_error=float(final_residual @ final_residual),
        message=(
            result.message
            + f"; contact-facing cone/turn projection smoothing_passes={int(smoothing_passes)}"
            + f" turn_half_window={int(turn_trigger_half_window_frames)}"
            + f" turn_span_deg={float(turn_trigger_span_deg):g}"
            + f" latch_exact={bool(latch_exact_alignment_after_turn)}"
            + f" turn_ramp_frames={int(turn_alignment_ramp_frames)}"
        ),
        canonical_sha256="",
    )
    payload = asdict(derived)
    payload.pop("canonical_sha256")
    return replace(derived, canonical_sha256=_canonical_hash(payload))


def smooth_adjacent_states(
    problem: SequenceOptimizationProblem,
    result: GenericSequenceSolveResult,
    *,
    translation_passes: int,
    rotation_passes: int,
) -> GenericSequenceSolveResult:
    """Apply a small binomial smoother to a solved state sequence.

    This opt-in production projection removes adjacent-frame spikes that a
    robust least-squares loss may deliberately downweight. Quaternion signs
    are made continuous before normalized convolution; sequence endpoints are
    preserved exactly.
    """

    if max(translation_passes, rotation_passes) <= 0 or len(result.states) < 5:
        return result
    states = np.asarray(result.states, dtype=float).copy()
    quaternion_groups: list[tuple[int, int, int, int]] = []
    if problem.state_parameterization is not None:
        offset = 0
        for dof in problem.state_parameterization.state_spec.dofs:
            if dof.kind.value == "rotation_so3":
                names = tuple(field.rsplit(".", 1)[-1] for field in dof.source_fields)
                quaternion_groups.append(
                    tuple(offset + names.index(name) for name in ("qw", "qx", "qy", "qz"))
                )
            offset += dof.dimension
    quaternion_indices = {index for group in quaternion_groups for index in group}
    scalar_indices = [index for index in range(states.shape[1]) if index not in quaternion_indices]
    kernel = np.asarray((1.0, 4.0, 6.0, 4.0, 1.0), dtype=float) / 16.0
    for pass_index in range(max(int(translation_passes), int(rotation_passes))):
        previous = states.copy()
        if pass_index < int(translation_passes):
            for index in scalar_indices:
                padded = np.pad(previous[:, index], (2, 2), mode="edge")
                states[:, index] = np.convolve(padded, kernel, mode="valid")
        if pass_index < int(rotation_passes):
            for group in quaternion_groups:
                continuous = previous[:, list(group)].copy()
                continuous /= np.linalg.norm(continuous, axis=1, keepdims=True)
                for frame_index in range(1, len(continuous)):
                    if float(continuous[frame_index - 1] @ continuous[frame_index]) < 0.0:
                        continuous[frame_index] *= -1.0
                padded = np.pad(continuous, ((2, 2), (0, 0)), mode="edge")
                filtered = np.stack(
                    [np.sum(padded[index:index + 5] * kernel[:, None], axis=0) for index in range(len(continuous))]
                )
                filtered /= np.linalg.norm(filtered, axis=1, keepdims=True)
                states[:, list(group)] = filtered
        states[0] = previous[0]
        states[-1] = previous[-1]
    smoothed_states = tuple(tuple(float(value) for value in row) for row in states)
    state_mapping = {frame: state for frame, state in zip(result.frames, smoothed_states)}
    final_residual = build_runtime_residual_vector(
        problem.residual_execution_plan,
        problem.residual_input_builder(state_mapping),
        problem.factor_ids,
    )
    derived_id = "generic-solve-" + _canonical_hash(
        {
            "parent": result.solve_attempt_id,
            "policy": "adjacent_state_binomial_smoothing",
            "translation_passes": int(translation_passes),
            "rotation_passes": int(rotation_passes),
            "states": smoothed_states,
        }
    )[:12]
    derived = replace(
        result,
        solve_attempt_id=derived_id,
        parent_solve_attempt_id=result.solve_attempt_id,
        states=smoothed_states,
        final_squared_error=float(final_residual @ final_residual),
        message=(
            result.message
            + f"; adjacent-state smoothing translation_passes={int(translation_passes)}"
            + f" rotation_passes={int(rotation_passes)}"
        ),
        canonical_sha256="",
    )
    payload = asdict(derived)
    payload.pop("canonical_sha256")
    return replace(derived, canonical_sha256=_canonical_hash(payload))


def repair_rotation_step_outliers(
    problem: SequenceOptimizationProblem,
    result: GenericSequenceSolveResult,
    *,
    maximum_step_deg: float,
    context_edges: int = 2,
) -> GenericSequenceSolveResult:
    """Repair only nonphysical adjacent quaternion spikes.

    Translation and all non-outlier rotation frames remain byte-for-byte
    unchanged, preserving irregular motion and stops from the solved video.
    """

    if maximum_step_deg <= 0.0 or len(result.states) < 3 or problem.state_parameterization is None:
        return result
    states = np.asarray(result.states, dtype=float).copy()
    quaternion_groups: list[tuple[int, int, int, int]] = []
    offset = 0
    for dof in problem.state_parameterization.state_spec.dofs:
        if dof.kind.value == "rotation_so3":
            names = tuple(field.rsplit(".", 1)[-1] for field in dof.source_fields)
            quaternion_groups.append(
                tuple(offset + names.index(name) for name in ("qw", "qx", "qy", "qz"))
            )
        offset += dof.dimension
    repaired_intervals: list[tuple[int, int]] = []
    for group in quaternion_groups:
        quaternions = states[:, list(group)].copy()
        quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
        for index in range(1, len(quaternions)):
            if float(quaternions[index - 1] @ quaternions[index]) < 0.0:
                quaternions[index] *= -1.0
        steps = np.degrees(
            2.0 * np.arccos(np.clip(np.abs(np.sum(quaternions[:-1] * quaternions[1:], axis=1)), 0.0, 1.0))
        )
        bad_edges = [int(index) for index in np.flatnonzero(steps > float(maximum_step_deg))]
        if not bad_edges:
            continue
        clusters: list[list[int]] = [[bad_edges[0]]]
        for edge in bad_edges[1:]:
            if edge <= clusters[-1][-1] + 3:
                clusters[-1].append(edge)
            else:
                clusters.append([edge])
        for cluster in clusters:
            left = max(0, cluster[0] - int(context_edges))
            touches_tail = cluster[-1] >= len(quaternions) - 2
            right = min(len(quaternions) - 1, cluster[-1] + 1 + int(context_edges))
            if right <= left + 1:
                continue
            q0 = quaternions[left].copy()
            if touches_tail:
                for index in range(left + 1, len(quaternions)):
                    quaternions[index] = q0
                repaired_intervals.append((int(result.frames[left + 1]), int(result.frames[-1])))
                continue
            q1 = quaternions[right].copy()
            if float(q0 @ q1) < 0.0:
                q1 *= -1.0
            span = float(right - left)
            for index in range(left + 1, right):
                alpha = (index - left) / span
                quaternion = (1.0 - alpha) * q0 + alpha * q1
                quaternion /= np.linalg.norm(quaternion)
                quaternions[index] = quaternion
            repaired_intervals.append((int(result.frames[left + 1]), int(result.frames[right - 1])))
        states[:, list(group)] = quaternions
    if not repaired_intervals:
        return result
    repaired_states = tuple(tuple(float(value) for value in row) for row in states)
    state_mapping = {frame: state for frame, state in zip(result.frames, repaired_states)}
    final_residual = build_runtime_residual_vector(
        problem.residual_execution_plan,
        problem.residual_input_builder(state_mapping),
        problem.factor_ids,
    )
    derived_id = "generic-solve-" + _canonical_hash(
        {
            "parent": result.solve_attempt_id,
            "policy": "local_rotation_step_outlier_repair",
            "maximum_step_deg": float(maximum_step_deg),
            "intervals": repaired_intervals,
            "states": repaired_states,
        }
    )[:12]
    derived = replace(
        result,
        solve_attempt_id=derived_id,
        parent_solve_attempt_id=result.solve_attempt_id,
        states=repaired_states,
        final_squared_error=float(final_residual @ final_residual),
        message=(
            result.message
            + f"; local rotation outlier repair max_step_deg={float(maximum_step_deg)}"
            + f" intervals={repaired_intervals}"
        ),
        canonical_sha256="",
    )
    payload = asdict(derived)
    payload.pop("canonical_sha256")
    return replace(derived, canonical_sha256=_canonical_hash(payload))
