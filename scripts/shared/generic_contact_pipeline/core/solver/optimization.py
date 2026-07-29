from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .factor_residuals import FactorResidualEvaluator
from .parameterization import StateSpecParameterization
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
