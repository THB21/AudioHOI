from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .optimization import GenericSequenceSolveResult, SequenceOptimizationProblem, build_runtime_residual_blocks


ISOLATED_ATTEMPT_FILENAMES = (
    "state.csv",
    "residuals.csv",
    "factor_ledger.json",
    "hard_metrics.json",
    "vlm_gates.json",
    "status.json",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@dataclass(frozen=True)
class IsolatedAttemptState:
    solve_attempt_id: str
    parent_solve_attempt_id: str | None
    sequence_contract_sha256: str
    state_spec_id: str | None
    frames: tuple[int, ...]
    states: tuple[tuple[float, ...], ...]
    result_sha256: str


def write_isolated_sequence_attempt(
    attempts_root: Path,
    problem: SequenceOptimizationProblem,
    result: GenericSequenceSolveResult,
) -> Path:
    """Atomically publish a solver attempt without canonical result names."""

    if not result.solve_attempt_id.startswith("generic-solve-"):
        raise ValueError("isolated attempt requires a generic solve id")
    if result.attempt_id != problem.attempt_id or result.sequence_contract_sha256 != problem.sequence_contract_sha256:
        raise ValueError("isolated attempt result must match its optimization problem")
    if result.frames != problem.frames or result.factor_ids != problem.factor_ids:
        raise ValueError("isolated attempt frames/factors must match its optimization problem")
    if result.accepted_outputs_written:
        raise ValueError("isolated attempt cannot publish a result that wrote accepted outputs")

    attempts_root.mkdir(parents=True, exist_ok=True)
    target = attempts_root / result.solve_attempt_id
    if target.exists():
        status_path = target / "status.json"
        status = json.loads(status_path.read_text()) if status_path.is_file() else {}
        if status.get("result_sha256") == result.canonical_sha256:
            return target
        raise FileExistsError(f"solve attempt directory already exists with different content: {target}")

    temp = Path(tempfile.mkdtemp(prefix=".generic-solve-", dir=attempts_root))
    try:
        state_fields = [f"state_{index:03d}" for index in range(len(result.states[0]))]
        state_rows = [
            {"frame": frame, **{field: f"{float(value):.17g}" for field, value in zip(state_fields, state)}}
            for frame, state in zip(result.frames, result.states)
        ]
        _write_csv(temp / "state.csv", ["frame", *state_fields], state_rows)

        state_mapping = {frame: state for frame, state in zip(result.frames, result.states)}
        residual_inputs = problem.residual_input_builder(state_mapping)
        blocks = build_runtime_residual_blocks(problem.residual_execution_plan, residual_inputs, problem.factor_ids)
        residual_rows: list[dict[str, object]] = []
        factor_records: list[dict[str, object]] = []
        for factor_id, values in blocks:
            factor_records.append(
                {
                    "factor_id": factor_id,
                    "residual_count": int(values.size),
                    "squared_error": float(values @ values),
                    "residual_sha256": hashlib.sha256(values.astype("<f8", copy=False).tobytes()).hexdigest(),
                }
            )
            residual_rows.extend(
                {
                    "factor_id": factor_id,
                    "residual_index": index,
                    "value": f"{float(value):.17g}",
                    "squared_error": f"{float(value * value):.17g}",
                }
                for index, value in enumerate(values)
            )
        _write_csv(temp / "residuals.csv", ["factor_id", "residual_index", "value", "squared_error"], residual_rows)
        _write_json(
            temp / "factor_ledger.json",
            {
                "schema_version": 1,
                "solve_attempt_id": result.solve_attempt_id,
                "sequence_contract_sha256": result.sequence_contract_sha256,
                "factor_count": len(factor_records),
                "residual_count": len(residual_rows),
                "factors": factor_records,
                "jacobian_sparsity_used": result.jacobian_sparsity_used,
                "jacobian_nonzero_count": result.jacobian_nonzero_count,
                "jacobian_density": result.jacobian_density,
            },
        )
        _write_json(
            temp / "hard_metrics.json",
            {"schema_version": 1, "status": "not_evaluated", "accepted_outputs_written": False},
        )
        _write_json(
            temp / "vlm_gates.json",
            {"schema_version": 1, "status": "not_evaluated", "continuous_pose_override": False},
        )
        artifact_hashes = {name: _sha256(temp / name) for name in ISOLATED_ATTEMPT_FILENAMES[:-1]}
        _write_json(
            temp / "status.json",
            {
                "schema_version": 1,
                "contract_attempt_id": result.attempt_id,
                "solve_attempt_id": result.solve_attempt_id,
                "parent_solve_attempt_id": result.parent_solve_attempt_id,
                "sequence_contract_sha256": result.sequence_contract_sha256,
                "state_spec_id": result.state_spec_id,
                "frame_count": len(result.frames),
                "state_width": len(result.states[0]),
                "factor_ids": list(result.factor_ids),
                "success": result.success,
                "message": result.message,
                "function_evaluations": result.function_evaluations,
                "initial_squared_error": result.initial_squared_error,
                "final_squared_error": result.final_squared_error,
                "result_sha256": result.canonical_sha256,
                "case_dispatch_used": False,
                "accepted_outputs_written": False,
                "artifacts": artifact_hashes,
            },
        )
        os.replace(temp, target)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return target


def load_isolated_attempt_state(
    attempt_dir: Path,
    *,
    sequence_contract_sha256: str,
    state_spec_id: str | None,
) -> IsolatedAttemptState:
    status_path = attempt_dir / "status.json"
    state_path = attempt_dir / "state.csv"
    if not status_path.is_file() or not state_path.is_file():
        raise FileNotFoundError("isolated attempt requires status.json and state.csv")
    status = json.loads(status_path.read_text())
    if status.get("sequence_contract_sha256") != sequence_contract_sha256:
        raise ValueError("isolated attempt sequence contract mismatch")
    if status.get("state_spec_id") != state_spec_id:
        raise ValueError("isolated attempt StateSpec mismatch")
    artifacts = status.get("artifacts", {})
    if not isinstance(artifacts, dict):
        raise ValueError("isolated attempt artifact hash ledger is missing")
    for name in ISOLATED_ATTEMPT_FILENAMES[:-1]:
        path = attempt_dir / name
        if not path.is_file() or artifacts.get(name) != _sha256(path):
            raise ValueError(f"isolated attempt artifact hash mismatch: {name}")
    with state_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    state_fields = [field for field in (rows[0] if rows else {}) if field.startswith("state_")]
    frames = tuple(int(row["frame"]) for row in rows)
    states = tuple(tuple(float(row[field]) for field in state_fields) for row in rows)
    if len(rows) != int(status.get("frame_count", 0)) or any(len(state) != int(status.get("state_width", 0)) for state in states):
        raise ValueError("isolated attempt state shape mismatch")
    return IsolatedAttemptState(
        solve_attempt_id=str(status["solve_attempt_id"]),
        parent_solve_attempt_id=status.get("parent_solve_attempt_id"),
        sequence_contract_sha256=sequence_contract_sha256,
        state_spec_id=state_spec_id,
        frames=frames,
        states=states,
        result_sha256=str(status["result_sha256"]),
    )
