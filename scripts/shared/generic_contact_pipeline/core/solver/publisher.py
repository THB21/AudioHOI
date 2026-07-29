"""Single atomic publisher for generic object sequence results."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from ..state import StateSpec
from .optimization import GenericSequenceSolveResult


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv_atomic(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


@dataclass(frozen=True)
class ObjectPublicationGate:
    passed: bool
    gate_ids: tuple[str, ...]
    blocking_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.gate_ids:
            raise ValueError("object publication requires explicit hard gate ids")
        if self.passed == bool(self.blocking_reasons):
            raise ValueError("object publication pass state and blocking reasons are inconsistent")


@dataclass(frozen=True)
class ObjectPublicationResult:
    status: str
    candidate_path: str
    candidate_sha256: str
    accepted_path: str | None
    accepted_sha256: str | None
    solve_attempt_id: str
    case_dispatch_used: bool
    human_state_optimized: bool


class AcceptedObjectOutputPublisher:
    """Write candidate state always and canonical object_pose.csv only after hard gates."""

    candidate_name = "generic_object_pose_candidate.csv"
    accepted_name = "object_pose.csv"

    def publish(
        self,
        *,
        result: GenericSequenceSolveResult,
        state_spec: StateSpec,
        template_rows: Sequence[Mapping[str, object]],
        candidate_dir: Path,
        accepted_result_dir: Path,
        gate: ObjectPublicationGate,
    ) -> ObjectPublicationResult:
        if candidate_dir.resolve() == accepted_result_dir.resolve():
            raise ValueError("generic object candidate and accepted directories must differ")
        by_frame = {int(row["frame"]): dict(row) for row in template_rows}
        if tuple(sorted(by_frame)) != result.frames:
            raise ValueError("object publication template rows must align with solved frames")
        state_fields = tuple(field for dof in state_spec.dofs for field in dof.source_fields)
        if len(state_fields) != len(result.states[0]):
            raise ValueError("object publication StateSpec fields must match solved state width")
        rows: list[dict[str, object]] = []
        for frame, state in zip(result.frames, result.states):
            row = by_frame[frame]
            row.update({field: f"{float(value):.9f}" for field, value in zip(state_fields, state)})
            row["source"] = "generic_sequence_executor"
            row["generic_solve_attempt_id"] = result.solve_attempt_id
            rows.append(row)
        fields = list(template_rows[0])
        for field in (*state_fields, "source", "generic_solve_attempt_id"):
            if field not in fields:
                fields.append(field)
        candidate_path = candidate_dir / self.candidate_name
        _write_csv_atomic(candidate_path, rows, fields)
        accepted_path: Path | None = None
        if gate.passed:
            accepted_path = accepted_result_dir / self.accepted_name
            _write_csv_atomic(accepted_path, rows, fields)
        return ObjectPublicationResult(
            status="accepted" if accepted_path is not None else "candidate_blocked",
            candidate_path=str(candidate_path),
            candidate_sha256=_sha256(candidate_path),
            accepted_path=str(accepted_path) if accepted_path is not None else None,
            accepted_sha256=_sha256(accepted_path) if accepted_path is not None else None,
            solve_attempt_id=result.solve_attempt_id,
            case_dispatch_used=False,
            human_state_optimized=False,
        )


def object_publication_record(
    result: ObjectPublicationResult,
    gate: ObjectPublicationGate,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": result.status,
        "candidate_path": result.candidate_path,
        "candidate_sha256": result.candidate_sha256,
        "accepted_path": result.accepted_path,
        "accepted_sha256": result.accepted_sha256,
        "solve_attempt_id": result.solve_attempt_id,
        "hard_gate": {
            "passed": gate.passed,
            "gate_ids": list(gate.gate_ids),
            "blocking_reasons": list(gate.blocking_reasons),
        },
        "case_dispatch_used": result.case_dispatch_used,
        "human_state_optimized": result.human_state_optimized,
    }
