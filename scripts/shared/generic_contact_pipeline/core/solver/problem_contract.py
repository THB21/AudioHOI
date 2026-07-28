from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SequenceProblemContract:
    schema_version: int
    sample_id: str
    state_spec_id: str
    geometry_kind: str
    required_dofs: tuple[str, ...]
    measurement_count: int
    contact_constraint_count: int
    interaction_frame_count: int
    compiled_factor_count: int
    input_hashes: dict[str, str]
    compiled_factor_ids: tuple[str, ...]
    canonical_sha256: str
    consumed_by_solver: bool = False

    def __post_init__(self) -> None:
        if not self.sample_id or not self.state_spec_id or not self.geometry_kind:
            raise ValueError("SequenceProblemContract requires sample_id, state_spec_id, and geometry_kind")
        if not self.required_dofs:
            raise ValueError("SequenceProblemContract requires required_dofs")
        for label, count in (
            ("measurement_count", self.measurement_count),
            ("contact_constraint_count", self.contact_constraint_count),
            ("interaction_frame_count", self.interaction_frame_count),
            ("compiled_factor_count", self.compiled_factor_count),
        ):
            if count <= 0:
                raise ValueError(f"SequenceProblemContract requires positive {label}")
        if len(self.compiled_factor_ids) != self.compiled_factor_count:
            raise ValueError("compiled_factor_ids must match compiled_factor_count")
        if self.consumed_by_solver:
            raise ValueError("SequenceProblemContract is shadow-only until the generic executor consumes it")


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def sequence_problem_contract_record(contract: SequenceProblemContract) -> dict[str, object]:
    return asdict(contract)


def build_sequence_problem_contract(
    *,
    sample_id: str,
    state_contract: dict[str, object],
    measurement_shadow: dict[str, object],
    contact_shadow: dict[str, object],
    interaction_shadow: dict[str, object],
    compiled_factor_shadow: dict[str, object],
) -> SequenceProblemContract:
    compiled_records = compiled_factor_shadow.get("records", [])
    if not isinstance(compiled_records, list):
        compiled_records = []
    compiled_factor_ids = tuple(str(record.get("factor_id")) for record in compiled_records if isinstance(record, dict) and record.get("factor_id"))
    input_hashes = {
        "measurements": str(measurement_shadow["measurements"]["canonical_sha256"]),
        "contact_constraints": str(contact_shadow["constraints"]["canonical_sha256"]),
        "interaction_state": str(interaction_shadow["canonical_sha256"]),
        "compiled_factors": str(compiled_factor_shadow["canonical_sha256"]),
    }
    payload = {
        "schema_version": 1,
        "sample_id": sample_id,
        "state_spec_id": str(state_contract["spec_id"]),
        "geometry_kind": str(state_contract["geometry_kind"]),
        "required_dofs": tuple(str(item) for item in state_contract["required_dofs"]),  # type: ignore[index]
        "measurement_count": int(measurement_shadow["measurements"]["count"]),
        "contact_constraint_count": int(contact_shadow["constraints"]["count"]),
        "interaction_frame_count": int(interaction_shadow["frame_count"]),
        "compiled_factor_count": int(compiled_factor_shadow["count"]),
        "input_hashes": input_hashes,
        "compiled_factor_ids": compiled_factor_ids,
    }
    return SequenceProblemContract(
        **payload,
        canonical_sha256=_canonical_hash(payload),
        consumed_by_solver=False,
    )
