from __future__ import annotations

from pathlib import Path

from ..base.io import REPO


def validate_factor_shadow(shadow: dict[str, object]) -> list[str]:
    errors: list[str] = []
    if shadow.get("mode") != "read_only_shadow":
        errors.append("factor shadow mode must be read_only_shadow")
    if shadow.get("consumed_by_solver") is not False:
        errors.append("factor shadow must not be consumed by solver")

    factors = shadow.get("factors", {})
    records = factors.get("records", []) if isinstance(factors, dict) else []
    seen: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            errors.append(f"factor[{index}] must be a record")
            continue
        factor_id = str(record.get("factor_id", ""))
        if not factor_id:
            errors.append(f"factor[{index}] missing factor_id")
        elif factor_id in seen:
            errors.append(f"duplicate factor_id {factor_id}")
        seen.add(factor_id)
        if record.get("consumed_by_solver") is not False:
            errors.append(f"{factor_id}: consumed_by_solver must be false")
        if not record.get("input_refs"):
            errors.append(f"{factor_id}: missing input_refs")
        source = record.get("residual_source", {})
        artifact = source.get("artifact", "") if isinstance(source, dict) else ""
        if not artifact:
            errors.append(f"{factor_id}: missing residual source artifact")
        elif Path(str(artifact)).is_absolute():
            errors.append(f"{factor_id}: residual source must be repo-relative")
        elif not (REPO / str(artifact)).exists():
            errors.append(f"{factor_id}: residual source does not exist: {artifact}")
        fields = source.get("fields", []) if isinstance(source, dict) else []
        if not fields:
            errors.append(f"{factor_id}: residual source fields must be recorded")

    for gap in shadow.get("gaps", []):
        if not isinstance(gap, dict):
            errors.append("gap must be a record")
            continue
        source = str(gap.get("source", ""))
        if not source:
            errors.append(f"{gap.get('gap_id', '<unknown>')}: gap source missing")
        elif Path(source).is_absolute():
            errors.append(f"{gap.get('gap_id', '<unknown>')}: gap source must be repo-relative")
        elif not (REPO / source).exists():
            errors.append(f"{gap.get('gap_id', '<unknown>')}: gap source does not exist: {source}")
    return errors
