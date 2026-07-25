from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..base.config import CaseProfile
from .adapters import adapt_legacy_state_rows
from .types import geometry_record, state_spec_record


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def build_state_shadow(profile: CaseProfile, pose_csv: Path, rows: list[dict[str, str]]) -> dict[str, object]:
    adapted = adapt_legacy_state_rows(profile, rows, str(pose_csv))
    spec_record = state_spec_record(adapted.state_spec)
    geometry = geometry_record(adapted.geometry)
    return {
        "schema_version": 1,
        "mode": "read_only_shadow",
        "consumed_by_solver": False,
        "sample_id": profile.case_name,
        "legacy_schema": adapted.schema,
        "source": {
            "path": str(pose_csv),
            "sha256": hashlib.sha256(pose_csv.read_bytes()).hexdigest(),
            "rows": len(rows),
        },
        "state_spec": spec_record,
        "geometry": geometry,
        "coverage": {
            "mapped_fields": list(adapted.mapped_fields),
            "unmapped_nonempty_fields": list(adapted.unmapped_nonempty_fields),
        },
        "canonical_sha256": _canonical_hash({"state_spec": spec_record, "geometry": geometry}),
    }
