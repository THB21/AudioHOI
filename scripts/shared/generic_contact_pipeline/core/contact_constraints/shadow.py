from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from ..base.io import repo_relative_value
from .adapters import adapt_legacy_contact_rows
from .types import constraint_record


def build_contact_constraint_shadow(sample_id: str, contact_csv: Path, rows: list[dict[str, str]]) -> dict[str, object]:
    source_path = str(repo_relative_value(contact_csv))
    adapted = adapt_legacy_contact_rows(sample_id, rows, source_path)
    records = [constraint_record(item) for item in adapted.constraints]
    payload = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema_version": 1,
        "mode": "read_only_shadow",
        "consumed_by_solver": False,
        "sample_id": sample_id,
        "legacy_schema": adapted.schema,
        "source": {"path": source_path, "sha256": hashlib.sha256(contact_csv.read_bytes()).hexdigest(), "rows": len(rows)},
        "constraints": {
            "count": len(records),
            "by_state": dict(sorted(Counter(item.state.value for item in adapted.constraints).items())),
            "by_mode": dict(sorted(Counter(item.mode.value for item in adapted.constraints).items())),
            "by_coordinate": dict(sorted(Counter(item.object_coordinate.kind if item.object_coordinate else "none" for item in adapted.constraints).items())),
            "canonical_sha256": hashlib.sha256(payload).hexdigest(),
        },
        "coverage": {"mapped_fields": list(adapted.mapped_fields), "unmapped_nonempty_fields": list(adapted.unmapped_nonempty_fields)},
    }
