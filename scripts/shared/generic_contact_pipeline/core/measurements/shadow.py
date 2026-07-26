from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from ..base.io import repo_relative_value
from .adapters import adapt_legacy_observation_rows
from .types import measurement_record


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def build_measurement_shadow(sample_id: str, observation_csv: Path, rows: list[dict[str, str]]) -> dict[str, object]:
    source_path = str(repo_relative_value(observation_csv))
    adapted = adapt_legacy_observation_rows(sample_id, rows, source_path)
    records = [measurement_record(item) for item in adapted.measurements]
    kinds = Counter(str(record["kind"]) for record in records)
    frames = {item.meta.frame for item in adapted.measurements}
    return {
        "schema_version": 1,
        "mode": "read_only_shadow",
        "consumed_by_solver": False,
        "sample_id": sample_id,
        "legacy_schema": adapted.schema,
        "source": {
            "path": source_path,
            "sha256": hashlib.sha256(observation_csv.read_bytes()).hexdigest(),
            "rows": len(rows),
        },
        "measurements": {
            "count": len(records),
            "frames": len(frames),
            "by_kind": dict(sorted(kinds.items())),
            "canonical_sha256": _canonical_hash(records),
        },
        "coverage": {
            "mapped_fields": list(adapted.mapped_fields),
            "unmapped_nonempty_fields": list(adapted.unmapped_nonempty_fields),
        },
    }
