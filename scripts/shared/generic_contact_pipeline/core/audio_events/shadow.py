from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..base.io import repo_relative_value
from .adapters import load_audio_events
from .types import audio_event_record


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def build_audio_event_shadow(sample_id: str, result_dir: Path) -> dict[str, object]:
    adapted = load_audio_events(sample_id, result_dir)
    records = repo_relative_value([audio_event_record(event) for event in adapted.events])
    source = str(repo_relative_value(adapted.events[0].source.artifact)) if adapted.events else ""
    core = {
        "schema": adapted.schema,
        "records": records,
    }
    return {
        "source": {
            "path": source,
            "producer": "audio_event_detector_adapter",
        },
        "count": len(records),
        "frames": sorted({int(record["frame"]) for record in records}),
        "canonical_sha256": _canonical_hash(core),
        "consumed_by_solver": False,
    }
