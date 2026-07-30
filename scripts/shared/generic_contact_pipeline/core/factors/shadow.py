from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from ..base.config import CaseProfile
from ..base.io import repo_relative_value
from .adapters import adapt_factor_rows, artifact_sha256
from .types import energy_record, factor_record, gap_record


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def build_factor_shadow(profile: CaseProfile, result_dir: Path) -> dict[str, object]:
    adapted = adapt_factor_rows(profile, result_dir)
    factors = [factor_record(factor) for factor in adapted.factors]
    summaries = [energy_record(summary) for summary in adapted.energy_summaries]
    gaps = [gap_record(gap) for gap in adapted.gaps]
    by_kind = Counter(str(record["kind"].value if hasattr(record["kind"], "value") else record["kind"]) for record in factors)
    loss_dir = result_dir / "loss_analysis"
    sources = {
        "per_frame_residuals": {
            "path": str(repo_relative_value(loss_dir / "per_frame_residuals.csv")),
            "sha256": artifact_sha256(loss_dir / "per_frame_residuals.csv"),
        },
        "loss_trace": {
            "path": str(repo_relative_value(loss_dir / "loss_trace.csv")),
            "sha256": artifact_sha256(loss_dir / "loss_trace.csv"),
        },
        "loss_summary": {
            "path": str(repo_relative_value(loss_dir / "loss_summary.json")),
            "sha256": artifact_sha256(loss_dir / "loss_summary.json"),
        },
    }
    canonical_payload = {"factors": factors, "energy_summaries": summaries, "gaps": gaps}
    return {
        "schema_version": 1,
        "mode": "read_only_shadow",
        "consumed_by_solver": False,
        "sample_id": profile.case_name,
        "result_dir": str(repo_relative_value(result_dir)),
        "sources": sources,
        "factors": {
            "count": len(factors),
            "by_kind": dict(sorted(by_kind.items())),
            "records": factors,
        },
        "energy_summaries": summaries,
        "gaps": gaps,
        "coverage": {
            "mapped_fields": list(adapted.mapped_fields),
            "unmapped_nonempty_fields": list(adapted.unmapped_nonempty_fields),
        },
        "canonical_sha256": _canonical_hash(canonical_payload),
    }
