from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from ..factors import (
    ARBITRATION_LABELS,
    FactorArbitrationLedger,
    FactorGateDecision,
    build_factor_arbitration_ledger,
)


QUERY_TYPE = "constraint_reliability_check"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _factor_ids_by_kind(
    factor_records: Sequence[Mapping[str, object]],
) -> dict[str, tuple[str, ...]]:
    by_kind: dict[str, list[str]] = {}
    for record in factor_records:
        factor_id = str(record.get("factor_id", ""))
        kind = str(record.get("kind", ""))
        if factor_id and kind:
            by_kind.setdefault(kind, []).append(factor_id)
    return {kind: tuple(values) for kind, values in by_kind.items()}


def _roles(row: Mapping[str, object], key: str) -> tuple[str, ...]:
    return tuple(value for value in str(row.get(key, "")).split("|") if value)


def _status_by_factor(
    label: str,
    row: Mapping[str, object],
    by_kind: Mapping[str, tuple[str, ...]],
) -> tuple[tuple[str, str], ...]:
    visual_ids = tuple(
        factor_id
        for kind in _roles(row, "visual_factor_roles")
        for factor_id in by_kind.get(kind, ())
    )
    contact_ids = tuple(
        factor_id
        for kind in _roles(row, "contact_factor_roles")
        for factor_id in by_kind.get(kind, ())
    )
    if not visual_ids or not contact_ids:
        raise ValueError("factor reliability query requires compiled visual and contact factors")
    # ``unclear`` is explicitly mapped to ``unclear_no_update`` by the VLM
    # gate.  Keep the compiled factors unchanged in that case; reducing both
    # groups silently changes the trajectory even though the model declined
    # to arbitrate.  Only a positive forced-choice judgment may downweight the
    # competing evidence source.
    visual_status = "downweighted" if label == "contact_relation_reliable" else "active"
    contact_status = "downweighted" if label == "visual_observation_reliable" else "active"
    return tuple(
        [(factor_id, visual_status) for factor_id in visual_ids]
        + [(factor_id, contact_status) for factor_id in contact_ids]
    )


def load_factor_arbitration_ledger(
    *,
    sample_id: str,
    result_dir: Path,
    factor_records: Sequence[Mapping[str, object]],
) -> FactorArbitrationLedger:
    """Load evaluated forced-choice results without exposing free-form text to the solver."""

    stage_dir = result_dir / "vlm" / "stage4"
    queries = [row for row in _rows(stage_dir / "vlm_queries.csv") if row.get("query_type") == QUERY_TYPE]
    raw_path = stage_dir / "qwen_raw_results.json"
    if not queries or not raw_path.is_file():
        return build_factor_arbitration_ledger(sample_id=sample_id, status="not_evaluated")
    raw_payload = json.loads(raw_path.read_text())
    if not isinstance(raw_payload, list):
        raise ValueError("Qwen raw result artifact must contain a list")
    raw_by_id = {
        str(row.get("query_id", "")): row
        for row in raw_payload
        if isinstance(row, Mapping) and row.get("query_type") == QUERY_TYPE
    }
    by_kind = _factor_ids_by_kind(factor_records)
    decisions: list[FactorGateDecision] = []
    blocking = False
    providers: set[str] = set()
    models: set[str] = set()
    for query in queries:
        query_id = str(query.get("query_id", ""))
        raw = raw_by_id.get(query_id)
        provider = str((raw or {}).get("provider", ""))
        model = str((raw or {}).get("model", ""))
        label = str((raw or {}).get("label", "unclear"))
        input_path = Path(str(query.get("input_render_path") or query.get("input_image_path") or ""))
        expected_evidence_hash = str(query.get("evidence_sha256", ""))
        evidence_valid = input_path.is_file() and _sha256_bytes(input_path.read_bytes()) == expected_evidence_hash
        evaluated = raw is not None and bool(provider) and bool(model) and label in ARBITRATION_LABELS and evidence_valid
        if not evaluated:
            label = "unclear"
            provider = provider or "missing_provider"
            model = model or "missing_model"
            blocking = True
        # A valid forced-choice ``unclear`` response means no factor update. It
        # must not reject publication or silently perturb the trajectory.
        # Missing/tampered evidence remains blocking through the branch above.
        providers.add(provider)
        models.add(model)
        prompt_payload = {
            "question": query.get("question", ""),
            "choices": query.get("choices", ""),
            "query_id": query_id,
        }
        response_payload = raw if raw is not None else {"status": "missing_result", "query_id": query_id}
        prompt_sha256 = _sha256_bytes(json.dumps(prompt_payload, sort_keys=True, separators=(",", ":")).encode())
        response_sha256 = _sha256_bytes(json.dumps(response_payload, sort_keys=True, separators=(",", ":")).encode())
        start_frame = int(query.get("start_frame") or query.get("frame") or 0)
        end_frame = int(query.get("end_frame") or query.get("frame") or 0)
        decisions.append(
            FactorGateDecision(
                decision_id=f"vlm-factor-{len(decisions) + 1:04d}",
                query_id=query_id,
                start_frame=start_frame,
                end_frame=end_frame,
                normalized_label=label,
                status_by_factor=_status_by_factor(label, query, by_kind),
                evidence_ids=(f"sha256:{expected_evidence_hash}",),
                provider=provider,
                model=model,
                prompt_sha256=prompt_sha256,
                response_sha256=response_sha256,
                provenance=(
                    str(stage_dir / "vlm_queries.csv"),
                    str(raw_path),
                    f"evidence_verified:{str(evidence_valid).lower()}",
                ),
            )
        )
    if not decisions:
        return build_factor_arbitration_ledger(sample_id=sample_id, status="not_evaluated")
    return build_factor_arbitration_ledger(
        sample_id=sample_id,
        status="evaluated",
        decisions=tuple(decisions),
        blocking=blocking,
        provider="|".join(sorted(providers)),
        model="|".join(sorted(models)),
    )
