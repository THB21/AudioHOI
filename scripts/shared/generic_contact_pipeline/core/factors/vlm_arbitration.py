from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass

from .activation import (
    ACTIVATION_STATES,
    FactorActivationInterval,
    FactorActivationLedger,
    FactorActivationRecord,
    activation_record,
)


ARBITRATION_LABELS = (
    "visual_observation_reliable",
    "contact_relation_reliable",
    "both_consistent",
    "unclear",
)
ARBITRATION_STATUSES = ("evaluated", "not_evaluated")
_STATUS_RANK = {"inactive": 0, "downweighted": 1, "active": 2}


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _valid_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value.lower())


@dataclass(frozen=True)
class FactorGateDecision:
    decision_id: str
    query_id: str
    start_frame: int
    end_frame: int
    normalized_label: str
    status_by_factor: tuple[tuple[str, str], ...]
    evidence_ids: tuple[str, ...]
    provider: str
    model: str
    prompt_sha256: str
    response_sha256: str
    provenance: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.decision_id or not self.query_id:
            raise ValueError("factor gate decision requires decision and query ids")
        if self.start_frame < 1 or self.end_frame < self.start_frame:
            raise ValueError("factor gate decision requires a positive ordered frame interval")
        if self.normalized_label not in ARBITRATION_LABELS:
            raise ValueError(f"invalid factor arbitration label: {self.normalized_label}")
        if not self.status_by_factor:
            raise ValueError("factor gate decision requires affected factor ids")
        factor_ids = tuple(factor_id for factor_id, _status in self.status_by_factor)
        if any(not factor_id for factor_id in factor_ids) or len(set(factor_ids)) != len(factor_ids):
            raise ValueError("factor gate decision factor ids must be nonempty and unique")
        if any(status not in ACTIVATION_STATES for _factor_id, status in self.status_by_factor):
            raise ValueError("factor gate decision has an invalid activation status")
        if not self.evidence_ids or any(not evidence_id for evidence_id in self.evidence_ids):
            raise ValueError("factor gate decision requires evidence ids")
        if not self.provider or not self.model:
            raise ValueError("factor gate decision requires provider and model provenance")
        if not _valid_sha256(self.prompt_sha256) or not _valid_sha256(self.response_sha256):
            raise ValueError("factor gate decision requires prompt and response sha256 values")
        if not self.provenance:
            raise ValueError("factor gate decision requires provenance")


@dataclass(frozen=True)
class FactorArbitrationLedger:
    schema_version: int
    sample_id: str
    status: str
    decisions: tuple[FactorGateDecision, ...]
    blocking: bool
    provider: str
    model: str
    canonical_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != 1 or not self.sample_id:
            raise ValueError("factor arbitration ledger requires schema version 1 and sample id")
        if self.status not in ARBITRATION_STATUSES:
            raise ValueError(f"invalid factor arbitration ledger status: {self.status}")
        if self.status == "evaluated" and (not self.decisions or not self.provider or not self.model):
            raise ValueError("evaluated factor arbitration ledger requires decisions and provider/model")
        if self.status == "not_evaluated" and self.decisions:
            raise ValueError("not-evaluated factor arbitration ledger cannot contain decisions")
        if not _valid_sha256(self.canonical_sha256):
            raise ValueError("factor arbitration ledger requires a canonical sha256")


def factor_gate_decision_record(decision: FactorGateDecision) -> dict[str, object]:
    payload = asdict(decision)
    payload["status_by_factor"] = [
        {"factor_id": factor_id, "status": status}
        for factor_id, status in decision.status_by_factor
    ]
    return payload


def factor_arbitration_ledger_record(ledger: FactorArbitrationLedger) -> dict[str, object]:
    return {
        "schema_version": ledger.schema_version,
        "sample_id": ledger.sample_id,
        "status": ledger.status,
        "decisions": [factor_gate_decision_record(decision) for decision in ledger.decisions],
        "blocking": ledger.blocking,
        "provider": ledger.provider,
        "model": ledger.model,
        "continuous_pose_override": False,
        "canonical_sha256": ledger.canonical_sha256,
    }


def build_factor_arbitration_ledger(
    *,
    sample_id: str,
    status: str,
    decisions: tuple[FactorGateDecision, ...] = (),
    blocking: bool = False,
    provider: str = "",
    model: str = "",
) -> FactorArbitrationLedger:
    payload = {
        "schema_version": 1,
        "sample_id": sample_id,
        "status": status,
        "decisions": [factor_gate_decision_record(decision) for decision in decisions],
        "blocking": bool(blocking),
        "provider": provider,
        "model": model,
        "continuous_pose_override": False,
    }
    return FactorArbitrationLedger(
        schema_version=1,
        sample_id=sample_id,
        status=status,
        decisions=decisions,
        blocking=bool(blocking),
        provider=provider,
        model=model,
        canonical_sha256=_canonical_hash(payload),
    )


def _expanded_statuses(record: FactorActivationRecord) -> dict[int, str]:
    statuses: dict[int, str] = {}
    for interval in record.intervals:
        for frame in range(interval.start_frame, interval.end_frame + 1):
            statuses[frame] = interval.status
    return statuses


def _compressed_intervals(statuses: dict[int, str]) -> tuple[FactorActivationInterval, ...]:
    if not statuses:
        return ()
    frames = sorted(statuses)
    intervals: list[FactorActivationInterval] = []
    start = frames[0]
    previous = frames[0]
    status = statuses[start]
    for frame in frames[1:]:
        next_status = statuses[frame]
        if frame != previous + 1 or next_status != status:
            intervals.append(FactorActivationInterval(start, previous, status))
            start = frame
            status = next_status
        previous = frame
    intervals.append(FactorActivationInterval(start, previous, status))
    return tuple(intervals)


def merge_factor_activation_ledger(
    base: FactorActivationLedger,
    arbitration: FactorArbitrationLedger,
) -> FactorActivationLedger:
    """Monotonically overlay evaluated VLM decisions on typed interaction activation."""

    if base.sample_id != arbitration.sample_id:
        raise ValueError("factor activation and arbitration sample ids must match")
    if arbitration.status == "not_evaluated":
        return base
    by_id = {record.factor_id: record for record in base.records}
    unknown = sorted(
        {
            factor_id
            for decision in arbitration.decisions
            for factor_id, _status in decision.status_by_factor
            if factor_id not in by_id
        }
    )
    if unknown:
        raise ValueError("factor arbitration targets missing factor ids: " + ",".join(unknown))

    decisions_by_factor: dict[str, list[FactorGateDecision]] = {}
    for decision in arbitration.decisions:
        for factor_id, _status in decision.status_by_factor:
            decisions_by_factor.setdefault(factor_id, []).append(decision)

    records: list[FactorActivationRecord] = []
    for record in base.records:
        statuses = _expanded_statuses(record)
        provenance = list(record.gate_provenance)
        for decision in decisions_by_factor.get(record.factor_id, []):
            requested = dict(decision.status_by_factor)[record.factor_id]
            for frame in range(decision.start_frame, decision.end_frame + 1):
                current = statuses.get(frame)
                if current is None:
                    continue
                statuses[frame] = min((current, requested), key=lambda value: _STATUS_RANK[value])
            marker = f"vlm_factor_arbitration:{decision.decision_id}"
            if marker not in provenance:
                provenance.append(marker)
        intervals = _compressed_intervals(statuses)
        counts = Counter(statuses.values())
        records.append(
            FactorActivationRecord(
                factor_id=record.factor_id,
                kind=record.kind,
                active_frames=counts.get("active", 0),
                downweighted_frames=counts.get("downweighted", 0),
                inactive_frames=counts.get("inactive", 0),
                activation_policy=(
                    record.activation_policy
                    if record.factor_id not in decisions_by_factor
                    else f"{record.activation_policy}+vlm_factor_arbitration"
                ),
                gate_provenance=tuple(provenance),
                intervals=intervals,
                consumed_by_solver=False,
            )
        )
    by_policy = Counter(record.activation_policy for record in records)
    payload = {
        "schema_version": base.schema_version,
        "sample_id": base.sample_id,
        "records": [
            {
                **activation_record(record),
                "intervals": [asdict(interval) for interval in record.intervals],
            }
            for record in records
        ],
        "by_policy": dict(sorted(by_policy.items())),
        "vlm_arbitration_sha256": arbitration.canonical_sha256,
    }
    return FactorActivationLedger(
        schema_version=base.schema_version,
        sample_id=base.sample_id,
        records=tuple(records),
        by_policy=dict(sorted(by_policy.items())),
        canonical_sha256=_canonical_hash(payload),
        consumed_by_solver=False,
    )
