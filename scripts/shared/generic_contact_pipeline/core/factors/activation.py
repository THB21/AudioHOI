from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass

from ..interaction import (
    ContactStateAxis,
    FrameInteractionState,
    InteractionContactMode,
    InteractionTimeline,
    MotionMode,
    VisibilityState,
)
from .types import FactorKind, FactorSpec


ACTIVATION_STATES = ("active", "downweighted", "inactive")


@dataclass(frozen=True)
class FactorActivationInterval:
    start_frame: int
    end_frame: int
    status: str

    def __post_init__(self) -> None:
        if self.start_frame < 1 or self.end_frame < self.start_frame:
            raise ValueError("factor activation interval requires a valid positive frame range")
        if self.status not in ACTIVATION_STATES:
            raise ValueError(f"invalid factor activation status: {self.status}")


@dataclass(frozen=True)
class FactorActivationRecord:
    factor_id: str
    kind: FactorKind
    active_frames: int
    downweighted_frames: int
    inactive_frames: int
    activation_policy: str
    gate_provenance: tuple[str, ...]
    intervals: tuple[FactorActivationInterval, ...] = ()
    consumed_by_solver: bool = False

    def __post_init__(self) -> None:
        if not self.factor_id or self.active_frames < 0 or self.downweighted_frames < 0 or self.inactive_frames < 0:
            raise ValueError("invalid factor activation record")
        if self.intervals:
            previous_end = 0
            interval_counts = Counter()
            for interval in self.intervals:
                if interval.start_frame <= previous_end:
                    raise ValueError("factor activation intervals must be ordered and non-overlapping")
                previous_end = interval.end_frame
                interval_counts[interval.status] += interval.end_frame - interval.start_frame + 1
            expected = {
                "active": self.active_frames,
                "downweighted": self.downweighted_frames,
                "inactive": self.inactive_frames,
            }
            if dict(interval_counts) != {key: value for key, value in expected.items() if value}:
                raise ValueError("factor activation interval counts do not match summary counts")
        if self.consumed_by_solver:
            raise ValueError("factor activation ledger is shadow-only in this branch")


@dataclass(frozen=True)
class FactorActivationLedger:
    schema_version: int
    sample_id: str
    records: tuple[FactorActivationRecord, ...]
    by_policy: dict[str, int]
    canonical_sha256: str
    consumed_by_solver: bool = False

    def __post_init__(self) -> None:
        if self.consumed_by_solver:
            raise ValueError("factor activation ledger is shadow-only in this branch")


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _visual_state(state: FrameInteractionState) -> str:
    if state.visibility_state in {VisibilityState.VISIBLE, VisibilityState.PARTIALLY_VISIBLE}:
        return "active"
    if state.visibility_state in {VisibilityState.OCCLUDED, VisibilityState.ABSENT}:
        return "downweighted"
    return "active"


def _contact_state(state: FrameInteractionState) -> str:
    if state.contact_state in {ContactStateAxis.ACTIVE, ContactStateAxis.PERSISTENT, ContactStateAxis.OCCLUDED_HOLD}:
        return "active"
    if state.contact_state in {ContactStateAxis.CANDIDATE, ContactStateAxis.RELEASE}:
        return "downweighted"
    return "inactive"


def _support_state(state: FrameInteractionState) -> str:
    if state.support_contact_ids or state.contact_mode in {
        InteractionContactMode.SUPPORT,
        InteractionContactMode.SLIDING,
        InteractionContactMode.ROLLING,
    }:
        return "active"
    if state.motion_mode in {MotionMode.SUPPORTED_STATIC, MotionMode.SUPPORTED_MOVING}:
        return "active"
    return "inactive"


def _temporal_velocity_state(state: FrameInteractionState) -> str:
    if state.motion_mode in {MotionMode.HIGH_SPEED, MotionMode.BALLISTIC}:
        return "downweighted"
    return "active"


def _temporal_acceleration_state(state: FrameInteractionState) -> str:
    if state.contact_mode == InteractionContactMode.IMPACT or state.motion_mode in {MotionMode.HIGH_SPEED, MotionMode.BALLISTIC}:
        return "downweighted"
    return "active"


def _audio_state(state: FrameInteractionState) -> str:
    if not state.audio_event_ids:
        return "inactive"
    if state.contact_state in {ContactStateAxis.ACTIVE, ContactStateAxis.RELEASE} or state.contact_mode in {
        InteractionContactMode.IMPACT,
        InteractionContactMode.RELEASE,
    } or state.motion_mode in {MotionMode.BALLISTIC, MotionMode.HIGH_SPEED}:
        return "active"
    return "downweighted"


def _static_state(state: FrameInteractionState) -> str:
    return "active" if state.motion_mode == MotionMode.SUPPORTED_STATIC else "inactive"


def _always_state(_state: FrameInteractionState) -> str:
    return "active"


def _policy_for_kind(kind: FactorKind) -> tuple[str, object, tuple[str, ...]]:
    if kind in {FactorKind.POINT_REPROJECTION, FactorKind.LINE_REPROJECTION, FactorKind.MASK_SILHOUETTE, FactorKind.METRIC_DEPTH}:
        return (
            "visibility_state_controls_visual_observation",
            _visual_state,
            ("VisibilityState",),
        )
    if kind == FactorKind.DEPTH_ORDER:
        return (
            "visibility_state_controls_depth_order",
            _visual_state,
            ("VisibilityState",),
        )
    if kind == FactorKind.CONTACT_DISTANCE:
        return (
            "contact_state_controls_contact_distance",
            _contact_state,
            ("ContactState", "ContactMode", "active_contact_ids"),
        )
    if kind == FactorKind.SUPPORT_AND_PENETRATION:
        return (
            "support_state_controls_support_penetration",
            _support_state,
            ("ContactMode", "MotionMode", "support_contact_ids"),
        )
    if kind == FactorKind.TEMPORAL_VELOCITY:
        return (
            "motion_mode_controls_velocity_smoothness",
            _temporal_velocity_state,
            ("MotionMode",),
        )
    if kind == FactorKind.TEMPORAL_ACCELERATION:
        return (
            "interaction_state_controls_acceleration_smoothness",
            _temporal_acceleration_state,
            ("ContactMode", "MotionMode"),
        )
    if kind == FactorKind.STATIC_FREEZE:
        return (
            "motion_mode_controls_static_freeze",
            _static_state,
            ("MotionMode",),
        )
    if kind == FactorKind.AUDIO_EVENT_PRIOR:
        return (
            "audio_and_interaction_transition_control_alignment",
            _audio_state,
            ("AudioEventIR", "audio_event_ids", "ContactState", "ContactMode"),
        )
    return (
        "always_on_state_or_geometry_regularizer",
        _always_state,
        ("StateSpec",),
    )


def _compile_intervals(
    states: tuple[FrameInteractionState, ...],
    selector: object,
) -> tuple[FactorActivationInterval, ...]:
    if not states:
        return ()
    intervals: list[FactorActivationInterval] = []
    start_frame = states[0].frame
    previous_frame = states[0].frame
    current_status = str(selector(states[0]))  # type: ignore[operator]
    for state in states[1:]:
        status = str(selector(state))  # type: ignore[operator]
        if state.frame != previous_frame + 1 or status != current_status:
            intervals.append(FactorActivationInterval(start_frame, previous_frame, current_status))
            start_frame = state.frame
            current_status = status
        previous_frame = state.frame
    intervals.append(FactorActivationInterval(start_frame, previous_frame, current_status))
    return tuple(intervals)


def build_factor_activation_ledger(
    sample_id: str,
    factors: tuple[FactorSpec, ...],
    timeline: InteractionTimeline,
) -> FactorActivationLedger:
    records: list[FactorActivationRecord] = []
    for factor in factors:
        policy, selector, provenance = _policy_for_kind(factor.kind)
        counts = Counter(str(selector(state)) for state in timeline.frames)  # type: ignore[misc]
        intervals = _compile_intervals(timeline.frames, selector)
        records.append(
            FactorActivationRecord(
                factor_id=factor.factor_id,
                kind=factor.kind,
                active_frames=counts.get("active", 0),
                downweighted_frames=counts.get("downweighted", 0),
                inactive_frames=counts.get("inactive", 0),
                activation_policy=policy,
                gate_provenance=provenance,
                intervals=intervals,
                consumed_by_solver=False,
            )
        )
    by_policy = Counter(record.activation_policy for record in records)
    payload = {
        "schema_version": 1,
        "sample_id": sample_id,
        "records": [activation_record(record) for record in records],
        "by_policy": dict(sorted(by_policy.items())),
    }
    return FactorActivationLedger(
        schema_version=1,
        sample_id=sample_id,
        records=tuple(records),
        by_policy=dict(sorted(by_policy.items())),
        canonical_sha256=_canonical_hash(payload),
        consumed_by_solver=False,
    )


def activation_record(record: FactorActivationRecord) -> dict[str, object]:
    payload = asdict(record)
    payload["kind"] = record.kind.value
    # The legacy activation shadow remains a compact summary. Production
    # intervals are promoted only through configured CompiledFactors.
    payload.pop("intervals")
    return payload
