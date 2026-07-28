from __future__ import annotations

import csv
from pathlib import Path

from .types import (
    ContactStateAxis,
    FrameInteractionState,
    InteractionContactMode,
    InteractionTimeline,
    MotionMode,
    VisibilityState,
)
from .validation import validate_interaction_timeline


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        raw = row.get(key, "")
        return default if raw in {"", None} else float(raw)
    except Exception:
        return default


def _truthy(row: dict[str, str], *keys: str) -> bool:
    for key in keys:
        value = str(row.get(key, "")).strip().lower()
        if value in {"1", "1.0", "true", "yes", "active"}:
            return True
    return False


def _visibility(row: dict[str, str]) -> VisibilityState:
    raw = str(row.get("vlm_visibility", row.get("visibility", ""))).strip().lower()
    if raw in {"visible", "direct", "clear"}:
        return VisibilityState.VISIBLE
    if raw in {"partial", "partially_visible"}:
        return VisibilityState.PARTIALLY_VISIBLE
    if raw in {"hidden", "occluded", "occluded_by_human"}:
        return VisibilityState.OCCLUDED
    if raw in {"absent", "missing"}:
        return VisibilityState.ABSENT
    confidence = max(_float(row, "observation_conf", -1.0), _float(row, "mask_conf", -1.0), _float(row, "semantic_conf", -1.0))
    if confidence > 0.0:
        return VisibilityState.VISIBLE
    return VisibilityState.UNKNOWN


def _primary_rows(result_dir: Path) -> list[dict[str, str]]:
    observations = _read_csv(result_dir / "object_observations.csv")
    if observations:
        return observations
    return _read_csv(result_dir / "contact_state_frames.csv")


def _contact_rows_by_frame(result_dir: Path) -> dict[int, list[dict[str, str]]]:
    rows = _read_csv(result_dir / "contact_state_frames.csv")
    by_frame: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        by_frame.setdefault(int(float(row["frame"])), []).append(row)
    return by_frame


def _audio_events_by_frame(result_dir: Path) -> dict[int, list[str]]:
    candidates = (
        result_dir / "contact_candidates_internal/audio_events.csv",
        result_dir / "events/audio_events.csv",
    )
    by_frame: dict[int, list[str]] = {}
    for path in candidates:
        for index, row in enumerate(_read_csv(path)):
            frame_raw = row.get("audio_frame", row.get("frame", ""))
            if frame_raw in {"", None}:
                continue
            frame = int(float(frame_raw))
            event_id = str(row.get("event", "")) or f"audio_event_{index + 1}"
            by_frame.setdefault(frame, []).append(event_id)
    return by_frame


def _contact_ids(sample_id: str, frame: int, rows: list[dict[str, str]], support: bool) -> tuple[str, ...]:
    ids: list[str] = []
    for index, row in enumerate(rows):
        if support:
            active = _truthy(row, "floor_contact_state", "plane_support_state")
            label = row.get("support_surface_type", "support") or "support"
        else:
            active = _truthy(row, "human_contact_state", "anchor_contact_state", "contact_active")
            label = row.get("contact_label", row.get("contact_part", "contact")) or "contact"
        if active:
            ids.append(f"{sample_id}:{frame}:{label}:{index}")
    return tuple(ids)


def _confidence(primary: dict[str, str], contacts: list[dict[str, str]]) -> float:
    values = [
        _float(primary, "observation_conf", -1.0),
        _float(primary, "mask_conf", -1.0),
        _float(primary, "semantic_conf", -1.0),
    ]
    for row in contacts:
        values.extend([_float(row, "anchor_score", -1.0), _float(row, "contact_conf", -1.0), _float(row, "support_conf", -1.0)])
    valid = [min(1.0, max(0.0, value)) for value in values if value >= 0.0]
    return float(max(valid)) if valid else 0.5


def _frame_state(
    sample_id: str,
    primary: dict[str, str],
    contacts: list[dict[str, str]],
    audio_event_ids: tuple[str, ...],
    previous_contact_active: bool,
    result_dir: Path,
) -> FrameInteractionState:
    frame = int(float(primary["frame"]))
    time = _float(primary, "time")
    active_contact_ids = _contact_ids(sample_id, frame, contacts, support=False)
    support_contact_ids = _contact_ids(sample_id, frame, contacts, support=True)
    any_contact = bool(active_contact_ids or support_contact_ids)
    visibility = _visibility(primary)
    if active_contact_ids:
        contact_state = ContactStateAxis.PERSISTENT if previous_contact_active else ContactStateAxis.ACTIVE
        contact_mode = InteractionContactMode.IMPACT if audio_event_ids and not previous_contact_active else InteractionContactMode.GRASP
        motion_mode = MotionMode.ATTACHED
    elif support_contact_ids:
        contact_state = ContactStateAxis.ACTIVE
        contact_mode = InteractionContactMode.SUPPORT
        motion_mode = MotionMode.SUPPORTED_STATIC
    elif previous_contact_active:
        contact_state = ContactStateAxis.RELEASE
        contact_mode = InteractionContactMode.RELEASE
        motion_mode = MotionMode.FREE
    else:
        contact_state = ContactStateAxis.INACTIVE
        contact_mode = InteractionContactMode.UNKNOWN
        motion_mode = MotionMode.BALLISTIC if audio_event_ids else MotionMode.FREE
    return FrameInteractionState(
        frame=frame,
        time=time,
        target_entity_id="target_object",
        visibility_state=visibility,
        contact_state=contact_state,
        contact_mode=contact_mode,
        motion_mode=motion_mode,
        active_contact_ids=active_contact_ids,
        support_contact_ids=support_contact_ids,
        audio_event_ids=audio_event_ids,
        semantic_relation_ids=(),
        confidence=_confidence(primary, contacts),
        provenance={
            "result_dir": str(result_dir),
            "primary_observation": "object_observations.csv",
            "contact_state": "contact_state_frames.csv" if contacts else "",
            "audio_events": "contact_candidates_internal/audio_events.csv|events/audio_events.csv" if audio_event_ids else "",
        },
    )


def build_interaction_timeline(sample_id: str, result_dir: Path) -> InteractionTimeline:
    primary_rows = _primary_rows(result_dir)
    contacts_by_frame = _contact_rows_by_frame(result_dir)
    audio_by_frame = _audio_events_by_frame(result_dir)
    frames: list[FrameInteractionState] = []
    previous_contact_active = False
    for primary in primary_rows:
        frame = int(float(primary["frame"]))
        contacts = contacts_by_frame.get(frame, [])
        state = _frame_state(
            sample_id,
            primary,
            contacts,
            tuple(audio_by_frame.get(frame, ())),
            previous_contact_active,
            result_dir,
        )
        frames.append(state)
        previous_contact_active = bool(state.active_contact_ids or state.support_contact_ids)
    timeline = InteractionTimeline(
        schema_version=1,
        sample_id=sample_id,
        target_entity_id="target_object",
        frames=tuple(frames),
        metrics={
            "frame_count": len(frames),
            "active_contact_frames": sum(1 for state in frames if state.contact_state in {ContactStateAxis.ACTIVE, ContactStateAxis.PERSISTENT}),
            "support_contact_frames": sum(1 for state in frames if state.support_contact_ids),
            "audio_event_frames": sum(1 for state in frames if state.audio_event_ids),
            "final_pose_read": False,
        },
    )
    errors = validate_interaction_timeline(timeline.frames)
    if errors:
        raise ValueError("; ".join(errors))
    return timeline
