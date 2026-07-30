from __future__ import annotations

import csv
from pathlib import Path

from ..audio_events import load_audio_events
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


def _contact_rows_by_frame(
    result_dir: Path,
    contact_state_artifact: Path | None = None,
) -> tuple[dict[int, list[dict[str, str]]], str]:
    candidates = tuple(
        path.resolve()
        for path in (
        contact_state_artifact,
        result_dir / "contact_state_frames.csv",
        result_dir / "object_contact_points.csv",
        result_dir / "stage4_generic_refine/object_contact_points_vlm_gated.csv",
        )
        if path is not None
    )
    # Contact is multi-edge state: human/object grasp evidence and
    # object/environment support evidence may be produced by different
    # adapters.  Selecting the first non-empty artifact silently drops the
    # other edge (notably during visual occlusion), so merge distinct sources
    # frame-wise.  Exact duplicate paths are still read only once.
    by_frame: dict[int, list[dict[str, str]]] = {}
    sources: list[str] = []
    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        rows = _read_csv(path)
        if not rows:
            continue
        sources.append(str(path))
        for row in rows:
            by_frame.setdefault(int(float(row["frame"])), []).append(row)
    return by_frame, ";".join(sources)


def _motion_rows_by_frame(result_dir: Path) -> tuple[dict[int, dict[str, str]], str]:
    path = result_dir / "motion_regime.csv"
    rows = _read_csv(path)
    return ({int(float(row["frame"])): row for row in rows}, str(path) if rows else "")


def _audio_events_by_frame(sample_id: str, result_dir: Path) -> tuple[dict[int, list[str]], str]:
    adapted = load_audio_events(sample_id, result_dir)
    by_frame: dict[int, list[str]] = {}
    source = ""
    for event in adapted.events:
        by_frame.setdefault(event.frame, []).append(event.event_id)
        source = event.source.artifact
    return by_frame, source


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
    contact_source: str,
    audio_source: str,
    motion_row: dict[str, str] | None,
    motion_source: str,
) -> FrameInteractionState:
    frame = int(float(primary["frame"]))
    time = _float(primary, "time")
    active_contact_ids = _contact_ids(sample_id, frame, contacts, support=False)
    support_contact_ids = _contact_ids(sample_id, frame, contacts, support=True)
    motion_regime = str((motion_row or {}).get("motion_regime", "")).strip().lower()
    if not active_contact_ids and not support_contact_ids and motion_regime == "static_hold":
        support_contact_ids = (f"{sample_id}:{frame}:static_support:motion_regime",)
    any_contact = bool(active_contact_ids or support_contact_ids)
    visibility = _visibility(primary)
    # A grasp and an environment support can coexist.  Environment support
    # determines the object's motion mode (resting), while the hand contact is
    # retained as an active interaction edge.  Treating every multi-contact
    # frame as merely ``attached`` prevents both the support and static factors
    # from activating after an object is placed down.
    if support_contact_ids:
        contact_state = (
            ContactStateAxis.OCCLUDED_HOLD
            if active_contact_ids and visibility == VisibilityState.OCCLUDED
            else ContactStateAxis.PERSISTENT if previous_contact_active else ContactStateAxis.ACTIVE
        )
        contact_mode = InteractionContactMode.GRASP if active_contact_ids else InteractionContactMode.SUPPORT
        motion_mode = MotionMode.SUPPORTED_STATIC
    elif active_contact_ids:
        contact_state = (
            ContactStateAxis.OCCLUDED_HOLD
            if visibility == VisibilityState.OCCLUDED
            else ContactStateAxis.PERSISTENT if previous_contact_active else ContactStateAxis.ACTIVE
        )
        contact_mode = InteractionContactMode.IMPACT if audio_event_ids and not previous_contact_active else InteractionContactMode.GRASP
        motion_mode = MotionMode.ATTACHED
    elif previous_contact_active:
        contact_state = ContactStateAxis.RELEASE
        contact_mode = InteractionContactMode.RELEASE
        motion_mode = MotionMode.FREE
    else:
        contact_state = ContactStateAxis.INACTIVE
        contact_mode = InteractionContactMode.UNKNOWN
        motion_mode = MotionMode.FREE
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
            "contact_state": contact_source if contacts else "",
            "audio_events": audio_source if audio_event_ids else "",
            "motion_mode": motion_source if motion_row else "",
        },
    )


def build_interaction_timeline(
    sample_id: str,
    result_dir: Path,
    contact_state_artifact: Path | None = None,
) -> InteractionTimeline:
    primary_rows = _primary_rows(result_dir)
    contacts_by_frame, contact_source = _contact_rows_by_frame(result_dir, contact_state_artifact)
    motion_by_frame, motion_source = _motion_rows_by_frame(result_dir)
    audio_by_frame, audio_source = _audio_events_by_frame(sample_id, result_dir)
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
            contact_source,
            audio_source,
            motion_by_frame.get(frame),
            motion_source,
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
            "active_contact_frames": sum(
                1
                for state in frames
                if state.contact_state
                in {ContactStateAxis.ACTIVE, ContactStateAxis.PERSISTENT, ContactStateAxis.OCCLUDED_HOLD}
            ),
            "support_contact_frames": sum(1 for state in frames if state.support_contact_ids),
            "audio_event_frames": sum(1 for state in frames if state.audio_event_ids),
            "final_pose_read": False,
        },
    )
    errors = validate_interaction_timeline(timeline.frames)
    if errors:
        raise ValueError("; ".join(errors))
    return timeline
