from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping

import numpy as np

from ..audio_events import AudioEvent, AudioEventType, load_audio_events
from ..semantics.relations import SemanticRelation, load_semantic_relations
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


def _audio_events_by_frame(sample_id: str, result_dir: Path) -> tuple[dict[int, list[AudioEvent]], str]:
    adapted = load_audio_events(sample_id, result_dir)
    by_frame: dict[int, list[AudioEvent]] = {}
    source = ""
    for event in adapted.events:
        start = event.start_frame if event.start_frame is not None else event.frame
        end = event.end_frame if event.end_frame is not None else event.frame
        for frame in range(start, end + 1):
            by_frame.setdefault(frame, []).append(event)
        source = event.source.artifact
    return by_frame, source


def _semantic_relations_by_frame(result_dir: Path) -> tuple[dict[int, list[SemanticRelation]], str]:
    path = result_dir / "vlm/stage4/semantic_relations.jsonl"
    relations = load_semantic_relations(path)
    by_frame: dict[int, list[SemanticRelation]] = {}
    for relation in relations:
        for frame in range(relation.start_frame, relation.end_frame + 1):
            by_frame.setdefault(frame, []).append(relation)
    return by_frame, str(path) if path.is_file() else ""


def _semantic_choice(
    relations: list[SemanticRelation], predicate: str
) -> SemanticRelation | None:
    candidates = [
        relation
        for relation in relations
        if relation.predicate == predicate and relation.label != "unclear"
    ]
    return max(candidates, key=lambda relation: relation.confidence, default=None)


def _semantic_visibility(
    visual: VisibilityState,
    relations: list[SemanticRelation],
) -> tuple[VisibilityState, list[str]]:
    relation = _semantic_choice(relations, "visibility")
    conflicts: list[str] = []
    if relation is None or relation.confidence < 0.65:
        return visual, conflicts
    mapping = {
        "visible": VisibilityState.VISIBLE,
        "partial": VisibilityState.PARTIALLY_VISIBLE,
        "human_occluded": VisibilityState.OCCLUDED,
        "absent": VisibilityState.ABSENT,
    }
    semantic = mapping.get(relation.label, VisibilityState.UNKNOWN)
    if visual not in {VisibilityState.UNKNOWN, semantic}:
        conflicts.append(f"visibility:{visual.value}!={semantic.value}:{relation.relation_id}")
    # A high-confidence semantic occlusion/absence may explain a degraded mask.
    # A semantic 'visible' label cannot erase an explicit visual occlusion.
    if semantic in {VisibilityState.OCCLUDED, VisibilityState.ABSENT}:
        return semantic, conflicts
    if visual == VisibilityState.UNKNOWN:
        return semantic, conflicts
    if visual == VisibilityState.PARTIALLY_VISIBLE and semantic == VisibilityState.VISIBLE:
        return visual, conflicts
    return semantic, conflicts


def _visual_speed_px(primary: dict[str, str], previous: dict[str, str] | None) -> float:
    if previous is None:
        return 0.0
    u = _float(primary, "ref_u", _float(primary, "center_x"))
    v = _float(primary, "ref_v", _float(primary, "center_y"))
    pu = _float(previous, "ref_u", _float(previous, "center_x"))
    pv = _float(previous, "ref_v", _float(previous, "center_y"))
    return float(((u - pu) ** 2 + (v - pv) ** 2) ** 0.5)


def _rigid_shape_speed_px_by_frame(
    result_dir: Path,
    *,
    maximum_gap_frames: int,
) -> tuple[dict[int, float], str]:
    """Measure tracked rigid-shape change after removing image translation.

    A supported object can rotate in place while its mask centre barely moves.
    Frame-aligned rigid feature tracks expose that motion through a changing 2D
    constellation.  Centring common tracks removes translation; the remaining
    RMS displacement is spread over sparse tracker samples as px/frame.
    """

    path = result_dir / "rigid_feature_measurements.csv"
    rows = _read_csv(path)
    grouped: dict[int, dict[str, list[tuple[float, float, float]]]] = {}
    for row in rows:
        track_id = str(row.get("track_id", "")).strip()
        if not track_id:
            continue
        frame = int(float(row["frame"]))
        weight = max(1e-6, _float(row, "confidence", 1.0))
        grouped.setdefault(frame, {}).setdefault(track_id, []).append(
            (_float(row, "u"), _float(row, "v"), weight)
        )
    points: dict[int, dict[str, np.ndarray]] = {}
    for frame, tracks in grouped.items():
        points[frame] = {}
        for track_id, samples in tracks.items():
            values = np.asarray([[u, v] for u, v, _ in samples], dtype=float)
            weights = np.asarray([weight for _, _, weight in samples], dtype=float)
            points[frame][track_id] = np.average(values, axis=0, weights=weights)
    speed_by_frame: dict[int, float] = {}
    observed_frames = sorted(points)
    for previous, frame in zip(observed_frames, observed_frames[1:]):
        gap = frame - previous
        if gap <= 0 or gap > maximum_gap_frames:
            continue
        common = sorted(set(points[previous]) & set(points[frame]))
        if len(common) < 4:
            continue
        left = np.stack([points[previous][track_id] for track_id in common])
        right = np.stack([points[frame][track_id] for track_id in common])
        left -= np.mean(left, axis=0)
        right -= np.mean(right, axis=0)
        speed = float(np.sqrt(np.mean(np.sum((right - left) ** 2, axis=1))) / gap)
        for selected in range(previous + 1, frame + 1):
            speed_by_frame[selected] = speed
    return speed_by_frame, str(path) if rows else ""


def _contact_ids(
    sample_id: str,
    frame: int,
    rows: list[dict[str, str]],
    support: bool,
    previous_contact_active: bool = False,
) -> tuple[str, ...]:
    ids: list[str] = []
    for index, row in enumerate(rows):
        if support:
            active = _truthy(row, "floor_contact_state", "plane_support_state")
            label = row.get("support_surface_type", "support") or "support"
        else:
            active = _truthy(row, "human_contact_state", "anchor_contact_state", "contact_active")
            label = row.get("contact_label", row.get("contact_part", "contact")) or "contact"
            confidence = max(_float(row, "contact_conf", 0.0), _float(row, "anchor_score", 0.0))
            direct_update = _truthy(row, "anchor_update") or confidence >= 0.5
            hidden_hold = (
                previous_contact_active
                and _truthy(row, "keep_previous")
                and str(row.get("visibility", "")).lower() in {"hidden", "occluded"}
            )
            active = active and (direct_update or hidden_hold)
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
    audio_events: tuple[AudioEvent, ...],
    semantic_relations: tuple[SemanticRelation, ...],
    previous_contact_active: bool,
    result_dir: Path,
    contact_source: str,
    audio_source: str,
    motion_row: dict[str, str] | None,
    motion_source: str,
    visual_speed_px: float,
    audio_visual_conflict: Mapping[str, object] | None = None,
) -> FrameInteractionState:
    frame = int(float(primary["frame"]))
    time = _float(primary, "time")
    active_contact_ids = _contact_ids(
        sample_id, frame, contacts, support=False,
        previous_contact_active=previous_contact_active,
    )
    support_contact_ids = _contact_ids(sample_id, frame, contacts, support=True)
    motion_regime = str((motion_row or {}).get("motion_regime", "")).strip().lower()
    if not active_contact_ids and not support_contact_ids and motion_regime == "static_hold":
        support_contact_ids = (f"{sample_id}:{frame}:static_support:motion_regime",)
    audio_event_ids = tuple(sorted({event.event_id for event in audio_events}))
    audio_types = {event.event_type for event in audio_events}
    semantic_relation_ids = tuple(sorted({relation.relation_id for relation in semantic_relations}))
    visibility, semantic_conflicts = _semantic_visibility(_visibility(primary), list(semantic_relations))
    audio_motion = bool(audio_types & {AudioEventType.SUSTAINED_MOTION, AudioEventType.SHORT_TUG})
    audio_silence = AudioEventType.SILENCE in audio_types
    visual_moving = visual_speed_px > 0.75
    visual_moving_strong = visual_speed_px > 3.0
    audio_conflicts: list[str] = []
    arbitration_decision = str((audio_visual_conflict or {}).get("decision", ""))
    silence_visual_conflict = arbitration_decision == "visual_motion_overrides_audio_silence"
    silence_confirmed_static = arbitration_decision == "audio_silence_confirms_static"
    audio_silence = audio_silence or silence_confirmed_static
    if audio_silence and (visual_moving_strong or silence_visual_conflict):
        audio_conflicts.append("audio_silence_conflicts_with_strong_visual_motion")
    if audio_motion and not visual_moving:
        audio_conflicts.append("audio_motion_without_visual_displacement")
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
        if audio_motion and visual_moving:
            contact_mode = InteractionContactMode.ROLLING
            motion_mode = MotionMode.SUPPORTED_MOVING
        elif audio_silence and not silence_visual_conflict and (silence_confirmed_static or not visual_moving):
            contact_mode = InteractionContactMode.GRASP if active_contact_ids else InteractionContactMode.SUPPORT
            motion_mode = MotionMode.SUPPORTED_STATIC
        elif visual_moving or silence_visual_conflict:
            contact_mode = InteractionContactMode.ROLLING
            motion_mode = MotionMode.SUPPORTED_MOVING
        else:
            contact_mode = InteractionContactMode.GRASP if active_contact_ids else InteractionContactMode.SUPPORT
            motion_mode = MotionMode.SUPPORTED_STATIC
    elif active_contact_ids:
        contact_state = (
            ContactStateAxis.OCCLUDED_HOLD
            if visibility == VisibilityState.OCCLUDED
            else ContactStateAxis.PERSISTENT if previous_contact_active else ContactStateAxis.ACTIVE
        )
        impact_audio = bool(audio_types & {AudioEventType.IMPACT, AudioEventType.CONTACT_ONSET, AudioEventType.SEAM_CLICK})
        contact_mode = InteractionContactMode.IMPACT if impact_audio and not previous_contact_active else InteractionContactMode.GRASP
        motion_mode = MotionMode.ATTACHED
    elif previous_contact_active:
        contact_state = ContactStateAxis.RELEASE
        contact_mode = InteractionContactMode.RELEASE
        motion_mode = MotionMode.FREE
    else:
        contact_state = ContactStateAxis.INACTIVE
        contact_mode = InteractionContactMode.UNKNOWN
        motion_mode = MotionMode.FREE
    semantic_grasp = _semantic_choice(list(semantic_relations), "grasp_state")
    if semantic_grasp is not None and semantic_grasp.confidence >= 0.65:
        if semantic_grasp.label == "released" and active_contact_ids:
            semantic_conflicts.append(
                f"grasp:typed_contact_active!=vlm_released:{semantic_grasp.relation_id}"
            )
        elif semantic_grasp.label == "active" and not active_contact_ids:
            semantic_conflicts.append(
                f"grasp:vlm_active_without_typed_contact:{semantic_grasp.relation_id}"
            )
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
        semantic_relation_ids=semantic_relation_ids,
        confidence=_confidence(primary, contacts),
        provenance={
            "result_dir": str(result_dir),
            "primary_observation": "object_observations.csv",
            "contact_state": contact_source if contacts else "",
            "audio_events": audio_source if audio_event_ids else "",
            "motion_mode": motion_source if motion_row else "",
            "audio_event_types": sorted(event_type.value for event_type in audio_types),
            "semantic_relations": sorted(
                f"{relation.predicate}:{relation.label}:{relation.confidence:.3f}"
                for relation in semantic_relations
            ),
            "semantic_conflicts": semantic_conflicts,
            "audio_conflicts": audio_conflicts,
            "visual_speed_px_per_frame": visual_speed_px,
            "audio_visual_arbitration": dict(audio_visual_conflict or {}),
        },
    )


def build_interaction_timeline(
    sample_id: str,
    result_dir: Path,
    contact_state_artifact: Path | None = None,
    environment_support_mode: str = "",
    audio_visual_arbitration: Mapping[str, object] | None = None,
    audio_enabled: bool = True,
    semantic_enabled: bool = True,
) -> InteractionTimeline:
    if environment_support_mode not in {"", "persistent", "rolling", "sliding"}:
        raise ValueError("environment support mode must be empty, persistent, rolling, or sliding")
    primary_rows = _primary_rows(result_dir)
    contacts_by_frame, contact_source = _contact_rows_by_frame(result_dir, contact_state_artifact)
    motion_by_frame, motion_source = _motion_rows_by_frame(result_dir)
    audio_by_frame, audio_source = (
        _audio_events_by_frame(sample_id, result_dir)
        if audio_enabled
        else ({}, "ablation:disable_audio_events")
    )
    semantic_by_frame, semantic_source = (
        _semantic_relations_by_frame(result_dir)
        if semantic_enabled
        else ({}, "ablation:disable_vlm_semantic_evidence")
    )
    arbitration = dict(audio_visual_arbitration or {})
    sustained_threshold = float(arbitration.get("sustained_visual_motion_px_per_frame", 1.5))
    minimum_conflict_frames = int(arbitration.get("minimum_conflict_frames", 4))
    conflict_fraction_threshold = float(arbitration.get("conflict_fraction", 0.5))
    silence_tail_hold_frames = int(arbitration.get("silence_tail_hold_frames", 4))
    shape_motion_threshold = float(
        arbitration.get("sustained_visual_shape_motion_px_per_frame", 0.3)
    )
    shape_motion_maximum_gap = int(arbitration.get("rigid_shape_maximum_gap_frames", 6))
    if (
        sustained_threshold <= 0.0
        or minimum_conflict_frames < 1
        or not 0.0 <= conflict_fraction_threshold <= 1.0
        or silence_tail_hold_frames < 0
        or shape_motion_threshold <= 0.0
        or shape_motion_maximum_gap < 1
    ):
        raise ValueError("invalid audio/visual arbitration thresholds")
    speed_by_frame: dict[int, float] = {}
    speed_history: list[dict[str, str]] = []
    previous_speed_row: dict[str, str] | None = None
    for row in primary_rows:
        frame = int(float(row["frame"]))
        one = _visual_speed_px(row, previous_speed_row)
        four = _visual_speed_px(row, speed_history[-4]) / 4.0 if len(speed_history) >= 4 else one
        speed_by_frame[frame] = max(one, four)
        previous_speed_row = row
        speed_history.append(row)
    shape_speed_by_frame, shape_speed_source = _rigid_shape_speed_px_by_frame(
        result_dir,
        maximum_gap_frames=shape_motion_maximum_gap,
    )
    arbitration_by_frame: dict[int, dict[str, object]] = {}
    unique_events = {event.event_id: event for events in audio_by_frame.values() for event in events}
    for event in unique_events.values():
        if event.event_type != AudioEventType.SILENCE:
            continue
        start = int(event.start_frame or event.frame)
        end = int(event.end_frame or event.frame)
        speeds = [speed_by_frame[frame] for frame in range(start, end + 1) if frame in speed_by_frame]
        strong = [value for value in speeds if value >= sustained_threshold]
        fraction = len(strong) / max(1, len(speeds))
        conflict = len(strong) >= minimum_conflict_frames and fraction >= conflict_fraction_threshold
        translation_strong_frames = {
            frame
            for frame in range(start, end + 1)
            if speed_by_frame.get(frame, 0.0) >= sustained_threshold
        }
        semantic_turn_frames = {
            frame
            for frame in range(start, end + 1)
            if any(
                relation.predicate == "turn_direction_screen"
                and relation.label in {"clockwise", "counterclockwise"}
                and relation.confidence >= 0.65
                for relation in semantic_by_frame.get(frame, ())
            )
        }
        shape_strong_frames = {
            frame
            for frame in semantic_turn_frames
            if shape_speed_by_frame.get(frame, 0.0) >= shape_motion_threshold
        }
        strong_frames = translation_strong_frames | shape_strong_frames
        local_conflict_frames: set[int] = set()
        run: list[int] = []
        for frame in range(start, end + 2):
            if frame in strong_frames:
                run.append(frame)
                continue
            if len(run) >= minimum_conflict_frames:
                local_conflict_frames.update(run)
            run = []
        common_record = {
            "interval_id": event.event_id,
            "median_speed_px_per_frame": float(np.median(speeds)),
            "p75_speed_px_per_frame": float(np.quantile(speeds, 0.75)),
            "conflict_frame_count": len(strong),
            "valid_frame_count": len(speeds),
            "conflict_fraction": fraction,
            "local_conflict_frame_count": len(local_conflict_frames),
            "local_conflict_runs_required_frames": minimum_conflict_frames,
            "translation_conflict_frame_count": len(translation_strong_frames),
            "rigid_shape_conflict_frame_count": len(shape_strong_frames),
            "rigid_shape_motion_threshold_px_per_frame": shape_motion_threshold,
            "rigid_shape_motion_source": shape_speed_source,
        }
        for frame in range(start, end + 1):
            local_conflict = conflict or frame in local_conflict_frames
            arbitration_by_frame[frame] = {
                **common_record,
                "decision": (
                    "visual_motion_overrides_audio_silence"
                    if local_conflict
                    else "audio_silence_confirms_static"
                ),
                "decision_scope": (
                    "whole_interval"
                    if conflict
                    else "local_contiguous_motion_run"
                    if frame in local_conflict_frames
                    else "local_static_region"
                ),
            }
        tail = (
            0
            if conflict or end in local_conflict_frames
            else silence_tail_hold_frames
        )
        for frame in range(end + 1, end + tail + 1):
            arbitration_by_frame[frame] = {
                **common_record,
                "decision": "audio_silence_confirms_static",
                "decision_scope": "static_tail_hold",
            }
    frames: list[FrameInteractionState] = []
    previous_contact_active = False
    previous_primary: dict[str, str] | None = None
    primary_history: list[dict[str, str]] = []
    for primary in primary_rows:
        frame = int(float(primary["frame"]))
        contacts = list(contacts_by_frame.get(frame, []))
        if environment_support_mode and not any(
            _truthy(row, "floor_contact_state", "plane_support_state") for row in contacts
        ):
            contacts.append(
                {
                    "frame": str(frame),
                    "plane_support_state": "1",
                    "support_surface_type": "environment_plane",
                    "support_conf": "1.0",
                    "source": f"profile_declared_{environment_support_mode}_support_capability",
                }
            )
        one_frame_speed = _visual_speed_px(primary, previous_primary)
        window_speed = (
            _visual_speed_px(primary, primary_history[-4]) / 4.0
            if len(primary_history) >= 4
            else one_frame_speed
        )
        state = _frame_state(
            sample_id,
            primary,
            contacts,
            tuple(audio_by_frame.get(frame, ())),
            tuple(semantic_by_frame.get(frame, ())),
            previous_contact_active,
            result_dir,
            contact_source,
            audio_source,
            motion_by_frame.get(frame),
            motion_source,
            max(one_frame_speed, window_speed),
            arbitration_by_frame.get(frame),
        )
        frames.append(state)
        previous_contact_active = bool(state.active_contact_ids)
        previous_primary = primary
        primary_history.append(primary)
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
            "semantic_relation_frames": sum(1 for state in frames if state.semantic_relation_ids),
            "semantic_conflict_frames": sum(
                1 for state in frames if state.provenance.get("semantic_conflicts")
            ),
            "audio_conflict_frames": sum(
                1 for state in frames if state.provenance.get("audio_conflicts")
            ),
            "audio_event_source": audio_source,
            "audio_visual_conflict_frames": sum(
                1 for record in arbitration_by_frame.values()
                if record["decision"] == "visual_motion_overrides_audio_silence"
            ),
            "semantic_relation_source": semantic_source,
            "environment_support_mode": environment_support_mode,
            "final_pose_read": False,
        },
    )
    errors = validate_interaction_timeline(timeline.frames)
    if errors:
        raise ValueError("; ".join(errors))
    return timeline
