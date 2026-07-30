from __future__ import annotations

from .types import ContactStateAxis, FrameInteractionState, MotionMode, VisibilityState


def validate_frame_interaction_state(state: FrameInteractionState) -> list[str]:
    errors: list[str] = []
    if state.contact_state == ContactStateAxis.INACTIVE and state.active_contact_ids:
        errors.append("inactive contact state must not have active_contact_ids")
    if state.contact_state in {ContactStateAxis.ACTIVE, ContactStateAxis.PERSISTENT} and not (
        state.active_contact_ids or state.support_contact_ids
    ):
        errors.append("active/persistent contact state must reference contact ids")
    if state.contact_state == ContactStateAxis.OCCLUDED_HOLD and state.visibility_state == VisibilityState.VISIBLE:
        errors.append("occluded_hold must not be visible")
    if state.motion_mode == MotionMode.ATTACHED and state.contact_state == ContactStateAxis.INACTIVE:
        errors.append("attached motion requires active or persistent contact")
    if state.motion_mode in {MotionMode.SUPPORTED_STATIC, MotionMode.SUPPORTED_MOVING} and not state.support_contact_ids:
        errors.append("supported motion requires support_contact_ids")
    return errors


def validate_interaction_timeline(frames: tuple[FrameInteractionState, ...]) -> list[str]:
    errors: list[str] = []
    previous = 0
    seen: set[int] = set()
    for state in frames:
        if state.frame in seen:
            errors.append(f"duplicate interaction frame {state.frame}")
        if state.frame <= previous:
            errors.append("interaction frames must be strictly increasing")
        seen.add(state.frame)
        previous = state.frame
        errors.extend(f"frame {state.frame}: {error}" for error in validate_frame_interaction_state(state))
    return errors
