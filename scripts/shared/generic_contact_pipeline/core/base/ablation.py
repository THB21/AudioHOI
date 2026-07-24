from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class AblationMechanism:
    flag: str
    consumers: tuple[str, ...]
    effect: str


RUNTIME_ABLATION_MECHANISMS = {
    item.flag: item
    for item in (
        AblationMechanism(
            "disable_audio_events",
            ("stage0_preprocess", "ball_contact_state"),
            "Do not expose audio events to preprocessing and ball contact-state inference.",
        ),
        AblationMechanism(
            "disable_vlm_contact_gate",
            ("vlm_gates",),
            "Do not activate contact constraints from VLM gate decisions.",
        ),
        AblationMechanism(
            "disable_depth_refine",
            ("anchor_depth",),
            "Disable the anchor-depth refinement component.",
        ),
        AblationMechanism(
            "disable_anchor_propagation",
            ("anchor_propagate_freeze", "small_se3"),
            "Disable anchor propagation/freezing in refinement policies.",
        ),
    )
}


LEGACY_NON_CONSUMED_FLAGS = {
    "no_vlm": "Use --vlm-mode none; this label has no algorithm consumer.",
    "no_llm": "Use --llm-mode none; this label has no algorithm consumer.",
    "no_contact_anchor": "No current pipeline component consumes this flag.",
    "object_only": "This is an evaluation/render label, not a solver ablation flag.",
}


def validate_ablation_flags(flags: Iterable[str]) -> list[str]:
    requested = [str(flag) for flag in flags]
    unknown = sorted(set(requested) - set(RUNTIME_ABLATION_MECHANISMS))
    if unknown:
        explanations = [
            f"{flag}: {LEGACY_NON_CONSUMED_FLAGS.get(flag, 'unknown flag')}"
            for flag in unknown
        ]
        raise ValueError("Unsupported runtime ablation flag(s): " + "; ".join(explanations))
    return requested


def describe_ablation_mechanisms(flags: Iterable[str]) -> dict[str, object]:
    requested = validate_ablation_flags(flags)
    return {
        "requested_flags": requested,
        "effective_mechanisms": [asdict(RUNTIME_ABLATION_MECHANISMS[flag]) for flag in requested],
        "all_requested_flags_have_consumers": True,
    }
