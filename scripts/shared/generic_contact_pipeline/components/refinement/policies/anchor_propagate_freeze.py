from __future__ import annotations

from ....core.base.config import CaseProfile


def apply(profile: CaseProfile) -> dict[str, object]:
    if "disable_anchor_propagation" in set(profile.data.get("ablation_flags", [])):
        return {
            "component": "anchor_propagate_freeze",
            "case_name": profile.case_name,
            "policy": "ablation_disabled_anchor_propagation",
            "ablation_flags": profile.data.get("ablation_flags", []),
        }
    return {
        "component": "anchor_propagate_freeze",
        "case_name": profile.case_name,
        "policy": "preserved_in_mainline_or_anchor_refined_pose",
    }
