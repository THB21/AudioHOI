from __future__ import annotations

from ...core.config import CaseProfile


def apply(profile: CaseProfile) -> dict[str, object]:
    return {
        "component": "backproject_xy",
        "case_name": profile.case_name,
        "policy": "preserved_in_solved_anchor_refined_pose",
    }
