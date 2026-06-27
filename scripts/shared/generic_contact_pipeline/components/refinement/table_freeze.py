from __future__ import annotations

from ...core.base.config import CaseProfile


def apply(profile: CaseProfile) -> dict[str, object]:
    return {
        "component": "table_freeze",
        "case_name": profile.case_name,
        "policy": "preserved_in_final_solved_pose",
    }
