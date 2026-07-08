from __future__ import annotations

from ...core.base.config import CaseProfile
from ...components.mainline import pose_init


def run(profile: CaseProfile) -> dict[str, object]:
    return pose_init.build(profile)
