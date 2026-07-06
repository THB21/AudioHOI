from __future__ import annotations

from ...core.base.config import CaseProfile
from ...components.mainline import observation


def run(profile: CaseProfile) -> dict[str, object]:
    return observation.build(profile)
