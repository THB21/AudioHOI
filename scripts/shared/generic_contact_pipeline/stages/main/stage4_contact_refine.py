from __future__ import annotations

from ...core.base.config import CaseProfile
from ...components.mainline import sequence_refine


def run(profile: CaseProfile) -> dict[str, object]:
    return sequence_refine.apply(profile)
