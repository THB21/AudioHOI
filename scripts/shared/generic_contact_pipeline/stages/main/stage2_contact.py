from __future__ import annotations

from ...core.base.config import CaseProfile
from ...components.mainline import contact_anchor


def run(profile: CaseProfile) -> dict[str, object]:
    return contact_anchor.build(profile)
