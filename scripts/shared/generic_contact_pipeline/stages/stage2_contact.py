from __future__ import annotations

import importlib

from ..core.config import CaseProfile


def run(profile: CaseProfile) -> dict[str, object]:
    name = profile.component("contact_policy")
    mod = importlib.import_module(f"scripts.shared.generic_contact_pipeline.components.contact.{name}")
    return mod.build(profile)
