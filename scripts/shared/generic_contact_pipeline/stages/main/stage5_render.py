from __future__ import annotations

import importlib

from ...core.base.config import CaseProfile


def run(profile: CaseProfile) -> dict[str, object]:
    name = profile.component("render_backend")
    mod = importlib.import_module(f"scripts.shared.generic_contact_pipeline.components.render.{name}")
    return mod.render(profile)
