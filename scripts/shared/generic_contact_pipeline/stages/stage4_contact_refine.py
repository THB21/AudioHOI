from __future__ import annotations

import importlib

from ..core.config import CaseProfile
from ..core.io import write_json
from ..core.schema import stage_paths


def run(profile: CaseProfile) -> dict[str, object]:
    results = []
    for name in profile.refinement_policies():
        mod = importlib.import_module(f"scripts.shared.generic_contact_pipeline.components.refinement.{name}")
        results.append(mod.apply(profile))
    metrics = {"stage": "stage4_contact_refine", "case_name": profile.case_name, "components": results}
    write_json(stage_paths(profile)["stage4_metrics"], metrics)
    return metrics
