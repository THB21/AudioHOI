from __future__ import annotations

from ...core.base.config import CaseProfile
from ...core.base.io import write_json
from ...core.base.schema import stage_paths
from ...components.mainline import contact_anchor
from ...components.observation.policies.mask_track_center import refine_offscreen_floor_impacts


def run(profile: CaseProfile) -> dict[str, object]:
    result = contact_anchor.build(profile)
    refinement = refine_offscreen_floor_impacts(profile)
    if refinement.get("status") == "refined":
        result = contact_anchor.build(profile)
    result = dict(result)
    result["interaction_conditioned_observation_refinement"] = refinement
    write_json(stage_paths(profile)["stage2_metrics"], result)
    return result
