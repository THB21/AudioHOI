from __future__ import annotations

from ....core.base.config import CaseProfile
from ....core.base.io import copy_file, write_json
from ....core.base.schema import stage_paths
from ....core.solver.sphere_sequence import SPHERE_CANDIDATE_NAME, solve_sphere_sequence_candidate


def apply(profile: CaseProfile) -> dict[str, object]:
    paths = stage_paths(profile)
    candidate_dir = paths["sphere_candidate"].parent
    attempt = solve_sphere_sequence_candidate(
        profile,
        profile.result_dir,
        contact_events_csv=paths["contact_events"],
        human_sites_csv=paths["human_sites"],
        support_geometry_json=paths["support_geometry"],
        candidate_dir=candidate_dir,
    )
    candidate_pose = candidate_dir / SPHERE_CANDIDATE_NAME
    promoted_pose = copy_file(candidate_pose, paths["object_pose"])
    promoted_contacts = copy_file(paths["contact_candidates"], paths["object_contact_points"])
    metrics = {
        "component": "generic_sphere_sequence",
        "candidate_attempt": attempt,
        "candidate_pose": str(candidate_pose),
        "object_pose": str(promoted_pose),
        "object_contact_points": str(promoted_contacts),
        "promotion": {
            "accepted_output_written": True,
            "policy": "Stage4 promotes the result-owned typed sphere candidate before the unchanged generic sequence smoother",
            "baseline_pose_read": False,
        },
        "policy": "Measurement IR + contact event/timeline + HumanSite trajectory + support observation -> translation3 sphere sequence",
    }
    write_json(profile.result_dir / "generic_sphere_sequence_metrics.json", metrics)
    return metrics
