from __future__ import annotations

from pathlib import Path

from ..base.config import CaseProfile
from ..base.io import write_json


def resolve_mug_m17_phase(profile: CaseProfile) -> tuple[Path, dict[str, object]]:
    """Resolve only the current run's observation-derived mug axial phase."""
    phase = profile.result_dir / "observation_seed/axial_phase.csv"
    report = profile.result_dir / "observation_seed/observation_seed_report.json"
    if not phase.exists():
        raise FileNotFoundError(
            f"Missing observation-derived mug phase {phase}; run Stage 1 observation seed generation first"
        )
    return phase, {
        "policy": "observation_derived_axial_phase",
        "phase_source": str(phase),
        "observation_seed_report": str(report),
        "historical_solved_seed_used": False,
        "snapshot_fallback_used": False,
    }


def write_mug_m17_reconstruction_report(profile: CaseProfile, info: dict[str, object]) -> Path:
    report = profile.result_dir / "observation_seed" / "phase_resolution_report.json"
    return write_json(report, info)


def resolve_chair_physical6d_seed(profile: CaseProfile) -> tuple[Path, dict[str, object]]:
    """Resolve only the current run's observation-derived Stage 3 chair pose."""
    seed = profile.result_dir / "object_pose_init.csv"
    if not seed.exists():
        raise FileNotFoundError(f"Missing current-run chair Stage 3 pose {seed}; run Stage 3 first")
    return seed, {
        "policy": "current_stage3_observation_fit",
        "seed_source": str(seed),
        "historical_solved_seed_used": False,
        "snapshot_fallback_used": False,
    }
