#!/usr/bin/env python3
"""Build the controlled VLM-only vs VLM+audio backward-basketball pilot.

This packages the already solved matched pair, checks that all non-audio inputs and
solver parameters are identical, evaluates both trajectories, and optionally renders
both arms.  It intentionally does not compare against the ground-only baseline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
DEFAULT_SAMPLE = REPO / "samples_known_object/12_back_view_basketball"
DEFAULT_OUT = REPO / "deliverables/backward_basketball_vlm_audio_increment_ablation"
RUN_ROOT = Path("results/full_audio_vlm/derived_stage4")
ARMS = {
    "vlm_only": RUN_ROOT / "active_visual_no_audio",
    "vlm_plus_audio": RUN_ROOT / "active_audio_visual",
}
TRAJECTORY_NAME = "generic_sphere_sequence_candidate.csv"
ATTEMPT_NAME = "generic_sphere_sequence_attempt.json"
RESIDUAL_NAME = "generic_sphere_sequence_residuals.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=REPO, check=True)


def validate_pair(manifests: dict[str, dict]) -> dict:
    left = manifests["vlm_only"]
    right = manifests["vlm_plus_audio"]
    errors: list[str] = []

    expected_flags = {"vlm_only": False, "vlm_plus_audio": True}
    for arm, expected in expected_flags.items():
        if bool(manifests[arm].get("audio_enabled")) is not expected:
            errors.append(f"{arm}: expected audio_enabled={expected}")

    shared_input_keys = [
        "object_measurements",
        "contact_events",
        "contact_timeline",
        "human_sites",
        "support_geometry",
        "trusted_audio_contact_ledger",
    ]
    shared_inputs = {}
    for key in shared_input_keys:
        lval = left.get("inputs", {}).get(key)
        rval = right.get("inputs", {}).get(key)
        if lval != rval:
            errors.append(f"non-audio input differs: {key}")
        shared_inputs[key] = lval

    if left.get("parameters") != right.get("parameters"):
        errors.append("solver parameters differ")
    for key in ("frames", "state_spec", "geometry_provider", "global_depth_shift_m"):
        if left.get(key) != right.get(key):
            errors.append(f"run invariant differs: {key}")

    audio_counts = {
        arm: {
            "audio_event_count": manifest.get("audio_event_count"),
            "audio_soft_floor_anchor_count": manifest.get("audio_soft_floor_anchor_count"),
            "audio_impact_count": manifest.get("audio_impact_count"),
        }
        for arm, manifest in manifests.items()
    }
    return {
        "valid_controlled_pair": not errors,
        "errors": errors,
        "hypothesis": (
            "Adding audio timing supervision to an otherwise identical VLM-guided 4D HOI "
            "pipeline improves contact timing and physical contact plausibility without "
            "materially degrading temporal smoothness."
        ),
        "independent_variable": "audio supervision enabled or disabled",
        "controlled_components": [
            "video frames",
            "human reconstruction",
            "visual object observations",
            "VLM-derived contact semantics and sustained contact state",
            "initialization",
            "solver parameters",
            "object geometry",
            "render settings",
        ],
        "shared_inputs": shared_inputs,
        "audio_factor_check": audio_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--python",
        type=Path,
        default=Path("/home/hebestreit/miniforge3/envs/gvhmr/bin/python"),
    )
    args = parser.parse_args()

    source_dirs = {arm: args.sample_dir / rel for arm, rel in ARMS.items()}
    shared_human_sites = args.sample_dir / "results/full_audio_vlm/human_sites.csv"
    if not shared_human_sites.exists():
        raise FileNotFoundError(f"missing shared read-only human sites: {shared_human_sites}")
    manifests = {
        arm: json.loads((source / ATTEMPT_NAME).read_text())
        for arm, source in source_dirs.items()
    }
    design = validate_pair(manifests)
    if not design["valid_controlled_pair"]:
        raise RuntimeError("invalid controlled pair: " + "; ".join(design["errors"]))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "experiment_design.json").write_text(json.dumps(design, indent=2) + "\n")

    evaluator = REPO / "scripts/shared/evaluation/compute_hoi_interaction_metrics.py"
    renderer = REPO / "scripts/shared/human_ball/render_full_scene_3d.py"
    outputs = {}
    for arm, source in source_dirs.items():
        arm_dir = args.out_dir / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        for name in (TRAJECTORY_NAME, ATTEMPT_NAME, RESIDUAL_NAME):
            shutil.copy2(source / name, arm_dir / name)
        # Publication-facing contract: each arm owns an independently solved
        # object_pose.csv, while human_sites.csv is the identical read-only GVHMR
        # trajectory in both arms.
        trajectory = arm_dir / "object_pose.csv"
        shutil.copy2(source / TRAJECTORY_NAME, trajectory)
        shutil.copy2(shared_human_sites, arm_dir / "human_sites.csv")
        metrics = arm_dir / "hoi_interaction_metrics.json"
        run([
            str(args.python), str(evaluator),
            "--sample-dir", str(args.sample_dir),
            "--trajectory-csv", str(trajectory),
            "--object-type", "sphere",
            "--label", f"backward_basketball__{arm}",
            "--out-json", str(metrics),
        ])
        if args.render:
            render_dir = arm_dir / "render"
            run([
                str(args.python), str(renderer),
                "--sample-dir", str(args.sample_dir),
                "--object-trajectory", str(trajectory),
                "--out-dir", str(render_dir),
                "--mode", "world",
                "--fps", "24",
                "--orbit-turns", "1",
                "--elevation-deg", "16",
            ])
        outputs[arm] = {
            "trajectory": str(trajectory.relative_to(args.out_dir)),
            "trajectory_sha256": sha256(trajectory),
            "human_sites": f"{arm}/human_sites.csv",
            "human_sites_sha256": sha256(arm_dir / "human_sites.csv"),
            "metrics": str(metrics.relative_to(args.out_dir)),
            "render": f"{arm}/render/world.mp4" if args.render else None,
        }

    summary = {
        "experiment": "backward_basketball_vlm_audio_increment_ablation",
        "design": design,
        "outputs": outputs,
        "validation_protocol": {
            "primary": [
                "contact distance at independently annotated contact frames",
                "contact timing error against independently annotated event times",
            ],
            "safety": [
                "non-collision score",
                "object temporal smoothness",
                "human temporal smoothness",
            ],
            "perceptual": (
                "counterbalanced blinded pairwise multi-view comparison; report ties and "
                "verify judge side-bias before using VLM judgments"
            ),
            "statistics": (
                "run the same paired comparison over all sequences; report per-sequence "
                "deltas, mean delta with bootstrap confidence interval, and failure cases"
            ),
        },
    }
    human_hashes = {row["human_sites_sha256"] for row in outputs.values()}
    object_hashes = {row["trajectory_sha256"] for row in outputs.values()}
    summary["paired_artifact_contract"] = {
        "human_sites_identical": len(human_hashes) == 1,
        "object_pose_independently_solved": len(object_hashes) == 2,
        "human_sites_sha256": next(iter(human_hashes)),
    }
    if len(human_hashes) != 1 or len(object_hashes) != 2:
        raise RuntimeError("paired object_pose/human_sites artifact contract failed")
    (args.out_dir / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
