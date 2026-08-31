from __future__ import annotations

import subprocess
from pathlib import Path

from ....core.base.config import CaseProfile
from ....core.base.io import REPO, copy_file, read_csv, repo_path, write_csv, write_json
from ....core.semantics.provenance import resolve_chair_physical6d_seed
from ....core.base.runtime import runtime_python
from ....core.base.schema import stage_paths
from ....core.gates.vlm_gates import apply_contact_gate_to_rows


def apply(profile: CaseProfile) -> dict[str, object]:
    paths = stage_paths(profile)
    flags = set(profile.data.get("ablation_flags", []))
    if "disable_audio_events" in flags:
        pose = copy_file(paths["object_pose_init"], paths["object_pose"])
        contacts = copy_file(paths["contact_candidates"], paths["object_contact_points"])
        metrics = {
            "component": "small_se3",
            "object_pose": str(pose),
            "object_contact_points": str(contacts),
            "policy": "audio-disabled arm preserves the shared visual/VLM Stage3 pose",
            "ablation_flags": sorted(flags),
        }
        write_json(paths["stage4_metrics"], metrics)
        return metrics
    generic_stage4_dir = profile.result_dir / "stage4_generic_refine"
    generic_stage4_render_dir = profile.render_dir / "stage4_generic_refine"
    script = REPO / "scripts/shared/generic_contact_pipeline/components/refinement/solvers/chair_audio_contact_static.py"
    cmd = [
        runtime_python("audiohoi", override_env="AUDIOHOI_PYTHON"),
        str(script),
        "--sample-dir",
        str(profile.sample_dir),
        "--result-dir",
        str(generic_stage4_dir),
        "--render-dir",
        str(generic_stage4_render_dir),
        "--pose-csv",
        str(paths["object_pose_init"]),
        "--observations-csv",
        str(paths["object_observations"]),
        "--leg-candidates-csv",
        str(profile.result_dir / "chair_leg_candidates.csv"),
        "--grasp-state-csv",
        str(profile.result_dir / "chair_grasp_anchor_state/chair_grasp_anchor_state.csv"),
        "--local-points-csv",
        str(paths["object_local_points"]),
        "--local-segments-csv",
        str(paths["object_local_segments"]),
        "--no-render",
    ]
    subprocess.run(cmd, cwd=REPO, check=True)

    pairprop_dir = profile.result_dir / "stage4_pairprop_contact_refine"
    pairprop_dir.mkdir(parents=True, exist_ok=True)
    pairprop_script = REPO / "scripts/shared/generic_contact_pipeline/components/refinement/solvers/chair_twohand_endpoint_se3.py"
    seed_csv, seed_info = resolve_chair_physical6d_seed(profile)
    use_rebuilt_seed = seed_info.get("policy") == "rebuilt_from_mainline_saved2d"
    pairprop_pose = pairprop_dir / "object_pose_pairprop_generic.csv"
    pairprop_metrics = pairprop_dir / "pairprop_contact_refine_metrics.json"
    pairprop_quality = pairprop_dir / "pairprop_quality_metrics.json"
    contacts_for_pairprop = generic_stage4_dir / "object_contact_points.csv"
    gated_contact_rows, vlm_gate_summary = apply_contact_gate_to_rows(profile, read_csv(contacts_for_pairprop), stage="stage2")
    gated_contacts_for_pairprop = generic_stage4_dir / "object_contact_points_vlm_gated.csv"
    write_csv(gated_contacts_for_pairprop, gated_contact_rows)
    if "disable_anchor_propagation" in set(profile.data.get("ablation_flags", [])):
        pose = copy_file(generic_stage4_dir / "object_pose.csv", paths["object_pose"])
        contacts = copy_file(gated_contacts_for_pairprop, paths["object_contact_points"])
        phase = copy_file(generic_stage4_dir / "object_phase.csv", paths["object_phase"])
        metrics = {
            "component": "small_se3",
            "object_pose": str(pose),
            "object_contact_points": str(contacts),
            "object_phase": str(phase),
            "generic_stage4_dir": str(generic_stage4_dir),
            "generic_stage4_contacts_vlm_gated": str(gated_contacts_for_pairprop),
            "vlm_gate_summary": vlm_gate_summary,
            "generic_pairprop_pose_accepted": False,
            "pose_policy": "ablation_disable_anchor_propagation_retains_generic_stage4_pose_without_pairprop_contact_refine",
            "ablation_flags": profile.data.get("ablation_flags", []),
        }
        write_json(paths["stage4_metrics"], metrics)
        return metrics
    pairprop_cmd = [
        runtime_python("audiohoi", override_env="AUDIOHOI_PYTHON"),
        str(pairprop_script),
        "--ref-pose-csv",
        str(seed_csv),
        "--init-pose-csv",
        str(seed_csv),
        "--contacts-csv",
        str(gated_contacts_for_pairprop),
        "--segments-csv",
        str(paths["object_local_segments"]),
        "--out-csv",
        str(pairprop_pose),
        "--metrics-json",
        str(pairprop_metrics),
        "--rot-bound",
        "0.55",
        "--xy-bound",
        "0.45",
        "--z-bound",
        "1.25",
        "--w-contact",
        "2.0" if use_rebuilt_seed else "4.0",
        "--w-2d",
        "0.35",
        "--w-prior-rot",
        "0.06",
        "--w-prior-xy",
        "0.05",
        "--w-prior-z",
        "0.03",
        "--w-temporal",
        "0.35" if use_rebuilt_seed else "0.55",
        "--sigma-contact-m",
        "0.035",
        "--sigma-px",
        "7.5",
        "--max-nfev",
        "240",
    ]
    subprocess.run(pairprop_cmd, cwd=REPO, check=True)
    quality_script = REPO / "scripts/shared/generic_contact_pipeline/components/refinement/solvers/evaluate_chair_pose_quality.py"
    quality_cmd = [
        runtime_python("audiohoi", override_env="AUDIOHOI_PYTHON"),
        str(quality_script),
        "--sample-dir",
        str(profile.sample_dir),
        "--candidate-pose-csv",
        str(pairprop_pose),
        "--baseline-pose-csv",
        str(repo_path(profile.baseline["final_pose_csv"])),
        "--observations-csv",
        str(paths["object_observations"]),
        "--segments-csv",
        str(paths["object_local_segments"]),
        "--contacts-csv",
        str(gated_contacts_for_pairprop),
        "--out-json",
        str(pairprop_quality),
        "--first-active",
        str(profile.data.get("contact_window", {}).get("start_frame", 20)),
        "--last-active",
        str(profile.data.get("contact_window", {}).get("end_frame", 144)),
    ]
    subprocess.run(quality_cmd, cwd=REPO, check=True)
    import json

    with Path(pairprop_quality).open() as f:
        quality = json.load(f)
    use_pairprop_pose = bool(quality.get("quality_gate", {}).get("pass"))

    if use_pairprop_pose:
        pose = copy_file(pairprop_pose, paths["object_pose"])
        pose_policy = "generic pairprop contact-refine pose accepted by semantic-2D/contact/freeze quality gate"
    else:
        pose = copy_file(generic_stage4_dir / "object_pose.csv", paths["object_pose"])
        pose_policy = "generic stage4 pose retained; pairprop quality gate did not pass and no solved-baseline fallback is used"
    contacts = copy_file(gated_contacts_for_pairprop, paths["object_contact_points"])
    phase = copy_file(generic_stage4_dir / "object_phase.csv", paths["object_phase"])
    metrics = {
        "component": "small_se3",
        "object_pose": str(pose),
        "object_contact_points": str(contacts),
        "object_phase": str(phase),
        "generic_stage4_dir": str(generic_stage4_dir),
        "generic_stage4_render_dir": str(generic_stage4_render_dir),
        "generic_stage4_contacts_vlm_gated": str(gated_contacts_for_pairprop),
        "vlm_gate_summary": vlm_gate_summary,
        "generic_pairprop_pose": str(pairprop_pose),
        "generic_pairprop_metrics": str(pairprop_metrics),
        "generic_pairprop_quality": str(pairprop_quality),
        "generic_pairprop_seed_csv": str(seed_csv),
        "generic_pairprop_seed_info": seed_info,
        "generic_pairprop_pose_accepted": use_pairprop_pose,
        "pose_policy": pose_policy,
        "policy": "generic audio/contact/static phase and contact points plus generic pairprop contact-refine pose gated by semantic 2D, palm-contact, and freeze metrics",
    }
    if pairprop_metrics.exists():
        with Path(pairprop_metrics).open() as f:
            metrics["generic_pairprop_summary"] = json.load(f)
    metrics["generic_pairprop_quality_summary"] = quality
    write_json(paths["stage4_metrics"], metrics)
    return metrics
