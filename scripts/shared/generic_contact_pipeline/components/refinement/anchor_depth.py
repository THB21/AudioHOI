from __future__ import annotations

import subprocess

from ...core.base.config import CaseProfile
from ...core.evaluation.ball_residuals import write_ball_residual_report
from ...core.base.camera import backproject_uvz
from ...core.base.io import copy_file, float_or_none, read_csv, repo_path, write_csv, write_json
from ...core.base.runtime import runtime_python
from ...core.base.schema import stage_paths
from ...core.gates.vlm_gates import apply_contact_gate_to_rows


def _interp_anchor_segment(frames: list[int], z_init: list[float], anchor_values: dict[int, float]) -> list[float]:
    if not anchor_values:
        return list(z_init)
    frame_to_i = {fr: i for i, fr in enumerate(frames)}
    anchors = sorted((frame_to_i[fr], z) for fr, z in anchor_values.items() if fr in frame_to_i)
    if not anchors:
        return list(z_init)
    out = list(z_init)
    first_i, first_z = anchors[0]
    last_i, last_z = anchors[-1]
    for i in range(0, first_i + 1):
        out[i] = first_z
    for (li, lz), (ri, rz) in zip(anchors[:-1], anchors[1:]):
        if ri <= li:
            continue
        span = ri - li
        for i in range(li, ri + 1):
            a = (i - li) / span
            out[i] = (1.0 - a) * lz + a * rz
    for i in range(last_i, len(out)):
        out[i] = last_z
    return [max(0.2, z) for z in out]


def _build_pose_rows(
    init_rows: list[dict[str, str]],
    z_values: list[float],
    profile: CaseProfile,
    contacts_by_frame: dict[int, dict[str, str]],
    source: str,
) -> list[dict[str, object]]:
    out_rows = []
    for row, z in zip(init_rows, z_values):
        u = float_or_none(row.get("u_obs")) or 0.0
        v = float_or_none(row.get("v_obs")) or 0.0
        tx, ty, tz = backproject_uvz(u, v, z, profile.camera)
        fr = int(float(row["frame"]))
        crow = contacts_by_frame.get(fr, {})
        new = dict(row)
        new.update(
            {
                "tx": f"{tx:.6f}",
                "ty": f"{ty:.6f}",
                "tz": f"{tz:.6f}",
                "z_ref_anchor_segment": f"{z:.6f}",
                "contact_frame": crow.get("contact_active", "0"),
                "contact_part": crow.get("human_part", ""),
                "contact_side": crow.get("human_side", ""),
                "contact_label": crow.get("human_part", ""),
                "contact_u": crow.get("contact_u", ""),
                "contact_v": crow.get("contact_v", ""),
                "contact_depth_offset_m": crow.get("contact_depth_offset_m", ""),
                "source": source,
            }
        )
        out_rows.append(new)
    return out_rows


def _max_abs_delta(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    return max(abs(x - y) for x, y in zip(a, b))


def _run_exact_anchor_refinement(profile: CaseProfile, pose_csv: str, proxy_csv: str, state_csv: str) -> dict[str, str]:
    python_bin = runtime_python("audiohoi", override_env="AUDIOHOI_PYTHON")
    script = repo_path("scripts/shared/generic_contact_pipeline/components/refinement/anchor_depth_solver.py")
    event_csv = profile.sample_dir / "results/contact_candidates_object_proxy/contact_candidates_labeled.csv"
    out_subdir = f"{profile.result_name}/exact_anchor_depth_reference"
    cmd = [
        python_bin,
        str(script),
        "--sample-dir",
        str(profile.sample_dir),
        "--out-subdir",
        out_subdir,
        "--contact-state-csv",
        str(repo_path(state_csv)),
        "--contact-event-csv",
        str(event_csv),
        "--object-proxy-observation-csv",
        str(repo_path(proxy_csv)),
        "--object-pose-csv",
        str(repo_path(pose_csv)),
    ]
    subprocess.run(cmd, cwd=repo_path("."), check=True)
    out_dir = profile.sample_dir / "results" / out_subdir
    return {
        "python": python_bin,
        "script": str(script),
        "out_dir": str(out_dir),
        "pose_csv": str(out_dir / "object_pose6d_sharedcam_contactphase_trajectory.csv"),
        "summary_txt": str(out_dir / "object_pose6d_sharedcam_contactphase_summary.txt"),
    }


def apply(profile: CaseProfile) -> dict[str, object]:
    paths = stage_paths(profile)
    flags = set(profile.data.get("ablation_flags", []))
    if "disable_depth_refine" in flags:
        out = copy_file(paths["object_pose_init"], paths["object_pose"])
        contacts = copy_file(paths["contact_candidates"], paths["object_contact_points"]) if paths["contact_candidates"].exists() else None
        metrics = {
            "component": "anchor_depth",
            "object_pose": str(out),
            "object_contact_points": str(contacts) if contacts else "",
            "policy": "ablation_disable_depth_refine_copy_stage3_pose",
            "ablation_flags": sorted(flags),
        }
        write_json(paths["stage4_metrics"], metrics)
        return metrics
    if profile.component("pose_model") != "translation3":
        metrics = {
            "component": "anchor_depth",
            "policy": "no_op_for_non_translation_pose_model",
            "object_pose": str(paths["object_pose"]) if paths["object_pose"].exists() else "",
        }
        write_json(paths["stage4_metrics"], metrics)
        return metrics

    init_rows = read_csv(paths["object_pose_init"])
    obs_rows = {int(float(r["frame"])): r for r in read_csv(paths["object_observations"])}
    raw_contact_rows = read_csv(paths["contact_candidates"])
    contact_rows, vlm_gate_summary = apply_contact_gate_to_rows(profile, raw_contact_rows, stage="stage2")
    contacts_by_frame = {int(float(r["frame"])): r for r in contact_rows}

    frames = [int(float(r["frame"])) for r in init_rows]
    z_init = [float_or_none(r.get("tz")) or 1.0 for r in init_rows]
    anchor_values: dict[int, float] = {}
    for fr in frames:
        crow = contacts_by_frame.get(fr, {})
        active = int(float(crow.get("contact_active", "0") or 0))
        if not active:
            continue
        if crow.get("human_part") in {"", "none", "floor"}:
            continue
        obs = obs_rows.get(fr, {})
        active_z = float_or_none(obs.get("active_part_z"))
        offset = float_or_none(crow.get("contact_depth_offset_m"))
        score = float_or_none(crow.get("anchor_score")) or 0.0
        if active_z is None or offset is None or score < 0.05:
            continue
        anchor_values[fr] = active_z - offset

    z_raw = _interp_anchor_segment(frames, z_init, anchor_values)
    raw_rows = _build_pose_rows(
        init_rows,
        z_raw,
        profile,
        contacts_by_frame,
        "generic_anchor_depth_raw_contact_interp_debug",
    )
    raw_debug_path = paths["object_pose"].with_name("object_pose_anchor_raw_debug.csv")
    write_csv(raw_debug_path, raw_rows)
    exact = _run_exact_anchor_refinement(
        profile=profile,
        pose_csv=str(paths["object_pose_init"]),
        proxy_csv=str(paths["contact_candidates"]),
        state_csv=str(paths["contact_state"]),
    )
    out = copy_file(exact["pose_csv"], paths["object_pose"])
    contacts = copy_file(paths["contact_candidates"], paths["object_contact_points"]) if paths["contact_candidates"].exists() else None
    final_rows = read_csv(paths["object_pose"])
    z_final = [float_or_none(r.get("z_ref_anchor_segment")) or float_or_none(r.get("tz")) or z for r, z in zip(final_rows, z_init)]
    residual_report = write_ball_residual_report(
        profile,
        paths["object_pose"],
        stage_label="stage4_anchor_depth_final",
        out_name="ball_stage4_anchor_depth_residuals.csv",
        init_csv=paths["object_pose_init"],
    )
    metrics = {
        "component": "anchor_depth",
        "object_pose": str(out),
        "object_contact_points": str(contacts) if contacts else "",
        "anchors_used": len(anchor_values),
        "policy": "exact_radius_free_anchor_refinement_audiohoi_env_using_stage2_contact_proxy",
        "exact_optimizer": exact,
        "raw_debug_pose": str(raw_debug_path),
        "raw_debug_policy": "pre-optimization anchor-segment reference only; final uses original least_squares logic",
        "raw_vs_final_max_abs_z_m": _max_abs_delta(z_raw, z_final),
        "optimizer_style_residual_report": residual_report,
        "vlm_gate_summary": vlm_gate_summary,
    }
    write_json(paths["stage4_metrics"], metrics)
    return metrics
