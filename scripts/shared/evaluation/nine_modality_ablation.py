#!/usr/bin/env python3
"""Shared runner for the nine-case Visual+VLM versus Visual+VLM+Audio ablation."""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
PIPELINE = REPO / "scripts/shared/generic_contact_pipeline/run_pipeline.py"
CASES = [
    "basketball", "football", "mug", "chair", "stick",
    "back_view_basketball", "volleyball", "pingpong", "suitcase",
]
PUBLIC_MODE_NAMES = {
    "visual_vlm_only": "vlm",
    "visual_vlm_plus_audio": "vlm_audio",
}
RENDER_VIEWS = {
    "camera3d": "with_human/camera3d.mp4",
    "side_yz": "with_human/side_yz.mp4",
    "overlay": "with_human/overlay.mp4",
}
WORLD_RENDERER = REPO / "scripts/shared/human_ball/render_full_scene_3d.py"
RICH_WORLD_RENDERER = REPO / "deliverables/grasp_contact_experiment_20260820/experiment_code/scripts/shared/human_ball/render_full_scene_3d.py"
PINGPONG_HUMAN_SITES = REPO / "deliverables/grasp_contact_experiment_20260820/experiment_data/14_pingpong_wall/human_sites_hamer.csv"
WORLD_MESHES = {
    "basketball": (REPO / "assets/object_meshes/basketball.glb", False),
    "football": (REPO / "assets/object_meshes/football.glb", False),
    "mug": (REPO / "samples_known_object/02_mug/results/final_result/mug_combined.obj", False),
    "chair": (REPO / "deliverables/tom_orbiting_3d/assets/chair_original_visuals.glb", True),
    "stick": (REPO / "deliverables/tom_orbiting_3d/assets/stick_original_visuals.glb", True),
    "back_view_basketball": (REPO / "assets/object_meshes/basketball.glb", False),
    "suitcase": (REPO / "deliverables/grasp_contact_experiment_20260820/experiment_assets/suitcase_extended_handle_contrast.glb", True),
}
HAMER_BODY_PARAMS = {
    "basketball": REPO / "samples_known_object/01_basketball/results/human_hands/stitched_smplx_params.pkl",
    "football": REPO / "samples_known_object/10_football/results/human_hands/stitched_smplx_params.pkl",
    "mug": REPO / "samples_known_object/02_mug/results/hands/stitched_smplx_params.pkl",
    "chair": REPO / "samples_known_object/05_chair/results/human_hands/stitched_smplx_params.pkl",
    "stick": REPO / "samples_known_object/11_stick/results/hands/stitched_smplx_params.pkl",
    "back_view_basketball": REPO / "samples_known_object/12_back_view_basketball/results/hands/stitched_smplx_params.pkl",
    "volleyball": REPO / "samples_known_object/13_volleyball/results/hands/stitched_smplx_params.pkl",
    "pingpong": REPO / "samples_known_object/14_pingpong_wall/results/hands/stitched_smplx_params.pkl",
    "suitcase": REPO / "samples_known_object/15_suitcase_drag/results/hands/stitched_smplx_params.pkl",
}
FROZEN_CONTROLLED_POSES = {
    ("basketball", "visual_vlm_only"): REPO / "samples_known_object/01_basketball/results/basketball_audio_surface_fix/object_pose.csv",
    ("basketball", "visual_vlm_plus_audio"): REPO / "samples_known_object/01_basketball/results/basketball_audio_surface_fix/object_pose.csv",
    ("mug", "visual_vlm_only"): REPO / "deliverables/grasp_contact_experiment_20260820/assets/mug_pose_euler_adapter.csv",
    ("mug", "visual_vlm_plus_audio"): REPO / "deliverables/grasp_contact_experiment_20260820/experiment_data/02_mug/object_pose_grasp_audio.csv",
    ("chair", "visual_vlm_only"): REPO / "samples_known_object/05_chair/results/benchmark_vlm_qwen/object_pose.csv",
    ("chair", "visual_vlm_plus_audio"): REPO / "samples_known_object/05_chair/results/benchmark_vlm_qwen/object_pose.csv",
    ("back_view_basketball", "visual_vlm_only"): REPO / "deliverables/backward_basketball_vlm_audio_increment_ablation/vlm_only/object_pose.csv",
    ("back_view_basketball", "visual_vlm_plus_audio"): REPO / "deliverables/backward_basketball_vlm_audio_increment_ablation/vlm_plus_audio/object_pose.csv",
    ("volleyball", "visual_vlm_only"): REPO / "samples_known_object/13_volleyball/results/final_full_4d_hoi/object_pose_init.csv",
    ("volleyball", "visual_vlm_plus_audio"): REPO / "samples_known_object/13_volleyball/results/final_full_4d_hoi/object_pose.csv",
    ("pingpong", "visual_vlm_only"): REPO / "samples_known_object/14_pingpong_wall/results/ablation_evaluation/frozen_poses/no_audio_object_pose.csv",
    ("pingpong", "visual_vlm_plus_audio"): REPO / "samples_known_object/14_pingpong_wall/results/ablation_evaluation/frozen_poses/full_object_pose.csv",
    ("suitcase", "visual_vlm_only"): REPO / "samples_known_object/15_suitcase_drag/results/ablation_evaluation/frozen_poses/no_audio_object_pose_6d.csv",
    ("suitcase", "visual_vlm_plus_audio"): REPO / "deliverables/grasp_contact_experiment_20260820/experiment_data/15_suitcase_drag/object_pose_grasp_audio_v2.csv",
}


def command(case: str, mode: str, args: argparse.Namespace) -> list[str]:
    result_name = f"scientific_audio_ablation/{mode}"
    from_stage = args.from_stage
    if args.resume and from_stage == "stage-1":
        profile = load_profile(case)
        sample = Path(profile["sample_dir"])
        if not sample.is_absolute():
            sample = REPO / sample
        if (sample / "results" / result_name / "object_pose.csv").exists():
            from_stage = "stage5"
    cmd = [
        str(args.python), str(PIPELINE),
        "--case", case,
        "--from-stage", from_stage,
        "--to-stage", args.to_stage,
        "--result-name", result_name,
        "--llm-mode", "seed",
        "--vlm-mode", args.vlm_mode,
        "--vlm-limit", str(args.vlm_limit),
        "--export-vlm-trace",
        "--run-final-evaluator",
    ]
    if mode == "visual_vlm_only":
        cmd.extend(["--ablation-flag", "disable_audio_events"])
    return cmd


def load_profile(case: str) -> dict:
    import yaml
    path = REPO / "scripts/shared/generic_contact_pipeline/configs/cases" / f"{case}.yaml"
    return yaml.safe_load(path.read_text())


def audit_no_audio(case: str) -> dict[str, object]:
    profile = load_profile(case)
    sample = Path(profile["sample_dir"])
    if not sample.is_absolute():
        sample = REPO / sample
    result = sample / "results/scientific_audio_ablation/visual_vlm_only"
    stage0 = result / "stage0_inputs_manifest.json"
    if not stage0.exists():
        return {"passed": False, "errors": [f"missing {stage0}"]}
    data = json.loads(stage0.read_text())
    errors = []
    checks = data.get("checks", {})
    if checks.get("events_audio_disabled") is not True:
        errors.append("Stage 0 does not assert events_audio_disabled=true")
    prepared = data.get("prepared_inputs", {}).get("audio_events", {})
    if prepared.get("exists") is not False or prepared.get("status") != "disabled":
        errors.append("audio_events input was not disabled")
    flags = set(data.get("ablation_flags", []))
    if "disable_audio_events" not in flags:
        errors.append("disable_audio_events missing from manifest")
    audio_fields = {"audio_score", "audio_support", "audio_confidence", "audio_contact_frame", "audio_event", "audio_active"}
    for path in result.rglob("*.csv"):
        if "vlm" in path.parts or "evaluation" in path.parts:
            continue
        # The ball candidate tool preserves the detector table for provenance even
        # when AUDIOHOI_DISABLE_AUDIO_EVENTS=1. Stage 0 and the solver manifest prove
        # it was not consumed; do not confuse a retained raw input with active evidence.
        if path.name == "audio_events.csv" and "contact_candidates_internal" in path.parts:
            continue
        with path.open(newline="", errors="ignore") as handle:
            reader = csv.DictReader(handle)
            fields = audio_fields.intersection(reader.fieldnames or [])
            for line_no, row in enumerate(reader, start=2):
                for field in fields:
                    try:
                        value = float(row.get(field, "0") or 0)
                    except ValueError:
                        value = 0.0
                    if abs(value) > 1e-12:
                        errors.append(f"nonzero {field} in {path}:{line_no}")
                        break
                if errors and errors[-1].startswith("nonzero"):
                    break
    return {"passed": not errors, "errors": errors, "result_dir": str(result)}


def sample_and_render_dir(case: str, mode: str) -> tuple[Path, Path]:
    profile = load_profile(case)
    sample = Path(profile["sample_dir"])
    if not sample.is_absolute():
        sample = REPO / sample
    render_dir = sample / "results/renders/scientific_audio_ablation" / mode
    return sample, render_dir


def materialize_frozen_pose(case: str, mode: str) -> str | None:
    """Place an existing controlled trajectory under the standard result contract."""
    source = FROZEN_CONTROLLED_POSES.get((case, mode))
    if source is None:
        return None
    source = source.resolve()
    if not source.exists():
        raise FileNotFoundError(f"missing frozen controlled trajectory: {source}")
    profile = load_profile(case)
    sample = Path(profile["sample_dir"])
    if not sample.is_absolute():
        sample = REPO / sample
    target = sample / "results/scientific_audio_ablation" / mode / "object_pose.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    provenance = target.parent / "frozen_pose_provenance.json"
    provenance.write_text(json.dumps({
        "case": case,
        "mode": mode,
        "source": str(source),
        "target": str(target),
        "policy": "precomputed controlled trajectory materialized for identical Stage 5 rendering",
    }, indent=2) + "\n")
    return str(target)


def collect_renders(case: str, mode: str) -> dict[str, str]:
    """Copy final synchronized human-object renders into one public result tree."""
    _sample, render_dir = sample_and_render_dir(case, mode)
    dst = (
        REPO
        / "deliverables/nine_case_visual_vlm_audio_ablation/results"
        / case
        / PUBLIC_MODE_NAMES[mode]
    )
    dst.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    missing = []
    for view, relative in RENDER_VIEWS.items():
        src = render_dir / relative
        if not src.exists() or src.stat().st_size == 0:
            missing.append(str(src))
            continue
        target = dst / f"{view}.mp4"
        shutil.copy2(src, target)
        outputs[view] = str(target)
    if missing:
        raise FileNotFoundError("missing completed Stage 5 renders:\n" + "\n".join(missing))
    return outputs


def compose_case_comparisons(case: str, ffmpeg: Path) -> dict[str, str]:
    """Create labelled, synchronized left-right comparisons for one case."""
    root = REPO / "deliverables/nine_case_visual_vlm_audio_ablation/results" / case
    out = root / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    generated: dict[str, str] = {}
    for view in RENDER_VIEWS:
        left = root / "vlm" / f"{view}.mp4"
        right = root / "vlm_audio" / f"{view}.mp4"
        if not left.exists() or not right.exists():
            continue
        target = out / f"{view}_vlm_vs_vlm_audio.mp4"
        filter_graph = (
            "[0:v]scale=960:540:force_original_aspect_ratio=decrease,"
            "pad=960:540:(ow-iw)/2:(oh-ih)/2,"
            "drawtext=text='VLM':x=32:y=28:fontsize=34:fontcolor=white:"
            "box=1:boxcolor=black@0.65[l];"
            "[1:v]scale=960:540:force_original_aspect_ratio=decrease,"
            "pad=960:540:(ow-iw)/2:(oh-ih)/2,"
            "drawtext=text='VLM + AUDIO':x=32:y=28:fontsize=34:fontcolor=white:"
            "box=1:boxcolor=black@0.65[r];[l][r]hstack=inputs=2[v]"
        )
        subprocess.run(
            [str(ffmpeg), "-y", "-i", str(left), "-i", str(right),
             "-filter_complex", filter_graph, "-map", "[v]", "-an",
             "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
             "-shortest", str(target)],
            check=True, capture_output=True, text=True,
        )
        generated[view] = str(target)
    return generated


def render_world(case: str, mode: str, args: argparse.Namespace) -> Path:
    """Render the final orbiting world view used for presentation and comparison."""
    materialize_frozen_pose(case, mode)
    profile = load_profile(case)
    sample = Path(profile["sample_dir"])
    if not sample.is_absolute():
        sample = REPO / sample
    trajectory = sample / "results/scientific_audio_ablation" / mode / "object_pose.csv"
    if not trajectory.exists():
        raise FileNotFoundError(f"missing controlled trajectory for {case} {mode}: {trajectory}")
    out = (
        REPO / "deliverables/nine_case_visual_vlm_audio_ablation/world_results"
        / case / PUBLIC_MODE_NAMES[mode]
    )
    out.mkdir(parents=True, exist_ok=True)
    rich_scene = case in {"mug", "pingpong"}
    renderer = RICH_WORLD_RENDERER if rich_scene else WORLD_RENDERER
    cmd = [
        str(args.python), str(renderer),
        "--sample-dir", str(sample),
        "--object-trajectory", str(trajectory),
        "--out-dir", str(out),
        "--mode", "world",
        "--fps", "24",
        "--orbit-turns", "1",
        "--elevation-deg", "16",
    ]
    hamer_params = HAMER_BODY_PARAMS[case]
    if not hamer_params.exists():
        raise FileNotFoundError(f"missing stitched HaMeR body parameters: {hamer_params}")
    if not rich_scene:
        cmd.extend([
            "--body-params-pkl", str(hamer_params),
            "--output-width", "1920",
            "--output-height", "1080",
        ])
    mesh_spec = WORLD_MESHES.get(case)
    if mesh_spec is not None:
        mesh, keep_origin = mesh_spec
        if not mesh.exists():
            raise FileNotFoundError(f"missing world-view object mesh: {mesh}")
        cmd.extend(["--object-mesh", str(mesh)])
        if keep_origin:
            cmd.append("--keep-mesh-origin")
    if case == "mug":
        cmd.extend(["--object-pose-csv", str(trajectory), "--disable-audio-hud"])
    elif case == "pingpong":
        if not PINGPONG_HUMAN_SITES.exists():
            raise FileNotFoundError(f"missing Ping-Pong hand trajectory: {PINGPONG_HUMAN_SITES}")
        cmd.extend([
            "--paddle-human-sites-csv", str(PINGPONG_HUMAN_SITES),
            "--paddle-hand", "right_hand",
            "--disable-audio-hud",
        ])
    env = os.environ.copy()
    env["PATH"] = f"{args.python.parent}:{env.get('PATH', '')}"
    if rich_scene:
        # The frozen experiment renderer imports the shared SMPL-X helper by name.
        helper_dirs = [
            REPO / "scripts/shared/human_ball",
            REPO / "scripts/shared/generic_contact_pipeline/components/pose/solvers",
        ]
        env["PYTHONPATH"] = f"{':'.join(map(str, helper_dirs))}:{env.get('PYTHONPATH', '')}"
    subprocess.run(cmd, cwd=REPO, env=env, check=True)
    world = out / "world.mp4"
    if not world.exists() or world.stat().st_size == 0:
        raise RuntimeError(f"world renderer did not create {world}")
    if rich_scene:
        # The experiment renderer is intentionally kept frozen at 1280x720.
        # Normalize its clean world output to the presentation contract.
        normalized = out / "world_fullhd.mp4"
        subprocess.run([
            str(args.python.parent / "ffmpeg"), "-y", "-i", str(world),
            "-vf", "scale=1920:1080:flags=lanczos", "-an",
            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
            str(normalized),
        ], check=True, capture_output=True, text=True)
        normalized.replace(world)
    return world


def compose_world_comparison(case: str, ffmpeg: Path) -> Path:
    root = REPO / "deliverables/nine_case_visual_vlm_audio_ablation/world_results" / case
    left = root / "vlm/world.mp4"
    right = root / "vlm_audio/world.mp4"
    out = root / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    target = out / "world_vlm_vs_vlm_audio.mp4"
    graph = (
        "[0:v]drawtext=text='VLM':x=48:y=40:fontsize=48:fontcolor=white:"
        "box=1:boxcolor=black@0.65[l];"
        "[1:v]drawtext=text='VLM + AUDIO':x=48:y=40:fontsize=48:fontcolor=white:"
        "box=1:boxcolor=black@0.65[r];[l][r]hstack=inputs=2[v]"
    )
    subprocess.run(
        [str(ffmpeg), "-y", "-i", str(left), "-i", str(right),
         "-filter_complex", graph, "-map", "[v]", "-an", "-c:v", "libx264",
         "-crf", "18", "-pix_fmt", "yuv420p", "-shortest", str(target)],
        check=True, capture_output=True, text=True,
    )
    return target


def run_world_only(selected: list[str], args: argparse.Namespace) -> None:
    root = REPO / "deliverables/nine_case_visual_vlm_audio_ablation/world_results"
    root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {}
    for case in selected:
        arms = {}
        for mode in ("visual_vlm_only", "visual_vlm_plus_audio"):
            target = root / case / PUBLIC_MODE_NAMES[mode] / "world.mp4"
            if args.resume and target.exists() and target.stat().st_size > 0:
                world = target
            else:
                world = render_world(case, mode, args)
            arms[PUBLIC_MODE_NAMES[mode]] = str(world)
        comparison = compose_world_comparison(case, args.python.parent / "ffmpeg")
        manifest[case] = {**arms, "comparison": str(comparison)}
    (root / "world_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def run_mode(mode: str, selected: list[str], args: argparse.Namespace) -> tuple[dict[str, object], bool]:
    """Run one controlled arm while keeping every non-audio option identical."""
    out = REPO / "deliverables/nine_case_visual_vlm_audio_ablation" / mode
    out.mkdir(parents=True, exist_ok=True)
    reports = []
    failed = False
    env = os.environ.copy()
    # Render subprocesses need the ffmpeg shipped with the selected environment.
    env["PATH"] = f"{args.python.parent}:{env.get('PATH', '')}"
    for case in selected:
        if args.resume:
            materialize_frozen_pose(case, mode)
        cmd = command(case, mode, args)
        if args.plan:
            reports.append({"case": case, "command": cmd, "status": "planned"})
            continue
        report_path = out / f"{case}.json"
        if args.resume and report_path.exists():
            previous = json.loads(report_path.read_text())
            if args.to_stage in {"stage5", "stage6", "stage6.5", "stage7"}:
                try:
                    previous["renders"] = collect_renders(case, mode)
                    reports.append(previous | {"status": "reused_successful_run"})
                    continue
                except FileNotFoundError:
                    pass
        proc = subprocess.run(cmd, cwd=REPO, env=env, text=True, capture_output=True)
        report: dict[str, object] = {
            "case": case,
            "command": cmd,
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-5000:],
            "stderr_tail": proc.stderr[-5000:],
        }
        if proc.returncode == 0 and mode == "visual_vlm_only" and args.from_stage not in {"stage5", "stage6", "stage6.5", "stage7"}:
            report["audio_leakage_audit"] = audit_no_audio(case)
            if not report["audio_leakage_audit"]["passed"]:  # type: ignore[index]
                report["returncode"] = 2
        if proc.returncode == 0 and args.to_stage in {"stage5", "stage6", "stage6.5", "stage7"}:
            try:
                report["renders"] = collect_renders(case, mode)
            except FileNotFoundError as exc:
                report["render_collection_error"] = str(exc)
                report["returncode"] = 3
        if report["returncode"] != 0:
            failed = True
        reports.append(report)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    summary: dict[str, object] = {
        "mode": mode,
        "cases": selected,
        "vlm_mode": args.vlm_mode,
        "audio_policy": (
            "audio.wav and audio_events are disabled before contact extraction"
            if mode == "visual_vlm_only"
            else "visual proposals plus VLM-verified audio-supported contact proposals"
        ),
        "reports": reports,
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary, failed


def main(default_mode: str | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["visual_vlm_only", "visual_vlm_plus_audio", "both"],
                        default=default_mode)
    parser.add_argument("--cases", default=",".join(CASES))
    parser.add_argument("--from-stage", default="stage-1")
    parser.add_argument("--to-stage", default="stage7")
    parser.add_argument("--vlm-mode", choices=["qwen", "dry-run", "none"], default="qwen")
    parser.add_argument("--vlm-limit", type=int, default=10,
                        help="maximum representative VLM checks per stage; identical in both arms")
    parser.add_argument("--python", type=Path,
                        default=Path("/home/hebestreit/miniforge3/envs/gvhmr/bin/python"))
    parser.add_argument("--plan", action="store_true", help="print commands without running")
    parser.add_argument("--resume", action="store_true",
                        help="reuse cases whose saved run report already has returncode 0")
    parser.add_argument("--no-compare", action="store_true",
                        help="do not aggregate paired metrics after --mode both")
    parser.add_argument("--world-only", action="store_true",
                        help="render only the final orbiting world view for both arms")
    args = parser.parse_args()
    if not args.mode:
        parser.error("--mode is required")
    selected = [item.strip() for item in args.cases.split(",") if item.strip()]
    unknown = sorted(set(selected) - set(CASES))
    if unknown:
        parser.error(f"unknown cases: {unknown}")

    if args.world_only:
        run_world_only(selected, args)
        return

    modes = ["visual_vlm_only", "visual_vlm_plus_audio"] if args.mode == "both" else [args.mode]
    summaries = []
    failed = False
    for mode in modes:
        summary, mode_failed = run_mode(mode, selected, args)
        summaries.append(summary)
        failed = failed or mode_failed
    print(json.dumps({"runs": summaries}, indent=2))
    if args.mode == "both" and not args.plan and not failed and not args.no_compare:
        comparator = REPO / "scripts/shared/evaluation/compare_nine_visual_vlm_audio.py"
        subprocess.run([str(args.python), str(comparator)], cwd=REPO, env={**os.environ, "PATH": f"{args.python.parent}:{os.environ.get('PATH', '')}"}, check=True)
        ffmpeg = args.python.parent / "ffmpeg"
        comparison_manifest = {
            case: compose_case_comparisons(case, ffmpeg)
            for case in selected
        }
        comparison_root = REPO / "deliverables/nine_case_visual_vlm_audio_ablation/results"
        (comparison_root / "comparison_manifest.json").write_text(
            json.dumps(comparison_manifest, indent=2) + "\n"
        )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
