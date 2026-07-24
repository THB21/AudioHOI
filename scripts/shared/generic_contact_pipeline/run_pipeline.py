#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

if __package__ in {None, ""}:
    REPO = Path(__file__).resolve().parents[3]
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))

from scripts.shared.generic_contact_pipeline.core.base.config import available_cases, load_case_profile, with_runtime_overrides  # noqa: E402
from scripts.shared.generic_contact_pipeline.core.base.ablation import describe_ablation_mechanisms, validate_ablation_flags  # noqa: E402
from scripts.shared.generic_contact_pipeline.core.base.io import read_csv, write_json  # noqa: E402
from scripts.shared.generic_contact_pipeline.core.base.schema import stage_paths  # noqa: E402
from scripts.shared.generic_contact_pipeline.core.evaluation.benchmark import run_benchmark as run_result_benchmark  # noqa: E402
from scripts.shared.generic_contact_pipeline.core.evaluation.final_hoi.ablation_registry import MATERIALIZED_DEFAULT_VARIANTS  # noqa: E402
from scripts.shared.generic_contact_pipeline.core.evaluation.final_hoi.ablation_runner import run_ablation_evaluation  # noqa: E402
from scripts.shared.generic_contact_pipeline.core.evaluation.final_hoi.summary_writer import run_unified_final_evaluation  # noqa: E402
from scripts.shared.generic_contact_pipeline.core.evaluation.vlm_trace import export_vlm_trace  # noqa: E402
from scripts.shared.generic_contact_pipeline.core.gates.stage_audit import write_stage_audit  # noqa: E402
from scripts.shared.generic_contact_pipeline.core.gates.vlm_provider import load_vlm_provider  # noqa: E402
from scripts.shared.generic_contact_pipeline.core.gates.vlm_gates import write_stage_gates  # noqa: E402
from scripts.shared.generic_contact_pipeline.core.provenance.attempts import StageAttempt  # noqa: E402
from scripts.shared.generic_contact_pipeline.components.mainline import contact_anchor, pose_init, sequence_refine  # noqa: E402
from scripts.shared.generic_contact_pipeline.stages.analysis import stage_llm_csv_audit, stage_loss_analysis  # noqa: E402
from scripts.shared.generic_contact_pipeline.stages.gates import stage_vlm_qwen, stage_vlm_verify  # noqa: E402
from scripts.shared.generic_contact_pipeline.stages.main import (  # noqa: E402
    stage_minus1_llm_prior,
    stage0_preprocess,
    stage1_observation,
    stage2_contact,
    stage3_initial_pose,
    stage4_contact_refine,
    stage5_render,
    stage6_compare,
)


STAGES = [
    ("stage-1", stage_minus1_llm_prior.run),
    ("stage0", stage0_preprocess.run),
    ("stage1", stage1_observation.run),
    ("stage2", stage2_contact.run),
    ("stage3", stage3_initial_pose.run),
    ("stage4", stage4_contact_refine.run),
    ("stage5", stage5_render.run),
    ("stage6", stage6_compare.run),
    ("stage6.5", stage_llm_csv_audit.run),
    ("stage7", stage_loss_analysis.run),
]


def selected_stages(from_stage: str, to_stage: str):
    names = [name for name, _fn in STAGES]
    if from_stage not in names:
        raise ValueError(f"Unknown from-stage {from_stage}; expected one of {names}")
    if to_stage not in names:
        raise ValueError(f"Unknown to-stage {to_stage}; expected one of {names}")
    i0 = names.index(from_stage)
    i1 = names.index(to_stage)
    if i0 > i1:
        raise ValueError(f"from-stage {from_stage} must be <= to-stage {to_stage}")
    return STAGES[i0 : i1 + 1]


def _run_stage_vlm(profile, stage_name: str, args: argparse.Namespace) -> dict[str, object] | None:
    if args.vlm_mode == "none" or stage_name not in stage_vlm_verify.STAGE_QUERY_TYPES:
        return None
    if args.vlm_mode == "dry-run":
        return stage_vlm_verify.run(profile, stage_name)
    if args.vlm_mode != "qwen":
        raise ValueError(f"Unknown VLM mode {args.vlm_mode!r}")

    provider = load_vlm_provider(args.vlm_provider)
    provider.apply_env_defaults()
    if not provider.is_current_python():
        script = Path(__file__).resolve().parent / "stages" / "gates" / "stage_vlm_qwen.py"
        cmd = [
            str(provider.python),
            str(script),
            "--case",
            profile.case_name,
            "--stage",
            stage_name,
            "--result-name",
            profile.result_name,
            "--provider",
            provider.name,
            "--model-id",
            args.vlm_model_id or provider.model_id,
            "--local-dir",
            args.vlm_local_dir or (str(provider.local_dir) if provider.local_dir else ""),
            "--device-map",
            args.vlm_device_map or provider.device_map,
            "--max-new-tokens",
            str(args.vlm_max_new_tokens or provider.max_new_tokens),
            "--resize-max",
            str(args.vlm_resize_max if args.vlm_resize_max is not None else provider.resize_max),
            "--limit",
            str(args.vlm_limit),
        ]
        if provider.load_4bit if args.vlm_load_4bit is None else args.vlm_load_4bit:
            cmd.append("--load-4bit")
        else:
            cmd.append("--no-load-4bit")
        if args.vlm_no_refresh_queries:
            cmd.append("--no-refresh-queries")
        env = os.environ.copy()
        env.update(provider.env)
        subprocess.run(cmd, cwd=Path(__file__).resolve().parents[3], env=env, check=True)
        out_dir = stage_paths(profile)["vlm_dir"] / stage_name
        decision_path = out_dir / ("stage_decision_qwen_debug.json" if args.vlm_limit > 0 else "stage_decision.json")
        if not decision_path.exists():
            raise FileNotFoundError(f"Qwen VLM finished but did not write {decision_path}")
        with decision_path.open() as f:
            return json.load(f)
    qwen_args = SimpleNamespace(
        provider=provider.name,
        model_id=args.vlm_model_id or provider.model_id,
        local_dir=args.vlm_local_dir or (str(provider.local_dir) if provider.local_dir else ""),
        device_map=args.vlm_device_map or provider.device_map,
        load_4bit=provider.load_4bit if args.vlm_load_4bit is None else args.vlm_load_4bit,
        max_new_tokens=args.vlm_max_new_tokens or provider.max_new_tokens,
        resize_max=args.vlm_resize_max if args.vlm_resize_max is not None else provider.resize_max,
        limit=args.vlm_limit,
        refresh_queries=not args.vlm_no_refresh_queries,
    )
    return stage_vlm_qwen.evaluate_stage(profile, stage_name, qwen_args)


def _csv_count(path: Path) -> int:
    return len(read_csv(path)) if path.exists() else 0


def _stage_artifact_summary(root: Path) -> dict[str, object]:
    stages: dict[str, dict[str, object]] = {}
    total_files = 0
    total_rows = 0
    for stage_dir in sorted([p for p in root.iterdir() if p.is_dir()]) if root.exists() else []:
        item: dict[str, object] = {"files": 0, "csv_rows": {}}
        for path in sorted(stage_dir.rglob("*")):
            if not path.is_file():
                continue
            total_files += 1
            item["files"] = int(item["files"]) + 1
            if path.suffix.lower() == ".csv":
                rows = _csv_count(path)
                total_rows += rows
                item["csv_rows"][path.name] = rows  # type: ignore[index]
        stages[stage_dir.name] = item
    return {
        "exists": root.exists(),
        "path": str(root),
        "stage_count": len(stages),
        "file_count": total_files,
        "csv_row_count": total_rows,
        "stages": stages,
    }


def _generic_mainline_manifest(profile) -> dict[str, object]:
    paths = stage_paths(profile)
    artifacts = {
        "stage0_inputs_manifest": paths["stage0_inputs_manifest"],
        "object_observations": paths["object_observations"],
        "object_correspondence": paths["object_correspondence"],
        "object_surface_points": paths["object_surface_points"],
        "object_semantic_points": paths["object_semantic_points"],
        "line_correspondence": paths["line_correspondence"],
        "line_observations": paths["line_observations"],
        "contact_candidates": paths["contact_candidates"],
        "anchor_state": paths["anchor_state"],
        "object_pose_init": paths["object_pose_init"],
        "object_pose_pre_smooth": paths["object_pose_pre_smooth"],
        "motion_regime": paths["motion_regime"],
        "physical_smooth_residuals": paths["physical_smooth_residuals"],
        "pose_jump_audit": paths["pose_jump_audit"],
        "optimizer_decisions": paths["optimizer_decisions"],
        "object_pose": paths["object_pose"],
        "object_contact_points": paths["object_contact_points"],
        "stage_audit_gates": paths["stage_audit_dir"] / "stage_audit_gates.csv",
        "stage_audit_latest": paths["stage_audit_dir"] / "stage_audit_latest.json",
    }
    out: dict[str, object] = {
        "enabled": True,
        "contract": "fixed_preprocess_observation_contact_anchor_se3_init_sequence_optimizer",
        "object_family": profile.data.get("object_family", ""),
        "legacy_yaml_selectors": {
            "observation_model": profile.data.get("observation_model", ""),
            "contact_policy": profile.data.get("contact_policy", ""),
            "pose_model": profile.data.get("pose_model", ""),
            "refinement_policy": profile.data.get("refinement_policy", []),
            "role": "stage1-3 are normalized through generic contracts; refinement_policy is ignored by Stage4 except the generic sequence optimizer marker",
        },
        "artifacts": {
            name: {"path": str(path), "exists": path.exists(), "rows": _csv_count(path)}
            for name, path in artifacts.items()
        },
        "actual_vlm_artifacts": _stage_artifact_summary(paths["vlm_dir"]),
        "actual_stage_audit_artifacts": _stage_artifact_summary(paths["stage_audit_dir"]),
    }
    audit_path = paths["pose_jump_audit"]
    if audit_path.exists():
        audit = read_csv(audit_path)
        out["pose_jump_audit"] = {
            "visual_spike_frames": [int(float(r["frame"])) for r in audit if r.get("visual_spike") == "1"],
            "contact_spike_frames": [int(float(r["frame"])) for r in audit if r.get("contact_spike") == "1"],
            "smoothness_spike_frames": [int(float(r["frame"])) for r in audit if r.get("smoothness_spike") == "1"],
            "propagated_frames": [int(float(r["frame"])) for r in audit if r.get("freeze_or_propagate") == "propagate"],
        }
    residual_path = paths["physical_smooth_residuals"]
    if residual_path.exists():
        residuals = read_csv(residual_path)
        out["residual_switches"] = {
            "visual_enabled_rows": sum(1 for r in residuals if r.get("residual_visual_enabled") == "1"),
            "contact_enabled_rows": sum(1 for r in residuals if r.get("residual_contact_enabled") == "1"),
            "smooth_enabled_rows": sum(1 for r in residuals if r.get("residual_smooth_enabled") == "1"),
            "propagated_rows": sum(1 for r in residuals if r.get("freeze_or_propagate") == "propagate"),
        }
    return out


def _refresh_generic_mainline_after_vlm(profile, stage_name: str, result: dict[str, object]) -> dict[str, object]:
    refreshed = dict(result)
    paths = stage_paths(profile)
    if stage_name == "stage2":
        refreshed["generic_contact_anchor_mainline"] = contact_anchor.build(profile)
        write_json(paths["stage2_metrics"], refreshed)
    elif stage_name == "stage4":
        gate_path = paths["vlm_dir"] / stage_name / "vlm_gates.csv"
        if gate_path.exists() and not any(row.get("is_effective") == "1" for row in read_csv(gate_path)):
            return refreshed
        smooth_result = sequence_refine.apply(profile)
        components = [
            item
            for item in list(refreshed.get("components", []))
            if not (isinstance(item, dict) and item.get("component") == "generic_sequence_se3_mainline")
        ]
        components.append(smooth_result)
        refreshed["components"] = components
        write_json(paths["stage4_metrics"], refreshed)
    return refreshed


def _repair_after_stage_audit(profile, stage_name: str, result: dict[str, object], audit_decision: dict[str, object]) -> dict[str, object]:
    if not audit_decision.get("rerun_stage"):
        return result
    repaired = dict(result)
    if stage_name == "stage2":
        repaired["stage_audit_repair"] = contact_anchor.build(profile)
    elif stage_name == "stage3":
        repaired["stage_audit_repair"] = pose_init.build(profile)
    elif stage_name == "stage4":
        repaired["stage_audit_repair"] = sequence_refine.apply(profile)
    else:
        repaired["stage_audit_repair"] = {
            "component": "stage_audit_repair",
            "enabled": False,
            "reason": f"no automatic repair registered for {stage_name}",
        }
    return repaired


def _attempt_summary(
    result: dict[str, object],
    vlm_result: dict[str, object] | None,
    stage_audit_result: dict[str, object],
) -> dict[str, object]:
    return {
        "stage_status": result.get("status", result.get("decision", "")),
        "vlm_decision": vlm_result.get("decision", "") if vlm_result else "not_run",
        "stage_audit_decision": stage_audit_result.get("decision", ""),
        "rerun_requested": bool(stage_audit_result.get("rerun_stage")),
    }


def run_case(case_name: str, from_stage: str, to_stage: str, *, args: argparse.Namespace) -> dict[str, object]:
    requested_ablation_flags = validate_ablation_flags(args.ablation_flag)
    ablation_mechanisms = describe_ablation_mechanisms(requested_ablation_flags)
    profile = with_runtime_overrides(
        load_case_profile(case_name),
        result_name=args.result_name or None,
        ablation_flags=requested_ablation_flags,
    )
    stage_results = []
    for stage_name, fn in selected_stages(from_stage, to_stage):
        print(f"[{case_name}] {stage_name}", flush=True)
        attempt = StageAttempt(
            profile,
            stage_name,
            trigger="scheduled_pipeline_stage",
            metadata={"from_stage": from_stage, "to_stage": to_stage},
        )
        attempt_finished = False
        try:
            if stage_name in {"stage-1", "stage6.5"}:
                result = fn(profile, args.llm_mode)
            else:
                result = fn(profile)
            vlm_result = _run_stage_vlm(profile, stage_name, args)
            vlm_gate_result = None
            if vlm_result is not None:
                vlm_gate_result = write_stage_gates(profile, stage_name)
                result = _refresh_generic_mainline_after_vlm(profile, stage_name, result)
                print(
                    f"[{case_name}] {stage_name} vlm={vlm_result.get('mode')} "
                    f"decision={vlm_result.get('decision')} gates={vlm_result.get('gate_counts', {})}",
                    flush=True,
                )
            stage_audit_result = write_stage_audit(profile, stage_name, llm_mode=args.llm_mode)
            if stage_audit_result.get("rerun_stage"):
                attempt.finish(
                    status="completed_rerun_requested",
                    result_summary=_attempt_summary(result, vlm_result, stage_audit_result),
                )
                attempt_finished = True
                rerun_attempt = StageAttempt(
                    profile,
                    stage_name,
                    trigger="stage_audit_rerun",
                    parent_attempt_id=attempt.attempt_id,
                    metadata={"audit_decision": stage_audit_result},
                )
                try:
                    result = _repair_after_stage_audit(profile, stage_name, result, stage_audit_result)
                    stage_audit_result = write_stage_audit(profile, stage_name, llm_mode=args.llm_mode)
                    rerun_attempt.finish(
                        result_summary=_attempt_summary(result, vlm_result, stage_audit_result)
                    )
                except Exception as exc:
                    rerun_attempt.finish(status="failed", error=exc)
                    raise
            else:
                attempt.finish(
                    result_summary=_attempt_summary(result, vlm_result, stage_audit_result)
                )
                attempt_finished = True
        except Exception as exc:
            if not attempt_finished:
                attempt.finish(status="failed", error=exc)
            raise
        if args.vlm_blocking and vlm_result and vlm_result.get("blocking"):
            stage_results.append({"stage": stage_name, "result": result, "vlm_verification": vlm_result, "vlm_gate": vlm_gate_result, "stage_audit": stage_audit_result})
            manifest = {
                "case_name": case_name,
                "profile": profile.data,
                "result_dir": str(profile.result_dir),
                "render_dir": str(profile.render_dir),
                "from_stage": from_stage,
                "to_stage": to_stage,
                "vlm_mode": args.vlm_mode,
                "llm_mode": args.llm_mode,
                "ablation_mechanisms": ablation_mechanisms,
                "vlm_blocked_at_stage": stage_name,
                "stage_results": stage_results,
            }
            write_json(stage_paths(profile)["pipeline_manifest"], manifest)
            raise RuntimeError(f"VLM blocked {case_name} at {stage_name}: {vlm_result.get('decision')}")
        stage_results.append({"stage": stage_name, "result": result, "vlm_verification": vlm_result, "vlm_gate": vlm_gate_result, "stage_audit": stage_audit_result})
    manifest = {
        "case_name": case_name,
        "profile": profile.data,
        "result_dir": str(profile.result_dir),
        "render_dir": str(profile.render_dir),
        "from_stage": from_stage,
        "to_stage": to_stage,
        "vlm_mode": args.vlm_mode,
        "llm_mode": args.llm_mode,
        "vlm_blocking": args.vlm_blocking,
        "ablation_mechanisms": ablation_mechanisms,
        "stage_results": stage_results,
    }
    manifest["generic_se3_mainline"] = _generic_mainline_manifest(profile)
    write_json(stage_paths(profile)["pipeline_manifest"], manifest)
    post_run: dict[str, object] = {}
    if getattr(args, "export_vlm_trace", False):
        post_run["vlm_trace"] = export_vlm_trace(profile)
    if getattr(args, "run_final_evaluator", False):
        post_run["final_hoi_evaluator"] = run_unified_final_evaluation(profile)
    if post_run:
        manifest["post_run"] = post_run
        write_json(stage_paths(profile)["pipeline_manifest"], manifest)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the generic modular AudioHOI contact pipeline.")
    ap.add_argument("--case", default="all", help="Case name or all")
    ap.add_argument("--from-stage", default="stage-1", choices=[name for name, _fn in STAGES])
    ap.add_argument("--to-stage", default="stage7", choices=[name for name, _fn in STAGES])
    ap.add_argument("--result-name", default="", help="Override case profile result_name, e.g. generic_pipeline_v2_llm_vlm_gate.")
    ap.add_argument("--ablation-flag", action="append", default=[], help="Runtime ablation flag consumed by components.")
    ap.add_argument("--llm-mode", default="seed", choices=["none", "seed", "qwen", "mistral"], help="Resolve Stage -1 HOI semantic prior.")
    ap.add_argument("--skip-vlm", action="store_true", help="Deprecated alias for --vlm-mode none.")
    ap.add_argument("--vlm-mode", default="dry-run", choices=["none", "dry-run", "qwen"], help="Stage-level VLM verification mode.")
    ap.add_argument("--vlm-blocking", action="store_true", help="Stop the pipeline when a real VLM result rejects a stage.")
    ap.add_argument("--vlm-provider", default=None, help="Provider from configs/vlm_provider.yaml. Defaults to the configured default provider.")
    ap.add_argument("--vlm-limit", type=int, default=0, help="Debug limit per stage for qwen mode. 0 means all queries.")
    ap.add_argument("--vlm-no-refresh-queries", action="store_true", help="In qwen mode, reuse existing query/evidence files.")
    ap.add_argument("--vlm-model-id", default="")
    ap.add_argument("--vlm-local-dir", default="")
    ap.add_argument("--vlm-device-map", default="")
    ap.add_argument("--vlm-load-4bit", dest="vlm_load_4bit", action="store_true", default=None)
    ap.add_argument("--vlm-no-load-4bit", dest="vlm_load_4bit", action="store_false")
    ap.add_argument("--vlm-max-new-tokens", type=int, default=0)
    ap.add_argument("--vlm-resize-max", type=int, default=None)
    ap.add_argument("--export-vlm-trace", action="store_true", help="Export standard vlm_trace/00_input..06_evaluation artifacts after the run.")
    ap.add_argument("--run-final-evaluator", action="store_true", help="Run the unified final HOI evaluator after the run.")
    ap.add_argument("--run-ablation-evaluation", action="store_true", help="Run the current final HOI ablation evaluator over materialized benchmark_* variants.")
    ap.add_argument("--run-benchmark", action="store_true", help="Legacy: aggregate benchmark_table.csv/benchmark_report.md after successful case runs.")
    ap.add_argument(
        "--benchmark-methods",
        nargs="+",
        default=["baseline_no_vlm", "vlm_gated", "no_anchor", "no_smooth", "no_static_tail"],
        help="Method labels to aggregate in --run-benchmark.",
    )
    ap.add_argument(
        "--benchmark-method-result",
        action="append",
        default=[],
        metavar="METHOD=RESULT_NAME",
        help="Evaluate a benchmark method label against a different result directory.",
    )
    args = ap.parse_args()
    if args.skip_vlm:
        args.vlm_mode = "none"

    cases = available_cases() if args.case == "all" else [args.case]
    failed = False
    successful_profiles = []
    for case in cases:
        try:
            run_case(case, args.from_stage, args.to_stage, args=args)
            successful_profiles.append(with_runtime_overrides(load_case_profile(case), result_name=args.result_name or None, ablation_flags=args.ablation_flag))
        except Exception as exc:
            failed = True
            print(f"[{case}] FAILED: {exc}", file=sys.stderr)
    if args.run_benchmark and successful_profiles:
        method_result_names: dict[str, str] = {}
        for item in args.benchmark_method_result:
            if "=" not in item:
                raise SystemExit(f"--benchmark-method-result must be METHOD=RESULT_NAME, got {item!r}")
            method, result_name = item.split("=", 1)
            method_result_names[method] = result_name
        run_result_benchmark(successful_profiles, methods=args.benchmark_methods, method_result_names=method_result_names, llm_mode=args.llm_mode)
    if args.run_ablation_evaluation and successful_profiles:
        run_ablation_evaluation(
            successful_profiles,
            variants=MATERIALIZED_DEFAULT_VARIANTS,
            output_dir=Path("final_result/evaluation/ablation"),
            require_existing=True,
        )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
