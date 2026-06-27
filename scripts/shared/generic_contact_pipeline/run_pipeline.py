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
from scripts.shared.generic_contact_pipeline.core.base.io import write_json  # noqa: E402
from scripts.shared.generic_contact_pipeline.core.base.schema import stage_paths  # noqa: E402
from scripts.shared.generic_contact_pipeline.core.gates.vlm_provider import load_vlm_provider  # noqa: E402
from scripts.shared.generic_contact_pipeline.core.gates.vlm_gates import write_stage_gates  # noqa: E402
from scripts.shared.generic_contact_pipeline.stages.analysis import stage_loss_analysis  # noqa: E402
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


def run_case(case_name: str, from_stage: str, to_stage: str, *, args: argparse.Namespace) -> dict[str, object]:
    profile = with_runtime_overrides(load_case_profile(case_name), result_name=args.result_name or None, ablation_flags=args.ablation_flag)
    stage_results = []
    for stage_name, fn in selected_stages(from_stage, to_stage):
        print(f"[{case_name}] {stage_name}", flush=True)
        if stage_name == "stage-1":
            result = fn(profile, args.llm_mode)
        else:
            result = fn(profile)
        vlm_result = _run_stage_vlm(profile, stage_name, args)
        vlm_gate_result = None
        if vlm_result is not None:
            vlm_gate_result = write_stage_gates(profile, stage_name)
            print(
                f"[{case_name}] {stage_name} vlm={vlm_result.get('mode')} "
                f"decision={vlm_result.get('decision')} gates={vlm_result.get('gate_counts', {})}",
                flush=True,
            )
        if args.vlm_blocking and vlm_result and vlm_result.get("blocking"):
            stage_results.append({"stage": stage_name, "result": result, "vlm_verification": vlm_result, "vlm_gate": vlm_gate_result})
            manifest = {
                "case_name": case_name,
                "profile": profile.data,
                "result_dir": str(profile.result_dir),
                "render_dir": str(profile.render_dir),
                "from_stage": from_stage,
                "to_stage": to_stage,
                "vlm_mode": args.vlm_mode,
                "llm_mode": args.llm_mode,
                "vlm_blocked_at_stage": stage_name,
                "stage_results": stage_results,
            }
            write_json(stage_paths(profile)["pipeline_manifest"], manifest)
            raise RuntimeError(f"VLM blocked {case_name} at {stage_name}: {vlm_result.get('decision')}")
        stage_results.append({"stage": stage_name, "result": result, "vlm_verification": vlm_result, "vlm_gate": vlm_gate_result})
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
        "stage_results": stage_results,
    }
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
    args = ap.parse_args()
    if args.skip_vlm:
        args.vlm_mode = "none"

    cases = available_cases() if args.case == "all" else [args.case]
    failed = False
    for case in cases:
        try:
            run_case(case, args.from_stage, args.to_stage, args=args)
        except Exception as exc:
            failed = True
            print(f"[{case}] FAILED: {exc}", file=sys.stderr)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
