from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

if __package__ in {None, ""}:
    REPO = Path(__file__).resolve().parents[5]
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))

from scripts.shared.generic_contact_pipeline.core.base.config import available_cases, load_case_profile  # noqa: E402
from scripts.shared.generic_contact_pipeline.core.base.io import write_csv, write_json  # noqa: E402


ABLATION_VARIANTS = [
    "A0_solved_baseline",
    "A1_generic_pipeline_v1",
    "A2_v2_no_llm_prior",
    "A3_v2_llm_prior_only",
    "A4_v2_vlm_gate_only",
    "A5_v2_llm_prior_plus_vlm_gate",
    "A6_v2_no_contact_gate",
    "A7_v2_no_depth_gate",
    "A8_v2_no_anchor_propagation",
]

VARIANT_SETTINGS = {
    "A2_v2_no_llm_prior": {"llm_mode": "none", "vlm_mode": "dry-run"},
    "A3_v2_llm_prior_only": {"llm_mode": "mistral", "vlm_mode": "none"},
    "A4_v2_vlm_gate_only": {"llm_mode": "none", "vlm_mode": "dry-run"},
    "A5_v2_llm_prior_plus_vlm_gate": {"llm_mode": "mistral", "vlm_mode": "dry-run"},
    "A6_v2_no_contact_gate": {"llm_mode": "mistral", "vlm_mode": "none", "ablation_flags": ["disable_vlm_contact_gate"]},
    "A7_v2_no_depth_gate": {"llm_mode": "mistral", "vlm_mode": "dry-run", "ablation_flags": ["disable_depth_refine"]},
    "A8_v2_no_anchor_propagation": {"llm_mode": "mistral", "vlm_mode": "dry-run", "ablation_flags": ["disable_anchor_propagation"]},
}

STAGE_ORDER = ["stage-1", "stage0", "stage1", "stage2", "stage3", "stage4", "stage5", "stage6", "stage7"]


def cap_stage(to_stage: str, max_stage: str) -> str:
    return to_stage if STAGE_ORDER.index(to_stage) <= STAGE_ORDER.index(max_stage) else max_stage


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        with path.open() as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _variant_dirs(profile, variant: str) -> tuple[Path, Path]:
    if variant == "A0_solved_baseline":
        render = profile.baseline.get("render_dir", "")
        return profile.sample_dir / "results" / "solved_baseline_reference", Path(render) if render else Path("")
    if variant == "A1_generic_pipeline_v1":
        return profile.sample_dir / "results" / "generic_pipeline_v1", profile.sample_dir / "results" / "renders" / "generic_pipeline_v1"
    name = f"generic_pipeline_v2_llm_vlm_gate/{variant}"
    return profile.sample_dir / "results" / name, profile.sample_dir / "results" / "renders" / name


def variant_result_name(variant: str) -> str:
    return f"generic_pipeline_v2_llm_vlm_gate/{variant}"


def run_variant(case_name: str, variant: str, *, to_stage: str, render: bool) -> dict[str, object]:
    if variant not in VARIANT_SETTINGS:
        return {"case_name": case_name, "variant": variant, "ran": False, "reason": "reference_or_unconfigured_variant"}
    settings = VARIANT_SETTINGS[variant]
    target_stage = to_stage if render else cap_stage(to_stage, "stage4")
    script = Path(__file__).resolve().parents[2] / "run_pipeline.py"
    cmd = [
        sys.executable,
        str(script),
        "--case",
        case_name,
        "--from-stage",
        "stage-1",
        "--to-stage",
        target_stage,
        "--result-name",
        variant_result_name(variant),
        "--llm-mode",
        settings["llm_mode"],
        "--vlm-mode",
        settings["vlm_mode"],
    ]
    for flag in settings.get("ablation_flags", []):
        cmd.extend(["--ablation-flag", flag])
    proc = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[5], text=True, capture_output=True)
    return {
        "case_name": case_name,
        "variant": variant,
        "ran": proc.returncode == 0,
        "returncode": proc.returncode,
        "command": " ".join(cmd),
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def summarize_case(case_name: str, *, run_variants: list[str] | None = None, to_stage: str = "stage7", render: bool = False) -> dict[str, object]:
    profile = load_case_profile(case_name)
    run_reports = []
    for variant in run_variants or []:
        run_reports.append(run_variant(case_name, variant, to_stage=to_stage, render=render))
    rows: list[dict[str, object]] = []
    for variant in ABLATION_VARIANTS:
        result_dir, render_dir = _variant_dirs(profile, variant)
        compare = _read_json(result_dir / "stage6_compare_report.json")
        loss = _read_json(result_dir / "loss_analysis" / "loss_summary.json")
        vlm_summary = result_dir / "vlm" / "vlm_summary.md"
        checks = compare.get("checks", {}) if isinstance(compare.get("checks"), dict) else {}
        pose_delta = checks.get("pose_delta", {}) if isinstance(checks.get("pose_delta"), dict) else {}
        row = {
            "case_name": case_name,
            "variant": variant,
            "result_dir": str(result_dir),
            "render_dir": str(render_dir),
            "result_exists": "1" if result_dir.exists() else "0",
            "render_exists": "1" if render_dir.exists() else "0",
            "median_2d_error": "",
            "median_contact_gap": "",
            "static_drift": "",
            "rotation_jump_count": "",
            "vlm_reject_count": "",
            "vlm_unclear_count": "",
            "pose_median_abs_delta": pose_delta.get("median_abs_delta", ""),
            "pose_p95_abs_delta": pose_delta.get("p95_abs_delta", ""),
            "pass_baseline_threshold": checks.get("overall_pass", ""),
            "loss_summary_exists": "1" if loss else "0",
            "vlm_summary_exists": "1" if vlm_summary.exists() else "0",
            "notes": "materialized or executable ablation row; use --run-variant to populate variant-specific outputs",
        }
        rows.append(row)
    out_dir = profile.sample_dir / "results" / "generic_pipeline_v2_llm_vlm_gate"
    csv_path = write_csv(out_dir / "ablation_summary.csv", rows)
    md_path = out_dir / "ablation_summary.md"
    lines = [f"# Ablation Summary: {case_name}", "", "| variant | result exists | pass | notes |", "|---|---:|---:|---|"]
    for row in rows:
        lines.append(f"| {row['variant']} | {row['result_exists']} | {row['pass_baseline_threshold']} | {row['notes']} |")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        "case_name": case_name,
        "ablation_summary_csv": str(csv_path),
        "ablation_summary_md": str(md_path),
        "variants": ABLATION_VARIANTS,
        "run_reports": run_reports,
    }
    write_json(out_dir / "ablation_summary.json", summary)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect generic pipeline v2 ablation summaries.")
    ap.add_argument("--case", default="all", help="Case name or all")
    ap.add_argument("--run-variant", action="append", default=[], choices=list(VARIANT_SETTINGS), help="Materialize a configured ablation variant before summarizing. Can be repeated.")
    ap.add_argument("--to-stage", default="stage7", choices=["stage-1", "stage0", "stage1", "stage2", "stage3", "stage4", "stage5", "stage6", "stage7"])
    ap.add_argument("--render", action="store_true", help="Allow run variants through stage5+ render. Without this, variants stop no later than stage4.")
    args = ap.parse_args()
    cases = available_cases() if args.case == "all" else [args.case]
    failed = False
    for case in cases:
        try:
            summary = summarize_case(case, run_variants=args.run_variant, to_stage=args.to_stage, render=args.render)
            print(f"{case}: {summary['ablation_summary_csv']}")
        except Exception as exc:
            failed = True
            print(f"{case}: FAILED: {exc}", file=sys.stderr)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
