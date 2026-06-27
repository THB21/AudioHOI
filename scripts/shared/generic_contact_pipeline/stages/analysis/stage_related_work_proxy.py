from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    REPO = Path(__file__).resolve().parents[5]
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))

from scripts.shared.generic_contact_pipeline.core.base.config import available_cases, load_case_profile, with_runtime_overrides  # noqa: E402
from scripts.shared.generic_contact_pipeline.core.base.io import read_csv, write_csv, write_json  # noqa: E402
from scripts.shared.generic_contact_pipeline.core.base.schema import REQUIRED_RENDER_FILES, stage_paths  # noqa: E402


VARIANTS = [
    {
        "variant": "video_only_tracking",
        "missing_information": "mesh/contact/audio/LLM/VLM",
        "proxy_question": "How far can mask/track-only 2D observation go before 6D/contact refinement?",
    },
    {
        "variant": "mesh_only_alignment",
        "missing_information": "human contact/audio/VLM",
        "proxy_question": "Does object geometry alone explain the overlay/depth?",
    },
    {
        "variant": "human_only_contact",
        "missing_information": "object semantic parts/mesh",
        "proxy_question": "Does GVHMR proximity alone identify stable contact?",
    },
    {
        "variant": "no_audio_event",
        "missing_information": "lift/place/static audio cues",
        "proxy_question": "Does static/freeze degrade without audio event gating?",
    },
    {
        "variant": "no_vlm_gate",
        "missing_information": "forced-choice visual verification",
        "proxy_question": "Do rejected/unclear candidates enter anchor/contact updates?",
    },
    {
        "variant": "no_llm_prior",
        "missing_information": "semantic HOI prior profile",
        "proxy_question": "Are VLM choices/contact labels less constrained without object-specific semantics?",
    },
    {
        "variant": "no_contact_refine",
        "missing_information": "Stage4 contact/depth/SE(3) refinement",
        "proxy_question": "How much does Stage4 improve over Stage3 initial 2D-to-6D pose?",
    },
    {
        "variant": "ours_full",
        "missing_information": "none",
        "proxy_question": "Full AudioHOI v2: LLM prior + VLM gate schema + optimizer outputs.",
    },
]


def _json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open() as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return len(read_csv(path))
    except Exception:
        return 0


def _vlm_counts(profile) -> tuple[int, int, int]:
    path = stage_paths(profile)["vlm_gates"]
    if not path.exists():
        return 0, 0, 0
    rows = read_csv(path)
    reject = sum(1 for r in rows if r.get("pass_gate") == "reject")
    unclear = sum(1 for r in rows if r.get("pass_gate") == "unclear")
    effective = sum(1 for r in rows if r.get("is_effective") == "1")
    return reject, unclear, effective


def _renders_ok(profile) -> bool:
    return all((profile.render_dir / rel).exists() and (profile.render_dir / rel).stat().st_size > 0 for rel in REQUIRED_RENDER_FILES)


def build_rows(profile) -> list[dict[str, object]]:
    paths = stage_paths(profile)
    compare = _json(paths["compare_report"])
    checks = compare.get("checks", {}) if isinstance(compare.get("checks"), dict) else {}
    loss = _json(paths["loss_summary"])
    vlm_reject, vlm_unclear, vlm_effective = _vlm_counts(profile)
    frame_count = _count(paths["object_pose"])
    init_count = _count(paths["object_pose_init"])
    contact_count = _count(paths["object_contact_points"])
    rows: list[dict[str, object]] = []
    for spec in VARIANTS:
        variant = spec["variant"]
        if variant == "ours_full":
            status = "materialized"
            result_dir = str(profile.result_dir)
            evidence = "stage6_compare_report.json + stage7_loss_residuals.csv + six render videos"
        elif variant == "no_contact_refine":
            status = "materialized_proxy"
            result_dir = str(paths["object_pose_init"])
            evidence = "object_pose_init.csv compared with object_pose.csv"
        elif variant in {"no_vlm_gate", "no_llm_prior"}:
            ablation_variant = "A3_v2_llm_prior_only" if variant == "no_vlm_gate" else "A2_v2_no_llm_prior"
            ablation_dir = profile.sample_dir / "results" / "generic_pipeline_v2_llm_vlm_gate" / ablation_variant
            status = "materialized_executable_variant" if (ablation_dir / "pipeline_manifest.json").exists() else "executable_variant_defined"
            result_dir = str(ablation_dir)
            evidence = (
                f"{ablation_variant} exists at {ablation_dir}"
                if status == "materialized_executable_variant"
                else f"run stage_ablation --run-variant {ablation_variant} to materialize full CSV/render outputs"
            )
        else:
            status = "method_proxy_defined"
            result_dir = str(profile.result_dir)
            evidence = "available metrics are read from v2 full outputs; missing-information condition is a related-work proxy, not a full rerun"
        rows.append(
            {
                "case_name": profile.case_name,
                "variant": variant,
                "status": status,
                "result_or_proxy_path": result_dir,
                "missing_information": spec["missing_information"],
                "proxy_question": spec["proxy_question"],
                "frame_count": frame_count,
                "init_frame_count": init_count,
                "contact_rows": contact_count,
                "overall_pass": checks.get("overall_pass", ""),
                "renders_ok": "1" if _renders_ok(profile) else "0",
                "vlm_effective_rows": vlm_effective,
                "vlm_reject_count": vlm_reject,
                "vlm_unclear_count": vlm_unclear,
                "loss_summary_exists": "1" if loss else "0",
                "evidence": evidence,
            }
        )
    return rows


def write_report(profile, rows: list[dict[str, object]]) -> dict[str, object]:
    out_csv = profile.result_dir / "related_work_proxy_summary.csv"
    out_md = profile.result_dir / "related_work_proxy_report.md"
    write_csv(out_csv, rows)
    lines = [
        f"# Related-work Proxy Summary: {profile.case_name}",
        "",
        "This report maps inspected related-work ideas to AudioHOI proxy variants. It does not claim full reproduction of every external method.",
        "",
        "| variant | status | missing information | evidence |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(f"| {row['variant']} | {row['status']} | {row['missing_information']} | {row['evidence']} |")
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- `materialized` means the current v2 directory contains concrete CSV/render evidence.",
            "- `materialized_proxy` means the proxy is available from existing intermediate outputs, such as Stage3 before Stage4 refinement.",
            "- `executable_variant_defined` means the variant has deterministic runner settings but has not necessarily been rendered in this directory yet.",
            "- `method_proxy_defined` records the comparison axis for related-work discussion using the full v2 evidence.",
        ]
    )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = {"case_name": profile.case_name, "summary_csv": str(out_csv), "summary_md": str(out_md), "variants": [r["variant"] for r in rows]}
    write_json(profile.result_dir / "related_work_proxy_summary.json", payload)
    return payload


def run_case(case_name: str, result_name: str) -> dict[str, object]:
    profile = with_runtime_overrides(load_case_profile(case_name), result_name=result_name or None)
    rows = build_rows(profile)
    return write_report(profile, rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Write v2 related-work proxy summaries into each case result directory.")
    ap.add_argument("--case", default="all", help="Case name or all")
    ap.add_argument("--result-name", default="generic_pipeline_v2_llm_vlm_gate")
    args = ap.parse_args()
    cases = available_cases() if args.case == "all" else [args.case]
    failed = False
    for case in cases:
        try:
            summary = run_case(case, args.result_name)
            print(f"{case}: {summary['summary_csv']}")
        except Exception as exc:
            failed = True
            print(f"{case}: FAILED: {exc}", file=sys.stderr)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
