#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    REPO = Path(__file__).resolve().parents[4]
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
else:
    REPO = Path(__file__).resolve().parents[4]

from scripts.shared.generic_contact_pipeline.core.config import available_cases  # noqa: E402
from scripts.shared.generic_contact_pipeline.core.io import read_csv, write_csv, write_json  # noqa: E402


OUT_DIR = REPO / "docs" / "related_work_audit"
STAGE_DIR = REPO / "scripts" / "shared" / "generic_contact_pipeline" / "stages"


@dataclass
class ProbeSpec:
    name: str
    script: str
    args: list[str]
    method_reference: str
    audiohoi_question: str
    expected_outputs: list[str]
    heavy: bool = False


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def run_cmd(cmd: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, cwd=REPO, text=True, capture_output=True)
    return proc.returncode, proc.stdout, proc.stderr


def file_state(paths: list[str]) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for item in paths:
        path = REPO / item
        out[item] = {
            "exists": path.exists(),
            "bytes": path.stat().st_size if path.exists() else 0,
        }
    return out


def probe_specs(args: argparse.Namespace) -> list[ProbeSpec]:
    case_args = ["--case", args.case]
    return [
        ProbeSpec(
            name="repo_audit",
            script="stage_related_work_audit.py",
            args=[],
            method_reference="all related-work repos",
            audiohoi_question="Which public sources are actually cloned/runnable, and which are method-only references?",
            expected_outputs=[
                "docs/related_work_audit/related_work_audit.csv",
                "docs/related_work_audit/related_work_attempt_log.md",
            ],
        ),
        ProbeSpec(
            name="smoke_source_checks",
            script="stage_related_work_smoke.py",
            args=["--skip-imports"] if args.skip_smoke_imports else [],
            method_reference="InterDiff / MOVER / Open3DHOI local source trees",
            audiohoi_question="Which imported/source-level entries can be inspected without pretending full reproduction?",
            expected_outputs=[
                "docs/related_work_audit/related_work_smoke_tests.csv",
                "docs/related_work_audit/related_work_smoke_tests.md",
            ],
        ),
        ProbeSpec(
            name="source_loss_catalog",
            script="stage_related_work_loss_catalog.py",
            args=[],
            method_reference="InterDiff, MOVER, Open3DHOI source losses",
            audiohoi_question="Which concrete losses appear in source, and what AudioHOI proxy can test the same idea?",
            expected_outputs=[
                "docs/related_work_audit/related_work_loss_catalog.csv",
                "docs/related_work_audit/related_work_loss_catalog.md",
            ],
        ),
        ProbeSpec(
            name="source_evidence_verification",
            script="stage_related_work_source_evidence.py",
            args=[],
            method_reference="InterDiff / MOVER / Open3DHOI cited source line ranges",
            audiohoi_question="Do the cited source files/line ranges actually contain the loss symbols that the report maps to AudioHOI probes?",
            expected_outputs=[
                "docs/related_work_audit/related_work_source_evidence_verified.csv",
                "docs/related_work_audit/related_work_source_evidence_verified.md",
            ],
        ),
        ProbeSpec(
            name="paper_evidence_verification",
            script="stage_related_work_paper_evidence.py",
            args=[],
            method_reference="InterCap / InterDiff / HOI-PAGE / ZeroHSI paper text evidence",
            audiohoi_question="Do the paper-text windows support the method claims used to design AudioHOI proxy probes?",
            expected_outputs=[
                "docs/related_work_audit/related_work_paper_evidence_verified.csv",
                "docs/related_work_audit/related_work_paper_evidence_verified.md",
            ],
        ),
        ProbeSpec(
            name="trajectory_contact_temporal_probe",
            script="stage_related_work_method_probe.py",
            args=case_args,
            method_reference="InterDiff relative/temporal loss, MOVER contact/depth/static, Open3DHOI visual proxy",
            audiohoi_question="Do solved AudioHOI trajectories match or beat baselines under temporal/contact/depth proxy metrics?",
            expected_outputs=[
                "docs/related_work_audit/method_loss_probe_metrics.csv",
                "docs/related_work_audit/method_loss_probe_ratios.csv",
                "docs/related_work_audit/method_loss_probe_summary.md",
            ],
        ),
        ProbeSpec(
            name="observation_probe",
            script="stage_related_work_observation_probe.py",
            args=case_args,
            method_reference="Open3DHOI / HOI-PAGE / ZeroHSI visual observation and part-affordance gates",
            audiohoi_question="Are mask/semantic observations stable enough before pose/contact optimization?",
            expected_outputs=[
                "docs/related_work_audit/related_work_observation_probe_metrics.csv",
                "docs/related_work_audit/related_work_observation_probe_ratios.csv",
                "docs/related_work_audit/related_work_observation_probe_summary.md",
            ],
        ),
        ProbeSpec(
            name="da3_depth_probe",
            script="stage_related_work_depth_probe.py",
            args=case_args,
            method_reference="MOVER/Open3DHOI ordinal/render depth constraints",
            audiohoi_question="Does DA3 depth support the object depth ordering/contact-stage depth used by the solved case?",
            expected_outputs=[
                "docs/related_work_audit/related_work_depth_probe_metrics.csv",
                "docs/related_work_audit/related_work_depth_probe_ratios.csv",
                "docs/related_work_audit/related_work_depth_probe_summary.md",
            ],
        ),
        ProbeSpec(
            name="semantic_contact_geometry_probe",
            script="stage_related_work_geometry_contact_probe.py",
            args=case_args,
            method_reference="MOVER contact, InterCap/HOI dense hand-object geometry, HOI-PAGE part contact",
            audiohoi_question="Are semantic contact anchors near the intended human part and object-local surface?",
            expected_outputs=[
                "docs/related_work_audit/related_work_geometry_contact_probe_metrics.csv",
                "docs/related_work_audit/related_work_geometry_contact_probe_ratios.csv",
                "docs/related_work_audit/related_work_geometry_contact_probe_summary.md",
            ],
        ),
        ProbeSpec(
            name="dense_mesh_proximity_probe",
            script="stage_related_work_dense_mesh_probe.py",
            args=case_args + ["--body-stride", str(args.body_stride)],
            method_reference="InterCap/MOVER/Open3DHOI mesh-level contact distance",
            audiohoi_question="How close are sampled object surfaces to sampled SMPL-X vertices in camera 3D?",
            expected_outputs=[
                "docs/related_work_audit/related_work_dense_mesh_probe_metrics.csv",
                "docs/related_work_audit/related_work_dense_mesh_probe_ratios.csv",
                "docs/related_work_audit/related_work_dense_mesh_probe_summary.md",
            ],
            heavy=True,
        ),
        ProbeSpec(
            name="render_depth_probe",
            script="stage_related_work_render_depth_probe.py",
            args=case_args + ["--body-stride", str(args.body_stride), "--bin-size", str(args.bin_size)],
            method_reference="Open3DHOI/ZeroHSI/MOVER rendered depth / ordinal depth",
            audiohoi_question="When human and object overlap in image bins, is their camera-depth relationship plausible?",
            expected_outputs=[
                "docs/related_work_audit/related_work_render_depth_probe_metrics.csv",
                "docs/related_work_audit/related_work_render_depth_probe_ratios.csv",
                "docs/related_work_audit/related_work_render_depth_probe_summary.md",
            ],
            heavy=True,
        ),
        ProbeSpec(
            name="sdf_penetration_probe",
            script="stage_related_work_sdf_penetration_probe.py",
            args=case_args
            + [
                "--frame-step",
                str(args.sdf_frame_step),
                "--max-object-vertices",
                str(args.max_object_vertices),
                "--max-selected-frames",
                str(args.max_selected_frames),
            ],
            method_reference="MOVER / InterDiff / Open3DHOI SDF penetration",
            audiohoi_question="Do sampled object vertices penetrate the per-frame SMPL-X body mesh?",
            expected_outputs=[
                "docs/related_work_audit/related_work_sdf_penetration_probe_metrics.csv",
                "docs/related_work_audit/related_work_sdf_penetration_probe_ratios.csv",
                "docs/related_work_audit/related_work_sdf_penetration_probe_summary.md",
            ],
            heavy=True,
        ),
    ]


def read_ratio_rows(path: str, source_probe: str) -> list[dict[str, object]]:
    csv_path = REPO / path
    if not csv_path.exists():
        return []
    rows: list[dict[str, object]] = []
    for row in read_csv(csv_path):
        row = dict(row)
        row["source_probe"] = source_probe
        rows.append(row)
    return rows


def build_metric_index() -> list[dict[str, object]]:
    ratio_sources = [
        ("trajectory_contact_temporal_probe", "docs/related_work_audit/method_loss_probe_ratios.csv"),
        ("observation_probe", "docs/related_work_audit/related_work_observation_probe_ratios.csv"),
        ("da3_depth_probe", "docs/related_work_audit/related_work_depth_probe_ratios.csv"),
        ("semantic_contact_geometry_probe", "docs/related_work_audit/related_work_geometry_contact_probe_ratios.csv"),
        ("dense_mesh_proximity_probe", "docs/related_work_audit/related_work_dense_mesh_probe_ratios.csv"),
        ("render_depth_probe", "docs/related_work_audit/related_work_render_depth_probe_ratios.csv"),
        ("sdf_penetration_probe", "docs/related_work_audit/related_work_sdf_penetration_probe_ratios.csv"),
    ]
    rows: list[dict[str, object]] = []
    for probe, path in ratio_sources:
        rows.extend(read_ratio_rows(path, probe))
    write_csv(OUT_DIR / "related_work_metric_index.csv", rows)
    return rows


def write_index_md(run_rows: list[dict[str, object]], metric_rows: list[dict[str, object]], cases: str) -> Path:
    path = OUT_DIR / "related_work_probe_index.md"
    lines = [
        "# Related Work Probe Index",
        "",
        "This file is generated by `stage_related_work_run_all_probes.py`.",
        "It records how the inspected related-work methods are converted into conservative AudioHOI-side checks, without claiming that the external baselines are fully reproduced.",
        "",
        f"- Cases: `{cases}`",
        f"- Python: `{sys.executable}`",
        "",
        "## Probe Run Status",
        "",
        "| Probe | Related method idea | AudioHOI question | Status | Outputs |",
        "|---|---|---|---|---|",
    ]
    for row in run_rows:
        outputs = []
        states = row.get("outputs", {})
        if isinstance(states, dict):
            for name, state in states.items():
                if isinstance(state, dict) and state.get("exists"):
                    outputs.append(f"`{name}`")
        lines.append(
            f"| {row['name']} | {row['method_reference']} | {row['audiohoi_question']} | "
            f"{row['status']} ({row['returncode']}) | {'<br>'.join(outputs) if outputs else 'missing'} |"
        )

    lines.extend(
        [
            "",
            "## Method-To-Loss Mapping",
            "",
            "| External method family | What AudioHOI tests now | Main output |",
            "|---|---|---|",
            "| InterDiff | temporal smoothness, relative/contact-frame stability, penetration proxy | `related_work_source_evidence_verified.*`, `method_loss_probe_*`, `related_work_sdf_penetration_probe_*` |",
            "| MOVER | contact gap, depth order, static support, body-object SDF penetration | `related_work_source_evidence_verified.*`, `related_work_depth_probe_*`, `related_work_geometry_contact_probe_*`, `related_work_sdf_penetration_probe_*` |",
            "| Open3DHOI / HOI-Gaussian | observation/mask quality, rendered depth ordering, dense mesh proximity | `related_work_source_evidence_verified.*`, `related_work_observation_probe_*`, `related_work_render_depth_probe_*`, `related_work_dense_mesh_probe_*` |",
            "| InterCap | known-mesh RGB-D object tracking, contact refinement, temporal smoothing | `related_work_paper_evidence_verified.*`, `related_work_dense_mesh_probe_*`, `related_work_render_depth_probe_*` |",
            "| HOI-PAGE / ZeroHSI | part/affordance verification, video/render optimization, contact/penetration metrics | `related_work_paper_evidence_verified.*`, `related_work_observation_probe_*`, `related_work_render_depth_probe_*`, `related_work_sdf_penetration_probe_*` |",
            "",
            "## Metric Ratio Rows",
            "",
            "Rows below are generic-over-baseline when the source CSV provides a ratio. For lower-is-better metrics, values below 1 mean the generic pipeline is better than the stored solved baseline.",
            "",
            "| Probe | Case | Metric | Generic | Baseline | Ratio |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in metric_rows[:240]:
        case = row.get("case", "")
        metric = row.get("metric", row.get("variant", ""))
        generic = row.get("generic_median", "")
        baseline = row.get("baseline_median", "")
        ratio = row.get("generic_over_baseline_median", "")
        if ratio == "" and row.get("variant") == "generic_over_baseline_ratio":
            metric_values = [(k, v) for k, v in row.items() if k not in {"case", "variant", "source_probe"} and v not in {"", None}]
            for k, v in metric_values[:8]:
                lines.append(f"| {row.get('source_probe')} | {case} | {k} |  |  | {v} |")
            continue
        lines.append(f"| {row.get('source_probe')} | {case} | {metric} | {generic} | {baseline} | {ratio} |")

    if len(metric_rows) > 240:
        lines.append(f"| ... | ... | omitted {len(metric_rows) - 240} rows; see `related_work_metric_index.csv` |  |  |  |")

    lines.extend(
        [
            "",
            "## Important Caveats",
            "",
            "- The external repos are not claimed as fully reproduced unless their own data/model adapters are implemented.",
            "- SDF penetration is a real signed-distance query against the GVHMR SMPL-X body mesh, but it is not the full MOVER scene SDF pipeline.",
            "- Render-depth is a point-splat z-buffer proxy, not differentiable Gaussian/PyTorch3D rasterization.",
            "- Dense mesh proximity is unsigned nearest-neighbor distance; it diagnoses contact closeness but cannot alone detect penetration.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="Run all related-work method probes and write a single reproducible index.")
    ap.add_argument("--case", default="all", choices=["all"] + available_cases())
    ap.add_argument("--skip-smoke-imports", action="store_true")
    ap.add_argument("--skip-heavy", action="store_true", help="Skip dense mesh, render-depth, and SDF probes.")
    ap.add_argument("--keep-going", action="store_true", help="Continue after a failed probe and record the failure.")
    ap.add_argument("--body-stride", type=int, default=40)
    ap.add_argument("--bin-size", type=int, default=8)
    ap.add_argument("--sdf-frame-step", type=int, default=32)
    ap.add_argument("--max-object-vertices", type=int, default=250)
    ap.add_argument("--max-selected-frames", type=int, default=120)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for spec in probe_specs(args):
        if args.skip_heavy and spec.heavy:
            rows.append(
                {
                    **asdict(spec),
                    "status": "skipped_heavy",
                    "returncode": "",
                    "stdout_tail": "",
                    "stderr_tail": "",
                    "outputs": file_state(spec.expected_outputs),
                }
            )
            continue
        cmd = [sys.executable, str(STAGE_DIR / spec.script), *spec.args]
        code, stdout, stderr = run_cmd(cmd)
        status = "ok" if code == 0 else "failed"
        row = {
            **asdict(spec),
            "cmd": " ".join(cmd),
            "status": status,
            "returncode": code,
            "stdout_tail": stdout[-3000:],
            "stderr_tail": stderr[-3000:],
            "outputs": file_state(spec.expected_outputs),
        }
        rows.append(row)
        print(json.dumps({"probe": spec.name, "status": status, "returncode": code}, ensure_ascii=False))
        if code != 0 and not args.keep_going:
            write_json(OUT_DIR / "related_work_probe_run_log.json", {"runs": rows})
            raise SystemExit(code)

    metric_rows = build_metric_index()
    index_md = write_index_md(rows, metric_rows, args.case)
    write_json(
        OUT_DIR / "related_work_probe_run_log.json",
        {
            "case": args.case,
            "runs": rows,
            "metric_index_csv": rel(OUT_DIR / "related_work_metric_index.csv"),
            "index_md": rel(index_md),
        },
    )
    write_csv(
        OUT_DIR / "related_work_probe_run_log.csv",
        [
            {
                "name": row["name"],
                "status": row["status"],
                "returncode": row["returncode"],
                "method_reference": row["method_reference"],
                "audiohoi_question": row["audiohoi_question"],
                "cmd": row.get("cmd", ""),
            }
            for row in rows
        ],
    )
    print(
        json.dumps(
            {
                "probe_count": len(rows),
                "metric_rows": len(metric_rows),
                "run_log": rel(OUT_DIR / "related_work_probe_run_log.json"),
                "metric_index": rel(OUT_DIR / "related_work_metric_index.csv"),
                "index_md": rel(index_md),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
