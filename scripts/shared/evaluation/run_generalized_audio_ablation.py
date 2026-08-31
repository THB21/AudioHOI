#!/usr/bin/env python3
"""Run paired Ground/Visual vs Audio+VLM+Contact HOI-paper metrics.

The registry records exact trajectory provenance.  Results are never silently reused
between methods.  Existing per-variant JSON files may be reused with ``--reuse``.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
EVALUATOR = REPO / "scripts/shared/evaluation/compute_hoi_interaction_metrics.py"
NINE = REPO
OUT = REPO / "samples_known_object/hoi_interaction_evaluation/audio_ablation"


def variant(path: str, label: str) -> dict[str, str]:
    return {"trajectory": path, "label": label}


CASES = [
    dict(case="basketball", sample="samples_known_object/01_basketball", geometry=["--object-type", "sphere"],
         ground=variant("samples_known_object/01_basketball/results/benchmark_vlm_qwen/object_pose_init.csv", "visual_stage3_init"),
         full=variant("samples_known_object/01_basketball/results/benchmark_vlm_qwen/object_pose.csv", "audio_vlm_contact_final")),
    dict(case="football", sample="samples_known_object/10_football", geometry=["--object-type", "sphere"],
         ground=variant("samples_known_object/10_football/results/benchmark_vlm_qwen/object_pose_init.csv", "visual_stage3_init"),
         full=variant("samples_known_object/10_football/results/benchmark_vlm_qwen/object_pose.csv", "audio_vlm_contact_final")),
    dict(case="mug", sample="samples_known_object/02_mug", geometry=["--object-type", "sphere", "--object-radius-m", "0.048"],
         ground=variant("samples_known_object/02_mug/results/benchmark_vlm_qwen/object_pose_init.csv", "visual_stage3_init"),
         full=variant("samples_known_object/02_mug/results/benchmark_vlm_qwen/object_pose.csv", "audio_vlm_contact_final")),
    dict(case="chair", sample="samples_known_object/05_chair", geometry=["--object-type", "mesh_sdf", "--object-mesh-path", "assets/object_meshes/chair_yixin.glb"],
         ground=variant("samples_known_object/05_chair/results/benchmark_vlm_qwen/object_pose_init.csv", "visual_stage3_init"),
         full=variant("samples_known_object/05_chair/results/benchmark_vlm_qwen/object_pose.csv", "audio_vlm_contact_final")),
    dict(case="stick", sample="samples_known_object/11_stick", geometry=["--object-type", "capsule", "--object-length-m", "1.86"],
         ground=variant("samples_known_object/11_stick/results/benchmark_vlm_qwen/object_pose_init.csv", "visual_stage3_init"),
         full=variant("samples_known_object/11_stick/results/benchmark_vlm_qwen/object_pose.csv", "audio_vlm_contact_final")),
    dict(case="back_view_basketball", sample=str(NINE / "samples_known_object/12_back_view_basketball"), geometry=["--object-type", "sphere"],
         ground=variant(str(NINE / "samples_known_object/12_back_view_basketball/results/pure_solver_no_audio_no_vlm/object_pose.csv"), "pure_solver_no_audio_no_vlm"),
         full=variant(str(NINE / "samples_known_object/12_back_view_basketball/results/active_audio_visual_llm_audit/object_pose.csv"), "active_audio_visual_llm_audit")),
    dict(case="volleyball", sample=str(NINE / "samples_known_object/13_volleyball"), geometry=["--object-type", "sphere"],
         ground=variant(str(NINE / "samples_known_object/13_volleyball/results/final_full_4d_hoi/object_pose_init.csv"), "visual_stage3_init"),
         full=variant(str(NINE / "samples_known_object/13_volleyball/results/final_full_4d_hoi/object_pose.csv"), "audio_vlm_contact_final")),
    dict(case="pingpong", sample=str(NINE / "samples_known_object/14_pingpong_wall"), geometry=["--object-type", "sphere"],
         ground=variant(str(NINE / "samples_known_object/14_pingpong_wall/results/ablation_evaluation/frozen_poses/no_audio_object_pose.csv"), "frozen_no_audio"),
         full=variant(str(NINE / "samples_known_object/14_pingpong_wall/results/ablation_evaluation/frozen_poses/full_object_pose.csv"), "frozen_audio_vlm")),
    dict(case="suitcase", sample=str(NINE / "samples_known_object/15_suitcase_drag"),
         geometry=["--object-type", "mesh_sdf", "--object-mesh-path", "deliverables/tom_orbiting_3d/assets/suitcase_original_visuals.glb"],
         ground=variant(str(NINE / "samples_known_object/15_suitcase_drag/results/ablation_evaluation/frozen_poses/no_audio_object_pose_6d.csv"), "frozen_no_audio"),
         full=variant(str(NINE / "samples_known_object/15_suitcase_drag/results/ablation_evaluation/frozen_poses/full_object_pose_6d.csv"), "frozen_audio_vlm")),
]

METRICS = [
    "hoi_paper_human_temporal_smoothness_m",
    "hoi_paper_object_temporal_smoothness_m",
    "hoi_paper_non_collision_score",
    "hoi_paper_contact_score",
]


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO / p


def run_variant(case: dict, method: str, reuse: bool) -> dict:
    spec = case[method]
    sample = resolve(case["sample"])
    trajectory = resolve(spec["trajectory"])
    output = OUT / "per_variant" / f"{case['case']}__{method}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if reuse and output.exists():
        return json.loads(output.read_text())
    if not sample.exists() or not trajectory.exists():
        raise FileNotFoundError(f"missing {case['case']} {method}: sample={sample}, trajectory={trajectory}")
    cmd = [sys.executable, str(EVALUATOR), "--sample-dir", str(sample),
           "--trajectory-csv", str(trajectory), "--label", f"{case['case']}__{method}",
           "--out-json", str(output), *case["geometry"]]
    subprocess.run(cmd, cwd=REPO, check=True, stdout=subprocess.DEVNULL)
    return json.loads(output.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--only", default="", help="comma-separated case names")
    args = parser.parse_args()
    selected = set(filter(None, args.only.split(",")))
    cases = [case for case in CASES if not selected or case["case"] in selected]
    rows = []
    for case in cases:
        ground = run_variant(case, "ground", args.reuse)
        full = run_variant(case, "full", args.reuse)
        row = {
            "case": case["case"],
            "ground_label": case["ground"]["label"],
            "ground_trajectory": str(resolve(case["ground"]["trajectory"])),
            "full_label": case["full"]["label"],
            "full_trajectory": str(resolve(case["full"]["trajectory"])),
        }
        for metric in METRICS:
            gv, fv = ground.get(metric), full.get(metric)
            row[f"ground_{metric}"] = gv
            row[f"full_{metric}"] = fv
            row[f"delta_full_minus_ground_{metric}"] = None if gv is None or fv is None else fv - gv
        rows.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "generalized_audio_ablation.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "generalized_audio_ablation.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
