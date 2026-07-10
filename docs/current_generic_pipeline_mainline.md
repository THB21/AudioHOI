# Current Generic Pipeline Mainline

This file records the cleaned-up convention after the July 6, 2026 result sweep.

## Kept / Pushed Result Directories

For each active case under `samples_known_object/*/results`, the clean pushed
result set keeps the compact final/ablation result directories and final render
directories:

- `benchmark_baseline_no_vlm`: baseline result with VLM/LLM disabled.
- `benchmark_vlm_qwen`: final VLM-gated result with Qwen evidence gates.
- `benchmark_llm_mistral`: LLM-only comparison result.
- `benchmark_no_llm`: no-LLM comparison result.
- `benchmark_audio_enabled`: audio-enabled comparison result.
- `benchmark_no_audio`: no-audio comparison result.
- `benchmark_no_anchor`: anchor ablation comparison result.
- `renders/benchmark_baseline_no_vlm` and `renders/benchmark_vlm_qwen`: render outputs for the baseline and final visual result.

Dense preprocessing artifacts are local/regenerable caches, not clean pushed
artifacts:

- extracted `frames/`
- `segmentation/` SAM2 frame/mask caches
- `tracking/` probe tracks
- `da3/` dense depth tensors
- `gvhmr/` and hand detector blobs
- per-frame VLM evidence images
- per-frame evaluator `render_masks/`

The pushed result should contain the derived compact CSV/JSON/MD summaries,
manifests, and final videos needed to inspect the outcome without uploading
every intermediate frame.

The final cross-case report is:

- `samples_known_object/generic_pipeline_v2_test_reports/five_case_full_qwen_benchmark`

The stick historical reference result is kept only as a known-good comparison point:

- `samples_known_object/11_stick/results/generic_pipeline_v2_stick_mainline_seq_anchor_full_qwen`

## Current Execution Mainline

The only intended pipeline entrypoint is:

- `scripts/shared/generic_contact_pipeline/run_pipeline.py`

It dispatches the fixed stage chain:

1. `stages/main/stage0_preprocess.py`
2. `stages/main/stage1_observation.py`
3. `stages/main/stage2_contact.py`
4. `stages/main/stage3_initial_pose.py`
5. `stages/main/stage4_contact_refine.py`
6. `stages/main/stage5_render.py`
7. `stages/main/stage6_compare.py`

Stage4 always enters:

- `components/mainline/sequence_refine.py`

The final pose is produced by the generic SE3 mainline. Old refinement policy files are not intended as pipeline branches. They remain only as compatibility seed, geometry adapter, or residual-builder code while their behavior is migrated into the shared mainline.

## Mainline Component Roles

- `components/mainline/observation.py`: normalizes observations into shared object observation/correspondence artifacts.
- `components/mainline/contact_anchor.py`: normalizes contact candidates and anchor state.
- `components/mainline/pose_init.py`: writes the shared SE3 pose init schema.
- `components/mainline/sequence_refine.py`: runs the generic sequence SE3 smoothing/anchor/gate refinement.
- `core/gates/stage_audit.py`: writes VLM/LLM audit decisions and gate signals.
- `core/evaluation/final_evaluator.py`: final hard-metric, VLM judge, and LLM audit summary.
- `core/evaluation/final_hoi/`: unified final HOI evaluator for object 6DoF, overlay, contact/anchor, physical, temporal, and QA artifacts.
- `core/evaluation/final_summary.py`: legacy object-only final summary.
- `core/evaluation/benchmark.py`: cross-case benchmark table/report generation.

## Reporting Entry Points

Use the final-only summary for reporting current results:

```bash
/home/yang/miniconda3/envs/audiohoi/bin/python \
  scripts/shared/generic_contact_pipeline/tools/run_final_hoi_evaluator.py \
  --result-name benchmark_vlm_qwen \
  --output-dir samples_known_object/final_result_evaluation
```

Use the current ablation evaluator for comparing real materialized method variants:

```bash
/home/yang/miniconda3/envs/audiohoi/bin/python \
  scripts/shared/generic_contact_pipeline/tools/run_ablation_evaluation.py \
  --output-dir samples_known_object/ablation_evaluation
```

`run_benchmark.py` is legacy and should only be used for old benchmark diagnostics.

Evaluation method documentation:

- `docs/final_result_evaluation_method_cn.md`
- `docs/final_result_evaluation_method_en.md`

## Cleanup Audit

The exact delete/keep plan for this cleanup is stored at:

- `cleanup_audit_20260706.json`

Removed items were old experiment outputs, duplicate render directories, old
report directories, Python `__pycache__` directories, extracted frame caches,
SAM2/DA3/tracking/GVHMR dense caches, and per-frame evidence images. Source code,
compact result CSV/JSON/MD artifacts, final videos, final previews, and
regeneration scripts are kept.
