# Current Generic Pipeline Mainline

This file records the cleaned-up convention after the July 6, 2026 result sweep.

## Kept Result Directories

For each active case under `samples_known_object/*/results`, the cleaned result set keeps:

- `benchmark_baseline_no_vlm`: baseline result with VLM/LLM disabled.
- `benchmark_vlm_qwen`: final VLM-gated result with Qwen evidence gates.
- `benchmark_llm_mistral`: LLM-only comparison result.
- `benchmark_no_llm`: no-LLM comparison result.
- `benchmark_audio_enabled`: audio-enabled comparison result.
- `benchmark_no_audio`: no-audio comparison result.
- `benchmark_no_anchor`: anchor ablation comparison result.
- `segmentation`, `tracking`, `da3`, `gvhmr`, `events`: reusable preprocessing artifacts.
- `renders/benchmark_baseline_no_vlm` and `renders/benchmark_vlm_qwen`: render outputs for the baseline and final visual result.

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
- `core/evaluation/final_summary.py`: final-only cross-case summary for the selected final result of each case.
- `core/evaluation/benchmark.py`: cross-case benchmark table/report generation.

## Reporting Entry Points

Use the final-only summary for reporting current results:

```bash
python scripts/shared/generic_contact_pipeline/tools/run_final_summary.py
```

Use benchmark only for comparing real method variants:

```bash
python scripts/shared/generic_contact_pipeline/tools/run_benchmark.py \
  --cases basketball football mug chair stick \
  --methods baseline_no_vlm vlm_gated no_llm llm_mistral no_audio audio_enabled no_anchor
```

Evaluation method documentation:

- `docs/final_result_evaluation_method_cn.md`
- `docs/final_result_evaluation_method_en.md`

## Cleanup Audit

The exact delete/keep plan for this cleanup is stored at:

- `cleanup_audit_20260706.json`

Removed items were old experiment outputs, duplicate render directories, old report directories, and Python `__pycache__` directories. Source code and reusable preprocessing artifacts were not removed.
