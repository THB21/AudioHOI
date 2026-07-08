# Push Scope for Generic Pipeline Update

This document separates upload-ready files from local-only notes and generated artifacts.

## Upload: Mainline Code

Upload the generic pipeline code and tests that define the current mainline:

- `scripts/shared/generic_contact_pipeline/run_pipeline.py`
- `scripts/shared/generic_contact_pipeline/stages/main/`
- `scripts/shared/generic_contact_pipeline/components/mainline/`
- `scripts/shared/generic_contact_pipeline/components/line_object/`
- `scripts/shared/generic_contact_pipeline/components/refinement/sequence_se3_optimizer.py`
- `scripts/shared/generic_contact_pipeline/components/refinement/policies/generic_line_physical_smooth.py`
- `scripts/shared/generic_contact_pipeline/components/refinement/policies/line_contact_lock.py`
- `scripts/shared/generic_contact_pipeline/core/gates/stage_audit.py`
- `scripts/shared/generic_contact_pipeline/core/evaluation/final_evaluator.py`
- `scripts/shared/generic_contact_pipeline/core/evaluation/final_summary.py`
- `scripts/shared/generic_contact_pipeline/core/evaluation/vlm_trace.py`
- `scripts/shared/generic_contact_pipeline/core/evaluation/benchmark.py`
- `scripts/shared/generic_contact_pipeline/tools/run_final_summary.py`
- `scripts/shared/generic_contact_pipeline/tools/run_final_evaluator.py`
- `scripts/shared/generic_contact_pipeline/tools/run_benchmark.py`
- `tests/test_vlm_trace_final_evaluator_benchmark.py`

## Upload: Mainline Explanation Docs

Upload only Markdown docs that explain the mainline and final evaluation:

- `docs/current_generic_pipeline_mainline.md`
- `docs/final_result_evaluation_method_cn.md`
- `docs/final_result_evaluation_method_en.md`
- `docs/push_scope_20260706.md`
- `scripts/shared/generic_contact_pipeline/components/mainline/README.md`
- `scripts/shared/generic_contact_pipeline/components/refinement/policies/README.md`
- `scripts/shared/generic_contact_pipeline/tools/README.md`

Do not upload meeting slide HTML exports.

## Upload: Final Evaluation Summary

Upload the final-only summary table and machine-readable manifest:

- `samples_known_object/final_result_evaluation/final_result_evaluation_summary.csv`
- `samples_known_object/final_result_evaluation/final_result_evaluation_summary.md`
- `samples_known_object/final_result_evaluation/final_result_evaluation_summary_manifest.json`

The HTML preview is local-only and ignored by `.gitignore`.

## Upload: Stick Final Result Core Artifacts

For the stick final result, upload compact CSV/JSON/YAML/MD artifacts from:

- `samples_known_object/11_stick/results/benchmark_vlm_qwen/`
- `samples_known_object/11_stick/results/renders/benchmark_vlm_qwen/`

Recommended compact artifacts:

- `object_pose.csv`
- `object_pose_init.csv`
- `object_pose_pre_smooth.csv`
- `object_observations.csv`
- `object_correspondence.csv`
- `object_surface_points.csv`
- `object_semantic_points.csv`
- `line_observations.csv`
- `line_correspondence.csv`
- `contact_candidates.csv`
- `anchor_state.csv`
- `object_contact_points.csv`
- `physical_smooth_residuals.csv`
- `pose_jump_audit.csv`
- `optimizer_decisions.csv`
- `motion_regime.csv`
- `stage*_metrics.json`
- `pipeline_manifest.json`
- `hoi_profile.json`
- `hoi_profile_resolved.yaml`
- `vlm/vlm_queries.csv`
- `vlm/vlm_results.csv`
- `vlm/vlm_gates.csv`
- `vlm/vlm_summary.md`
- `vlm_trace/trace_manifest.json`
- `vlm_trace/trace_completeness.json`
- `vlm_trace/06_evaluation/*`
- `renders/benchmark_vlm_qwen/outputs.json`
- `renders/benchmark_vlm_qwen/**/*_quality.csv`
- `renders/benchmark_vlm_qwen/**/*_preview.png`
- final render videos if remote storage allows them

Do not upload per-frame VLM evidence images under `vlm/**/evidence/`; they account for most of the 580 MB result size and are ignored.
Do not upload copied visual trace media under `vlm_trace/**/sampled_frames/` or `vlm_trace/**/*.jpg|png|pgm|mp4|html`; keep compact CSV/JSON/MD trace outputs instead.

## Local Only

Keep these local:

- `next_step.md`
- `evaluate.md`
- `cleanup_audit_*.json`
- `docs/report_slides_*.html`
- `docs/*completion_audit*.md`
- `docs/stick_full_pipeline_status.md`
- `samples_known_object/final_result_evaluation/*.html`
- `samples_known_object/*/results/benchmark_*/vlm/**/evidence/`
- `samples_known_object/*/results/benchmark_*/loss_analysis/*.png`
- `samples_known_object/*/results/benchmark_*/vlm_trace/**/sampled_frames/`
- `samples_known_object/*/results/benchmark_*/vlm_trace/**/*.jpg|png|pgm|mp4|html`
- non-final benchmark result variants unless explicitly needed for a method comparison

## Reply Draft

```text
Great, thanks! The penetration issue being gone is really good news. I’m also cleaning up my side now: I’ll push the latest generic mainline code, the stick final-result artifacts, and the final evaluation code/docs before tomorrow evening.

On my side, the remaining focus is exactly the missing part you mentioned: direct visual comparison against related papers, plus clearer reporting of object 6DoF translation/rotation in the animation. I have now separated the final-result evaluation from the benchmark tables, so the pushed version should include a cleaner final summary table and documentation of the hard metrics / VLM / LLM audit.

I’ll avoid pushing local meeting HTMLs and intermediate experiment artifacts, so the remote should only contain the mainline explanation, final evaluator, stick final result, and compact evaluation outputs.
```

## Suggested Staging Commands

```bash
# Mainline code + evaluator/test code
git add .gitignore \
  scripts/shared/generic_contact_pipeline \
  tests/test_vlm_trace_final_evaluator_benchmark.py \
  video_sample/11_stick.jpg video_sample/11_stick_video.mp4

# Upload docs only, no local meeting HTML
git add docs/current_generic_pipeline_mainline.md \
  docs/final_result_evaluation_method_cn.md \
  docs/final_result_evaluation_method_en.md \
  docs/push_scope_20260706.md

# Final-only evaluate summary, no HTML preview
git add samples_known_object/final_result_evaluation/final_result_evaluation_summary.csv \
  samples_known_object/final_result_evaluation/final_result_evaluation_summary.md \
  samples_known_object/final_result_evaluation/final_result_evaluation_summary_manifest.json

# Stick sample metadata, Articraft asset, final qwen artifacts and final render.
# Heavy VLM evidence images / sampled frames are ignored by .gitignore.
git add samples_known_object/11_stick/metadata.json \
  samples_known_object/11_stick/audio.wav \
  samples_known_object/11_stick/articraft \
  samples_known_object/11_stick/results/benchmark_vlm_qwen \
  samples_known_object/11_stick/results/renders/benchmark_vlm_qwen

# Optional: stage remote cleanup of previously tracked old experiment outputs.
git add -u samples_known_object/01_basketball/results/generic_pipeline_v2_llm_vlm_gate \
  samples_known_object/02_mug/results/generic_pipeline_v2_llm_vlm_gate \
  samples_known_object/05_chair/results/generic_pipeline_v2_llm_vlm_gate \
  samples_known_object/10_football/results/generic_pipeline_v2_llm_vlm_gate \
  samples_known_object/01_basketball/results/renders/generic_pipeline_v2_llm_vlm_gate \
  samples_known_object/02_mug/results/renders/generic_pipeline_v2_llm_vlm_gate \
  samples_known_object/05_chair/results/renders/generic_pipeline_v2_llm_vlm_gate \
  samples_known_object/10_football/results/renders/generic_pipeline_v2_llm_vlm_gate
```
