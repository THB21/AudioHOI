# Current Push Scope

This file records what should be included in the current integration push and what should stay local.

## Include In Push

### Mainline code

```text
scripts/shared/generic_contact_pipeline/run_pipeline.py
scripts/shared/generic_contact_pipeline/core/evaluation/final_evaluator.py
scripts/shared/generic_contact_pipeline/core/evaluation/final_hoi/
scripts/shared/generic_contact_pipeline/tools/run_final_hoi_evaluator.py
scripts/shared/generic_contact_pipeline/tools/run_ablation_evaluation.py
scripts/shared/generic_contact_pipeline/tools/README.md
scripts/shared/generic_contact_pipeline/configs/runtime_envs.yaml
scripts/shared/human_ball/render_full_scene_3d.py
tests/test_final_hoi_evaluation.py
```

Reason:

```text
These files define the current unified final HOI evaluator, ablation evaluator,
pipeline post-run hooks, runtime environment map, render-mask evaluation support,
and tests that keep the default materialized ablation variants strict.
```

### Evaluation design and implementation docs

```text
README.md
docs/current_generic_pipeline_mainline.md
docs/final_result_evaluation_method_cn.md
docs/final_result_evaluation_method_en.md
docs/next_final_evaluation_ablation_plan_cn.md
evalute/
```

Reason:

```text
These docs explain the current pipeline, runtime environments, final evaluation
metrics, ablation design, and next evaluation plan.
```

### Final result summary artifacts

```text
samples_known_object/final_result_evaluation/README.md
samples_known_object/final_result_evaluation/final_evaluation_detailed.csv
samples_known_object/final_result_evaluation/final_evaluation_human_readable.md
samples_known_object/final_result_evaluation/final_evaluation_summary_manifest.json
```

Reason:

```text
These are the current final five-case summaries. They are the human-readable and
machine-readable result table for the selected final result directory
`benchmark_vlm_qwen`.
```

### Ablation result summary artifacts

```text
samples_known_object/ablation_evaluation/README.md
samples_known_object/ablation_evaluation/ablation_table.csv
samples_known_object/ablation_evaluation/ablation_delta_table.csv
samples_known_object/ablation_evaluation/ablation_evaluation_manifest.json
samples_known_object/ablation_evaluation/ablation_method_registry.csv
samples_known_object/ablation_evaluation/ablation_method_registry_manifest.json
samples_known_object/ablation_evaluation/ablation_report.md
```

Reason:

```text
These compare six real materialized method variants for five cases:
full_audio_vlm_llm, audio_enabled, no_audio, no_vlm, no_llm, no_contact_anchor.
The registry manifest currently records require_existing=true and missing=0.
```

### Per-case final evaluation artifacts

For each final case result:

```text
samples_known_object/<case>/results/benchmark_vlm_qwen/evaluation/*.csv
samples_known_object/<case>/results/benchmark_vlm_qwen/evaluation/*.json
samples_known_object/<case>/results/benchmark_vlm_qwen/vlm_trace/06_evaluation/pipeline_qa_summary.csv
samples_known_object/<case>/results/benchmark_vlm_qwen/vlm_trace/06_evaluation/pipeline_qa_summary.json
samples_known_object/<case>/results/benchmark_vlm_qwen/vlm_trace/06_evaluation/pipeline_qa_summary.md
samples_known_object/<case>/results/benchmark_vlm_qwen/vlm_trace/06_evaluation/evaluation_summary.json
samples_known_object/<case>/results/benchmark_vlm_qwen/vlm_trace/06_evaluation/llm_audit_report.json
samples_known_object/<case>/results/benchmark_vlm_qwen/vlm_trace/06_evaluation/llm_eval_summary.md
samples_known_object/<case>/results/benchmark_vlm_qwen/pipeline_manifest.json
```

Reason:

```text
These artifacts make each final result auditable: object 6DoF, overlay,
part/contact, penetration/floating, temporal/audio metrics, and QA aggregation.
The legacy trace evaluation files are kept as compatibility summaries, but now
point to the final HOI evaluator and its pipeline QA aggregation.
```

## Keep Local / Do Not Push

### Render-mask frame cache

```text
samples_known_object/*/results/*/evaluation/render_masks/
```

Reason:

```text
These are per-frame local render masks used to compute overlay mask metrics.
The current cache is about 1.24GB across five cases. Push the derived
overlay_mask_metrics.csv/json instead. The masks are reproducible by rerunning
the final HOI evaluator.
```

### Dense preprocessing / audit working caches

```text
samples_known_object/*/results/segmentation/
samples_known_object/*/results/da3/
samples_known_object/*/results/gvhmr/
samples_known_object/*/results/hands/
samples_known_object/*/results/human_hands/
samples_known_object/*/results/human_audio_semantics/
samples_known_object/*/results/tracking/
samples_known_object/*/annotations/
samples_known_object/*/keyframes/
samples_known_object/*/keyframes_pose_fit/
samples_baseline_results/*/results/segmentation/
samples_baseline_results/*/results/sam2_jpg_frames/
samples_baseline_results/*/results/masks/
samples_baseline_results/*/results/da3/
samples_baseline_results/*/results/tracking/
```

Reason:

```text
These are dense intermediate caches: extracted SAM2 frames/masks, DA3 tensors,
GVHMR blobs, hand detector outputs, tracking probes, VLM crop sheets, and
manual/debug keyframes. They are useful locally for debugging but should not be
uploaded as the current clean integration artifact. The pushed result should
contain final renders, final compact CSV/JSON summaries, QA summaries, manifests,
and the code/docs needed to regenerate these caches.
```

### Meeting transcript working file

```text
*转写结果.docx
```

Reason:

```text
This is a local meeting transcript/working input, not a clean repository
artifact. The distilled plan is already in docs/evalute Markdown files.
```

### Legacy object-only summary files

`final_result_evaluation_summary.*` is not part of the current pushed final
result set. It can be regenerated through `run_final_summary.py` only for
compatibility, but the current final HOI reporting table is
`final_evaluation_human_readable.md` and `final_evaluation_detailed.csv`.

## Pre-Push Verification Commands

```bash
PYTHONDONTWRITEBYTECODE=1 /home/yang/miniconda3/bin/pytest -q \
  tests/test_final_hoi_evaluation.py \
  tests/test_vlm_trace_final_evaluator_benchmark.py
```

```bash
/home/yang/miniconda3/envs/audiohoi/bin/python -B -m compileall -q \
  scripts/shared/generic_contact_pipeline
```

```bash
/home/yang/miniconda3/envs/audiohoi/bin/python \
  scripts/shared/generic_contact_pipeline/tools/run_ablation_evaluation.py \
  --output-dir samples_known_object/ablation_evaluation \
  --require-existing
```

## Suggested Add Groups

Use these groups when preparing the commit:

```bash
git add .gitignore README.md docs/current_generic_pipeline_mainline.md \
  docs/final_result_evaluation_method_cn.md docs/final_result_evaluation_method_en.md \
  docs/current_push_scope.md docs/next_final_evaluation_ablation_plan_cn.md evalute/
```

```bash
git add scripts/shared/generic_contact_pipeline/run_pipeline.py \
  scripts/shared/generic_contact_pipeline/configs/runtime_envs.yaml \
  scripts/shared/generic_contact_pipeline/core/evaluation/final_evaluator.py \
  scripts/shared/generic_contact_pipeline/core/evaluation/final_hoi/ \
  scripts/shared/generic_contact_pipeline/tools/run_final_hoi_evaluator.py \
  scripts/shared/generic_contact_pipeline/tools/run_ablation_evaluation.py \
  scripts/shared/generic_contact_pipeline/tools/README.md \
  scripts/shared/human_ball/render_full_scene_3d.py \
  tests/test_final_hoi_evaluation.py
```

```bash
git add samples_known_object/final_result_evaluation/README.md \
  samples_known_object/final_result_evaluation/final_evaluation_detailed.csv \
  samples_known_object/final_result_evaluation/final_evaluation_human_readable.md \
  samples_known_object/final_result_evaluation/final_evaluation_summary_manifest.json \
  samples_known_object/ablation_evaluation/
```

```bash
git add samples_known_object/*/results/benchmark_vlm_qwen/pipeline_manifest.json \
  samples_known_object/*/results/benchmark_vlm_qwen/evaluation/*.csv \
  samples_known_object/*/results/benchmark_vlm_qwen/evaluation/*.json \
  samples_known_object/*/results/benchmark_vlm_qwen/vlm_trace/06_evaluation/evaluation_summary.json \
  samples_known_object/*/results/benchmark_vlm_qwen/vlm_trace/06_evaluation/llm_audit_report.json \
  samples_known_object/*/results/benchmark_vlm_qwen/vlm_trace/06_evaluation/llm_eval_summary.md \
  samples_known_object/*/results/benchmark_vlm_qwen/vlm_trace/06_evaluation/pipeline_qa_summary.csv \
  samples_known_object/*/results/benchmark_vlm_qwen/vlm_trace/06_evaluation/pipeline_qa_summary.json \
  samples_known_object/*/results/benchmark_vlm_qwen/vlm_trace/06_evaluation/pipeline_qa_summary.md
```
