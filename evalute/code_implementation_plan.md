# Evaluation Code Implementation Plan

This file maps the evaluation design to concrete code changes. It intentionally separates existing code, new modules, inputs, outputs, and command-line entry points.

## 1. Existing code to reuse

Object-level evaluator:

```text
scripts/shared/generic_contact_pipeline/core/evaluation/final_evaluator.py
scripts/shared/generic_contact_pipeline/core/evaluation/final_summary.py
scripts/shared/generic_contact_pipeline/core/evaluation/vlm_trace.py
scripts/shared/generic_contact_pipeline/core/evaluation/llm_csv_audit.py
```

HOI / human layer:

```text
scripts/shared/evaluation/compute_hoi_interaction_metrics.py
scripts/shared/human_ball/contact/object_geometry.py
scripts/shared/human_ball/contact/refine_body_pose_contact.py
scripts/shared/human_ball/render_full_scene_3d.py
src/audio/*
```

## 2. New target directory

Add implementation under:

```text
scripts/shared/generic_contact_pipeline/core/evaluation/final_hoi/
```

Proposed files:

```text
__init__.py
schemas.py
object_6d_metrics.py
overlay_metrics.py
part_metrics.py
hoi_contact_metrics.py
penetration_floating_metrics.py
temporal_audio_metrics.py
vlm_final_judge.py
llm_final_auditor.py
ablation_registry.py
ablation_runner.py
summary_writer.py
```

## 3. Shared schemas

`schemas.py`

```python
@dataclass
class EvaluationPaths:
    sample_dir: Path
    result_dir: Path
    render_dir: Path
    evaluation_dir: Path
    hoi_eval_json: Path | None
    human_params_dir: Path | None

@dataclass
class MetricBlock:
    name: str
    metrics: dict[str, float | int | str | bool | None]
    artifacts: dict[str, str]
    warnings: list[str]
```

All metric modules should return `MetricBlock`.

## 4. Object 6DoF metrics

`object_6d_metrics.py`

Inputs:

```text
object_pose.csv
object_pose_pre_smooth.csv
pose_jump_audit.csv
physical_smooth_residuals.csv
```

Functions:

```python
def compute_object_6d_metrics(paths: EvaluationPaths) -> MetricBlock:
    ...
```

Outputs:

```text
evaluation/object_6d_metrics.csv
evaluation/object_6d_metrics.json
```

Metrics:

```text
se3_valid
translation_valid_rate
rotation_valid_rate
quat_norm_error_mean
translation_velocity_mean/max
translation_acceleration_mean/max
rotation_velocity_mean/max
rotation_acceleration_mean/max
jump_count
static_tail_drift_m
geometry_spread_m
```

## 5. Overlay metrics

`overlay_metrics.py`

Purpose: replace VLM-primary overlay with hard metrics.

Inputs:

```text
SAM2 masks or segmentation masks
rendered object masks / silhouettes
object_observations.csv
object_correspondence.csv
line_correspondence.csv
render frames
```

Functions:

```python
def compute_overlay_metrics(paths: EvaluationPaths) -> MetricBlock:
    ...

def silhouette_iou(render_mask: np.ndarray, obs_mask: np.ndarray) -> float:
    ...

def chamfer_edge_distance(render_edge: np.ndarray, obs_edge: np.ndarray) -> float:
    ...

def line_alignment_metrics(render_axis_2d, observed_axis_2d) -> dict[str, float]:
    ...
```

Outputs:

```text
evaluation/overlay_metrics.csv
evaluation/overlay_mask_metrics.csv
evaluation/render_masks/
evaluation/overlay_worst_frames.csv
evaluation/overlay_evidence_frames/
```

Important rule:

```text
VLM can only add overlay_vlm_judge and overlay_conflict_flag.
It cannot replace overlay_iou/chamfer/line alignment.
If rendered masks are missing, generate evaluation-only render masks first.
```

Current implemented first pass:

```text
overlay_metrics.py discovers observed masks under results/segmentation/masks.
It discovers rendered masks under render_dir/object_masks, render_dir/masks, result_dir/render_masks, or evaluation/render_masks.
If rendered masks are missing, it first tries full URDF/Articraft geometry rasterization from object_pose.csv.
If no parseable geometry is available, it generates lightweight proxy render masks from object_pose.csv plus object_observations.csv into evaluation/render_masks.
Then it computes framewise IoU, mask coverage, and false coverage in overlay_mask_metrics.csv.
```

Next implementation upgrade:

```text
Replace remaining ball proxy masks with calibrated sphere render masks when full camera/radius evidence is available.
Keep the same output contract so final_evaluator and reports do not change.
```

## 6. Part metrics

`part_metrics.py`

Inputs:

```text
GVHMR result.pkl
HaMeR stitched_smplx_params.pkl
contact_refine/contact_refined_smplx_params.pkl
object_pose.csv
object semantic config / Articraft / URDF metadata
```

Functions:

```python
def compute_part_metrics(paths: EvaluationPaths, config: dict) -> MetricBlock:
    ...
```

Outputs:

```text
evaluation/human_parts.csv
evaluation/object_parts.csv
evaluation/object_part_vocab_map.csv
evaluation/part_metrics.csv
```

Implemented behavior:

```text
human_parts.csv:
  canonical human parts and contact evidence counts.

object_parts.csv:
  canonical object part used by metrics;
  raw_parts column preserving original config/contact/surface names;
  surface_point_rows and contact_evidence_rows after canonicalization.

object_part_vocab_map.csv:
  raw_part -> canonical_part mapping;
  source records case_config, object_surface_points, or hoi_contact_pairs.
```

Planned extension:

```text
evaluation/human_part_points.csv
evaluation/object_part_surface_points.csv
```

## 7. HOI contact metrics

`hoi_contact_metrics.py`

Inputs:

```text
contact_candidates.csv
anchor_state.csv
src/audio contact_records.csv or human_audio_semantics/contact_records.csv
human part points
object part surface points
object_pose.csv
```

Functions:

```python
def build_hoi_contact_pairs(paths: EvaluationPaths) -> Path:
    ...

def compute_hoi_contact_metrics(paths: EvaluationPaths) -> MetricBlock:
    ...
```

Outputs:

```text
evaluation/hoi_contact_pairs.csv
evaluation/hoi_contact_intervals.csv
evaluation/hoi_contact_metrics.csv
```

Metrics:

```text
contact_frame_ratio
contact_interval_recall
contact_gap_mm
contact_proxy = exp(-contact_gap_mm / 50)
part_correct_ratio
contact_drift_mean_mm
contact_drift_max_mm
switch_accuracy
```

## 8. Penetration / floating metrics

`penetration_floating_metrics.py`

Reuse:

```text
scripts/shared/human_ball/contact/object_geometry.py
```

Functions:

```python
def compute_penetration_floating_metrics(paths: EvaluationPaths) -> MetricBlock:
    ...

def compute_tradeoff_score(contact_gap_mean_mm: float, penetration_depth_mean_mm: float) -> float:
    ...
```

Outputs:

```text
evaluation/penetration_metrics.csv
evaluation/floating_metrics.csv
evaluation/contact_physics_tradeoff.csv
```

Metrics:

```text
penetration_frame_ratio
penetration_vertex_ratio
penetration_depth_mean_mm
penetration_depth_max_mm
floating_rate
floating_gap_mean_mm
tradeoff_score
part_weighted_penalty
```

Implementation note:

```text
Do not report object-pipeline contact_depth_offset_m as final penetration.
It may remain as a proxy/debug field only.
```

## 9. Temporal and audio-aware metrics

`temporal_audio_metrics.py`

Inputs:

```text
object_pose.csv
audio/contact events
hoi_contact_intervals.csv
pose_jump_audit.csv
```

Functions:

```python
def compute_temporal_audio_metrics(paths: EvaluationPaths) -> MetricBlock:
    ...
```

Outputs:

```text
evaluation/temporal_audio_metrics.csv
evaluation/audio_event_windows.csv
```

Metrics:

```text
object_velocity_mean/max
object_acceleration_mean/max
object_jerk
rotation_velocity_mean/max
rotation_acceleration_mean/max
accel_at_events
accel_in_flight
contact_ratio_audio_windows
impact_timing_error_frames
static_tail_drift_m
high_speed_recall
oversmooth_rate
```

## 10. VLM final judge

`vlm_final_judge.py`

Inputs:

```text
overlay_worst_frames.csv
penetration worst frames
contact gap worst frames
render evidence images
```

Functions:

```python
def build_vlm_final_queries(paths: EvaluationPaths) -> Path:
    ...

def parse_vlm_final_results(paths: EvaluationPaths) -> MetricBlock:
    ...
```

Outputs:

```text
evaluation/vlm_final_queries.csv
evaluation/vlm_final_results.csv
evaluation/vlm_final_scores.csv
evaluation/vlm_final_report.html
```

VLM questions must be simple and localized. Example:

```json
{
  "question": "Ignoring the occluded part, does the rendered stick shaft align with the visible real stick?",
  "answers": ["pass", "unclear", "fail"]
}
```

## 11. LLM final auditor

`llm_final_auditor.py`

Inputs:

```text
evaluation/*.csv
pipeline_manifest.json
stage_audit/stage_audit_gates.csv
vlm_trace/04_gating/gate_timeline.csv
optimizer_decisions.csv
```

Functions:

```python
def run_llm_final_audit(paths: EvaluationPaths, llm_mode: str) -> MetricBlock:
    ...
```

Outputs:

```text
vlm_trace/06_evaluation/pipeline_qa_summary.csv
vlm_trace/06_evaluation/pipeline_qa_summary.json
vlm_trace/06_evaluation/pipeline_qa_summary.md
vlm_trace/06_evaluation/vlm_eval_queries.csv
vlm_trace/06_evaluation/vlm_eval_raw_responses.jsonl
vlm_trace/06_evaluation/vlm_eval_parsed_scores.csv
vlm_trace/06_evaluation/vlm_eval_summary.json
vlm_trace/06_evaluation/llm_eval_summary.md
vlm_trace/06_evaluation/qa_audit_report.html
```

Current implementation:

```text
run_unified_final_evaluation(profile)
  -> hard metric blocks under evaluation/
  -> run_final_evaluator(profile, method="final_hoi", llm_mode="none")
  -> QA audit artifacts under vlm_trace/06_evaluation/
     - pipeline_qa_summary.* aggregates stage-level VLM/LLM QA and gates
     - vlm_eval_* records final-evaluator representative interval QA
  -> final_evaluation_summary.json records qa.* artifact paths
```

## 12. Unified final evaluator CLI

Add tool:

```text
scripts/shared/generic_contact_pipeline/tools/run_final_hoi_evaluator.py
```

CLI:

```bash
/home/yang/miniconda3/envs/audiohoi/bin/python \
  scripts/shared/generic_contact_pipeline/tools/run_final_hoi_evaluator.py \
  --case stick \
  --result-name benchmark_vlm_qwen \
  --output-dir final_result/evaluation
```

Cross-case:

```bash
/home/yang/miniconda3/envs/audiohoi/bin/python \
  scripts/shared/generic_contact_pipeline/tools/run_final_hoi_evaluator.py \
  --result-name benchmark_vlm_qwen \
  --output-dir final_result/evaluation
```

Outputs:

```text
final_result/evaluation/final_evaluation_detailed.csv
final_result/evaluation/final_evaluation_human_readable.md
final_result/evaluation/final_evaluation_summary_manifest.json
```

## 13. Implementation order

1. Implement `schemas.py` and path resolver.
2. Move current object final evaluator metrics into `object_6d_metrics.py`.
3. Import `hoi_summary` fields as a first version of HOI metrics.
4. Add hard overlay metrics; keep VLM overlay only as secondary judge.
5. Add penetration/floating tradeoff score.
6. Add temporal/audio metrics from pose and audio/contact windows.
7. Add unified summary writer.
8. Add ablation registry and benchmark runner.
