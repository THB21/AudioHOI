# Ablation Evaluation Report

This report compares real result directories. It does not reuse one result under multiple method labels.

## Summary

- rows: 3
- ok rows: 3
- missing rows: 0
- non-baseline rows with identical object_pose.csv hash: 0
- non-baseline rows with identical selected metrics: 0
- delta rows: 2

## How to read

- `same_pose_as_baseline=True` means the variant's `object_pose.csv` is byte-identical to `full_audio_vlm_llm` for that case.
- `metrics_identical_to_baseline=True` means the selected final metrics are identical to the baseline, even if files differ.
- If pose differs but metrics are identical, the current metrics are not sensitive to that variant or shared aggregate HOI metrics dominate the table.
- `audio`, `VLM`, `LLM`, and `flags` show the intended variant configuration; this is what prevents the table from silently reusing one result under several method labels.
- `gate status=ok` uses the frame-level gate timeline.
- `gate status=ok:stage_audit_fallback` means the frame-level timeline was empty, so the report uses stage-audit gate records instead of treating the row as missing.
- `gate status=disabled_by_ablation` is expected for the VLM+LLM-off variant.

## Variant audit

| case | method | status | result | audio | VLM | LLM | flags | same pose | same metrics | contact proxy | overlay IoU | overlay source | gate status | gate source | gate events | gates active | reweight | pose delta max | final pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| volleyball | full | ok | volleyball_full | True | qwen | none |  | True | True | 0.005 | 0.697 | generated_eval_proxy_render_mask_iou | partial:missing_optimizer_decisions|missing_physical_smooth_residuals|missing_object_pose_pre_smooth | frame_timeline | 25 | 25 | 0 |  | False |
| volleyball | no_audio | ok | volleyball_no_audio | False | qwen | none | disable_audio_events | False | False | 0.001 | 0.699 | generated_eval_proxy_render_mask_iou | partial:missing_optimizer_decisions|missing_physical_smooth_residuals|missing_object_pose_pre_smooth | stage_audit_fallback | 13 | 0 | 0 |  | False |
| volleyball | no_vlm | ok | volleyball_no_vlm | True | none | none | disable_vlm_semantic_evidence | False | False | 0.002 | 0.431 | generated_eval_proxy_render_mask_iou | disabled_by_ablation | disabled_by_ablation | 0 | 0 | 0 |  | False |

## Focused Ablation Deltas

`delta = method - full`. For error-like metrics, positive is worse. For recall-like metrics, negative is worse.

| case | method | intervention valid | mechanism changed | outcome changed | interpretation | Δ contact proxy | Δ overlay | Δ high-speed recall | Δ oversmooth | Δ gate events | Δ gate active | Δ reweight frames | Δ pose delta max | Δ anchor updates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| volleyball | no_audio | True | True | True | measurable_downstream_effect | -0.004 | 0.002 |  |  | -12 | -25 | 0 |  | -50 |
| volleyball | no_vlm | True | True | True | measurable_downstream_effect | -0.003 | -0.267 |  |  | -25 | -25 | 0 |  | -6 |

## Current Interpretation

- `no_audio` tests whether audio timing/contact evidence changes the optimizer while VLM+LLM remain enabled.
- `no_vlm`/`no_vlm_llm` tests whether the configured VLM semantic gate path changes the result while audio remains enabled.
- If contact proxy and overlay remain unchanged but gate/pose/temporal deltas change, the current hard metrics are too coarse to show visual improvement and the gate-impact metrics should be used as the evidence.
- If a future VLM-off row has zero pose/temporal delta, inspect `intervention_valid`, `mechanism_changed`, and `outcome_changed` before blaming the model or the evaluator.
