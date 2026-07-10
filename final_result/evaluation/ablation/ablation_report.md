# Ablation Evaluation Report

This report compares real result directories. It does not reuse one result under multiple method labels.

## Summary

- rows: 6
- ok rows: 6
- missing rows: 0
- non-baseline rows with identical object_pose.csv hash: 0
- non-baseline rows with identical selected metrics: 0
- delta rows: 4

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
| basketball | full_audio_vlm_llm | ok | clean_ablation_full_audio_vlm_llm | True | qwen | mistral |  | True | True | 0.48 | 0.831 | generated_eval_proxy_render_mask_iou | ok | frame_timeline | 36 | 36 | 192 | 0.29 | False |
| basketball | no_audio | ok | clean_ablation_no_audio | False | qwen | mistral | disable_audio_events | False | False | 0.48 | 0.831 | generated_eval_proxy_render_mask_iou | ok | frame_timeline | 36 | 36 | 192 | 0.324 | False |
| basketball | no_vlm_llm | ok | clean_ablation_no_vlm_llm | True | none | none | no_vlm|no_llm | False | False | 0.48 | 0.831 | generated_eval_proxy_render_mask_iou | disabled_by_ablation | disabled_by_ablation | 0 | 0 | 0 | 0.314 | False |
| football | full_audio_vlm_llm | ok | clean_ablation_full_audio_vlm_llm | True | qwen | mistral |  | True | True | 0.007 | 0.805 | generated_eval_proxy_render_mask_iou | ok | frame_timeline | 36 | 36 | 242 | 0.12 | False |
| football | no_audio | ok | clean_ablation_no_audio | False | qwen | mistral | disable_audio_events | False | False | 0.007 | 0.805 | generated_eval_proxy_render_mask_iou | ok | frame_timeline | 36 | 36 | 242 | 0.294 | False |
| football | no_vlm_llm | ok | clean_ablation_no_vlm_llm | True | none | none | no_vlm|no_llm | False | False | 0.007 | 0.805 | generated_eval_proxy_render_mask_iou | disabled_by_ablation | disabled_by_ablation | 0 | 0 | 242 | 0.077 | False |

## Focused Ablation Deltas

`delta = method - full_audio_vlm_llm`. For error-like metrics, positive is worse. For recall-like metrics, negative is worse.

| case | method | intervention valid | mechanism changed | outcome changed | interpretation | Δ contact proxy | Δ overlay | Δ high-speed recall | Δ oversmooth | Δ gate events | Δ gate active | Δ reweight frames | Δ pose delta max | Δ anchor updates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| basketball | no_audio | True | True | True | measurable_downstream_effect | 0 | 0 | -0.161 | 0.161 | 0 | 0 | 0 | 0.034 | -48 |
| basketball | no_vlm_llm | True | True | True | measurable_downstream_effect | 0 | 0 | -0.032 | 0.032 | -36 | -36 | -192 | 0.024 | 0 |
| football | no_audio | True | True | True | measurable_downstream_effect | 0 | 0 | -0.105 | 0.105 | 0 | 0 | 0 | 0.174 | -4 |
| football | no_vlm_llm | True | True | True | measurable_downstream_effect | 0 | 0 | 0 | 0 | -36 | -36 | 0 | -0.043 | 0 |

## Current Interpretation

- `no_audio` tests whether audio timing/contact evidence changes the optimizer while VLM+LLM remain enabled.
- `no_vlm_llm` tests whether the VLM+LLM gate/audit path changes the result while audio remains enabled.
- If contact proxy and overlay remain unchanged but gate/pose/temporal deltas change, the current hard metrics are too coarse to show visual improvement and the gate-impact metrics should be used as the evidence.
- If a future `no_vlm_llm` row has zero pose/temporal delta, inspect `intervention_valid`, `mechanism_changed`, and `outcome_changed` before blaming the model or the evaluator.
