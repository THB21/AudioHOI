# Ablation Evaluation Design

This document defines the ablation protocol. The main rule is strict:

```text
Each method must map to a real result directory. Same-result relabeling is invalid.
```

The more important rule is causal:

```text
An ablation is valid only if it proves three things:
1. the intervention actually changed the intended input or gate;
2. that change reached the optimizer as a different constraint/residual/update decision;
3. the final pose/contact/temporal result changed, or the report explains why it did not.
```

Therefore the ablation report must not stop at "metrics did not change". It must
separate **intervention validity**, **mechanism activation**, and **outcome effect**.

## 1. Current materialized method variants

The current runnable default set contains only variants with real result directories for the complete final-result cases:

- `basketball`
- `football`

This avoids `unavailable` rows and avoids relabeling one directory as several methods. Mug/chair/stick should be added to this ablation table only after their final videos and paired object/human/contact/gate artifacts are complete and frame-aligned.

| Method | Meaning | Expected evidence |
| --- | --- | --- |
| `full_audio_vlm_llm` | complete pipeline | best final result |
| `no_audio` | remove audio timing/contact records while keeping VLM/LLM on | worse event timing or high-speed preservation if audio helps |
| `no_vlm_llm` | disable the VLM+LLM gate/audit path together | fewer gate effects, possible worse contact/overlay repairs |

Current default runner methods:

```text
full_audio_vlm_llm
no_audio
no_vlm_llm
```

## 1.1 Planned optional method variants

These are still useful scientifically, but they should not appear in the default benchmark until real result directories exist for every case:

| Method | Meaning | Expected evidence |
| --- | --- | --- |
| `audio_enabled` | audio-enabled control run | separate debugging control, not a default comparison |
| `no_vlm` | disable Qwen only | optional diagnostic if we later want to separate VLM from LLM |
| `no_llm` | disable Mistral/LLM only | optional diagnostic if we later want to separate LLM from VLM |
| `no_contact_anchor` | disable contact anchor residual/update | larger contact gap/drift |
| `no_surface_gap_anchor` | disable surface-gap/body-side contact refinement | lower floating maybe but higher penetration, or vice versa |
| `no_sequence_smooth` | disable velocity/acceleration sequence optimizer | more jumps/jitter |
| `no_static_tail` | disable static-tail freeze | static drift increases |
| `object_only` | no human-side refinement | worse HOI contact/penetration |
| `oracle_contact` | manual labels if available | upper bound only when labels exist |

## 2. Result directory naming

Current materialized benchmark result directories:

```text
full_audio_vlm_llm -> benchmark_vlm_qwen
no_audio           -> benchmark_no_audio
no_vlm_llm         -> benchmark_baseline_no_vlm
```

Future clean-room reruns may use `eval_*` names, but the current runner defaults to existing `benchmark_*` directories so the table is not filled with missing placeholders.
`no_vlm_llm` currently maps to the materialized gate-off baseline directory. It should be replaced by a cleaner `benchmark_no_vlm_llm` directory once that rerun is available.

Every result directory must include:

```text
pipeline_manifest.json
object_pose.csv
evaluation/final_evaluation_summary.json
```

## 3. Ablation table

The main table is only the delivery-level summary. It is not enough by itself.

```text
Case,Method,ResultDir,
Audio,VLM,LLM,AblationFlags,
PoseSHA256,SamePoseAsBaseline,MetricsIdenticalToBaseline,
OverlayHard↑,ContactGap↓,PartCorrect↑,
PenFrame↓,PenDepthMax↓,FloatingRate↓,Tradeoff↑,
ObjectJerk↓,HighSpeedRecall↑,StaticDrift↓,
AudioWindowContact↑,GateEffectCount↑,
VLMJudge↑,LLMFailureStage,FinalPass
```

Delta table:

```text
Delta(method) = Metric(method) - Metric(full_audio_vlm_llm)
```

Key deltas:

- `full - no_audio`
- `full - no_vlm_llm`

The required causal table is:

```text
Case,Method,
InterventionValid,
MechanismChanged,
OutcomeChanged,
InputDelta,
GateDelta,
ResidualDelta,
AnchorUpdateDelta,
FreezeDelta,
PoseDelta,
TemporalDelta,
Interpretation
```

Reading rule:

- `InterventionValid=False`: rerun or fix the variant; the result is not usable.
- `InterventionValid=True` and `MechanismChanged=False`: pipeline design problem. The ablated signal is not actually wired into optimizer decisions.
- `MechanismChanged=True` and `OutcomeChanged=False`: either hard constraints dominate, the affected frames are visually irrelevant, or the final metrics are too coarse. The report must show which one.
- `OutcomeChanged=True`: the ablated component has measurable downstream effect.

Planned optional deltas after real reruns exist:

- `full - no_surface_gap_anchor`
- `full - no_sequence_smooth`
- `full - no_contact_anchor`
- `full - object_only`

## 4. Audio ablation

Question: does audio help timing and physical motion?

Valid intervention evidence:

```text
audio_event_count
audio_contact_frame_count
motion_regime_changed_frames
audio_window_changed_frames
```

Mechanism evidence:

```text
audio_residual_enabled_frames
motion_regime_delta
contact_anchor_update_delta
optimizer_reweight_delta
```

Outcome metrics:

```text
contact_ratio_audio_windows
impact_timing_error_frames
accel_at_events
accel_in_flight
high_speed_recall
oversmooth_rate
VLM pairwise preference full vs no_audio
```

Expected examples:

- basketball: bounce/contact timing should improve with audio.
- football: kick event timing should improve, but foot-depth errors may remain.
- mug: set-down or tap timing may improve; grasp may still rely more on vision.
- stick: audio may be weak; honest result can show limited audio contribution.
- chair: audio is useful only if drag/support events are audible.

## 5. VLM / LLM ablation

Question: are VLM/LLM active or decorative?

Valid intervention evidence:

```text
vlm_query_count
llm_audit_query_count
gate_event_count
gate_active_count
gate_disabled_by_ablation
```

Mechanism evidence:

```text
gate_changed_residual_count
gate_blocked_anchor_update_count
gate_triggered_freeze_count
llm_audit_failure_count
llm_rerun_or_reweight_count
```

Outcome metrics:

```text
pose_delta_translation_max_m
translation_spike_delta
rotation_spike_delta
high_speed_recall_delta
oversmooth_delta
contact_proxy_delta
overlay_delta
```

If `full` and `no_vlm_llm` are identical, the report must not simply say "gate weak".
It must classify the failure:

1. `intervention_invalid`: VLM/LLM were not actually disabled, or the variant reused the full result.
2. `mechanism_not_connected`: VLM/LLM gates exist but do not change residuals, anchor update, freeze/interpolation, or optimizer weights.
3. `mechanism_connected_but_dominated`: gates changed optimizer decisions, but visual/depth/contact hard constraints dominated the final solution.
4. `metric_insensitive`: pose changed but current overlay/contact/temporal metrics cannot see the change.
5. `noncritical_frames`: gates affected frames outside the evaluated contact/high-speed/static intervals.
6. `parse_or_gate_mapping_failed`: VLM/LLM answers were produced but not parsed into gates.

This is not a display bug. It is a scientific result and must be reported.

Current design decision:

```text
no_vlm_llm must be treated as a causal intervention, not just a label.
If no_vlm_llm shows no outcome delta, the next action is to inspect mechanism
columns before blaming the model:
  gate_delta -> residual_delta -> anchor_update_delta -> freeze_delta -> pose_delta
```

## 6. Optional contact / penetration tradeoff ablation

This is not part of the current default table. If `no_surface_gap_anchor` and `no_contact_anchor` are reintroduced later, report both penetration and floating/contact gap.

Expected tradeoff table:

| Method | Contact Gap ↓ | Pen Depth ↓ | Floating ↓ | Tradeoff ↑ | Interpretation |
| --- | --- | --- | --- | --- | --- |
| full | low | low | low | high | desired |
| no_contact_anchor | high | low | high | low | object/hands float apart |
| no_surface_gap_anchor | low | high | low | low | contact kept by penetration |

This directly matches the meeting discussion: contact and penetration must be judged together.

## 7. Runner implementation

Add:

```text
scripts/shared/generic_contact_pipeline/configs/evaluation/method_variant_registry.yaml
scripts/shared/generic_contact_pipeline/tools/run_ablation_evaluation.py
scripts/shared/generic_contact_pipeline/core/evaluation/final_hoi/ablation_registry.py
scripts/shared/generic_contact_pipeline/core/evaluation/final_hoi/ablation_runner.py
```

Registry example:

```yaml
methods:
  full_audio_vlm_llm:
    result_name: benchmark_vlm_qwen
    ablation_flags: []
    audio: true
    vlm: qwen
    llm: mistral
  no_audio:
    result_name: benchmark_no_audio
    ablation_flags: [no_audio]
    audio: false
    vlm: qwen
    llm: mistral
  no_vlm_llm:
    result_name: benchmark_baseline_no_vlm
    ablation_flags: [no_vlm, no_llm]
    audio: true
    vlm: none
    llm: none
```

Runner checks:

```text
- same case cannot map two methods to the same result_dir unless --allow-same-result-debug
- missing result is recorded as missing_result unless --require-existing is used
- oracle_contact requires manual labels
- each ok result must have object_pose.csv
- pose_sha256 is computed from object_pose.csv
- same_pose_as_baseline reports byte-identical pose CSVs
- metrics_identical_to_baseline reports identical selected final metrics
```

## 8. Outputs

Current implemented outputs:

```text
final_result/evaluation/ablation/ablation_table.csv
final_result/evaluation/ablation/ablation_delta_table.csv
final_result/evaluation/ablation/ablation_evaluation_manifest.json
final_result/evaluation/ablation/ablation_method_registry.csv
final_result/evaluation/ablation/ablation_method_registry_manifest.json
final_result/evaluation/ablation/ablation_report.md
```

Planned split-out outputs after the next metric expansion:

```text
final_result/evaluation/ablation/audio_ablation_table.csv
final_result/evaluation/ablation/vlm_llm_gate_effectiveness.csv
final_result/evaluation/ablation/contact_penetration_tradeoff.csv
```

The current report explicitly flags whether a variant changed `object_pose.csv` and whether selected final metrics changed. If `same_pose_as_baseline=False` but `metrics_identical_to_baseline=True`, the result directory is real but the current metric set is not sensitive to that change, or shared aggregate HOI metrics dominate the table.
