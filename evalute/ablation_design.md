# Ablation Evaluation Design

This document defines the ablation protocol. The main rule is strict:

```text
Each method must map to a real result directory. Same-result relabeling is invalid.
```

## 1. Current materialized method variants

The current runnable default set contains only variants with real result directories for all five cases. This avoids `unavailable` rows and avoids relabeling one directory as several methods.

| Method | Meaning | Expected evidence |
| --- | --- | --- |
| `full_audio_vlm_llm` | complete pipeline | best final result |
| `audio_enabled` | audio-enabled control run | should be a separate result directory from `full_audio_vlm_llm` |
| `no_audio` | remove audio timing/contact records | worse event timing or high-speed preservation if audio helps |
| `no_vlm` | disable Qwen gates/final judge | fewer gate effects, possible worse contact/overlay repairs |
| `no_llm` | seed profile only, no Mistral/LLM audit | weaker semantic profile/audit localization |
| `no_contact_anchor` | disable contact anchor residual/update | larger contact gap/drift |

Current default runner methods:

```text
full_audio_vlm_llm
audio_enabled
no_audio
no_vlm
no_llm
no_contact_anchor
```

## 1.1 Planned optional method variants

These are still useful scientifically, but they should not appear in the default benchmark until real result directories exist for every case:

| Method | Meaning | Expected evidence |
| --- | --- | --- |
| `no_surface_gap_anchor` | disable surface-gap/body-side contact refinement | lower floating maybe but higher penetration, or vice versa |
| `no_sequence_smooth` | disable velocity/acceleration sequence optimizer | more jumps/jitter |
| `no_static_tail` | disable static-tail freeze | static drift increases |
| `object_only` | no human-side refinement | worse HOI contact/penetration |
| `oracle_contact` | manual labels if available | upper bound only when labels exist |

## 2. Result directory naming

Current materialized benchmark result directories:

```text
full_audio_vlm_llm -> benchmark_vlm_qwen
audio_enabled      -> benchmark_audio_enabled
no_audio           -> benchmark_no_audio
no_vlm             -> benchmark_baseline_no_vlm
no_llm             -> benchmark_no_llm
no_contact_anchor  -> benchmark_no_anchor
```

Future clean-room reruns may use `eval_*` names, but the current runner defaults to existing `benchmark_*` directories so the table is not filled with missing placeholders.

Every result directory must include:

```text
pipeline_manifest.json
object_pose.csv
evaluation/final_evaluation_summary.json
```

## 3. Ablation table

Main table:

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
- `full - no_vlm`
- `full - no_llm`
- `full - no_contact_anchor`
- `full - audio_enabled`

Planned optional deltas after real reruns exist:

- `full - no_surface_gap_anchor`
- `full - no_sequence_smooth`
- `full - object_only`

## 4. Audio ablation

Question: does audio help timing and physical motion?

Metrics:

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

Metrics:

```text
vlm_query_count
vlm_gate_count
effective_gate_count
gate_changed_residual_count
gate_blocked_anchor_update_count
gate_triggered_freeze_count
llm_audit_failure_count
llm_rerun_or_reweight_count
```

If `full` and `no_vlm` are identical, report one of the following:

1. VLM gates are too weak and do not affect optimizer.
2. Hard constraints already dominate; VLM is only audit.
3. VLM queried non-critical frames.
4. VLM parse/gate mapping failed.

This is not a display bug. It is a scientific result and must be reported.

## 6. Contact / penetration tradeoff ablation

For `no_surface_gap_anchor` and `no_contact_anchor`, report both penetration and floating/contact gap.

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
  no_vlm:
    result_name: benchmark_baseline_no_vlm
    ablation_flags: [no_vlm]
    audio: true
    vlm: none
    llm: mistral
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
samples_known_object/ablation_evaluation/ablation_table.csv
samples_known_object/ablation_evaluation/ablation_delta_table.csv
samples_known_object/ablation_evaluation/ablation_evaluation_manifest.json
samples_known_object/ablation_evaluation/ablation_method_registry.csv
samples_known_object/ablation_evaluation/ablation_method_registry_manifest.json
samples_known_object/ablation_evaluation/ablation_report.md
```

Planned split-out outputs after the next metric expansion:

```text
samples_known_object/ablation_evaluation/audio_ablation_table.csv
samples_known_object/ablation_evaluation/vlm_llm_gate_effectiveness.csv
samples_known_object/ablation_evaluation/contact_penetration_tradeoff.csv
```

The current report explicitly flags whether a variant changed `object_pose.csv` and whether selected final metrics changed. If `same_pose_as_baseline=False` but `metrics_identical_to_baseline=True`, the result directory is real but the current metric set is not sensitive to that change, or shared aggregate HOI metrics dominate the table.
