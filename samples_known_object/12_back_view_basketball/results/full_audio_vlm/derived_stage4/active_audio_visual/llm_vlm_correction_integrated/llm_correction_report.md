# LLM/VLM correction audit

- sample: `samples_known_object/12_back_view_basketball`
- trajectory: `samples_known_object/12_back_view_basketball/results/full_audio_vlm/derived_stage4/active_audio_visual/generic_sphere_sequence_candidate.csv`
- frames audited: 240

## Deterministic findings

| label | severity | frames | evidence |
|---|---|---|---|
| boundary_drift | medium | 231-240 | last 10 frames tz std 0.068 m > 0.05 m |
| contact_gap_violation | high | 24, 43, 59, 79, 97, 127-129, 145, 148-160 | f24: +1.145 m vs left_hand; f43: +0.712 m vs left_hand; f59: +0.710 m vs left_hand; f79: +0.653 m vs left_hand; f97: +0.653 m vs right_hand; f127: +0.159 m vs left_hand; f128: +0.218 m vs left_hand; f129: +0.244 m vs left_hand |

## VLM render check

VLM disabled (--vlm-backend none) — frames saved for human review. Loaded 8 existing hash-verified Stage-4 VLM arbitration decisions from samples_known_object/12_back_view_basketball/results/full_audio_vlm/generic_stage4_candidate/generic_problem_preparation.json; the model was not rerun.

| frame | kind | expected part | judgement | VLM says |
|---|---|---|---|---|
| 1 | contact | right_hand | for_human_review | - |
| 2 | contact | right_hand | for_human_review | - |
| 161 | contact | left_hand | for_human_review | - |
| 162 | contact | left_hand | for_human_review | - |
| 240 | boundary_last | left_hand | for_human_review | - |
| 1 | existing_stage4_constraint_reliability | - | pass | contact=None impact=None entities=None "" |
| 23 | existing_stage4_constraint_reliability | - | unclear | contact=None impact=None entities=None "" |
| 42 | existing_stage4_constraint_reliability | - | unclear | contact=None impact=None entities=None "" |
| 58 | existing_stage4_constraint_reliability | - | unclear | contact=None impact=None entities=None "" |
| 73 | existing_stage4_constraint_reliability | - | unclear | contact=None impact=None entities=None "" |
| 78 | existing_stage4_constraint_reliability | - | unclear | contact=None impact=None entities=None "" |
| 96 | existing_stage4_constraint_reliability | - | unclear | contact=None impact=None entities=None "" |
| 161 | existing_stage4_constraint_reliability | - | unclear | contact=None impact=None entities=None "" |

Sampled frames saved under `samples_known_object/12_back_view_basketball/results/full_audio_vlm/derived_stage4/active_audio_visual/llm_vlm_correction_integrated/frames`.

## Verdict

**FAIL** — 1 high-severity finding(s) covering 22 frames (9.2% of 240)
