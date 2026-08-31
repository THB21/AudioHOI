# LLM/VLM correction audit

- sample: `samples_known_object/12_back_view_basketball`
- trajectory: `samples_known_object/12_back_view_basketball/results/full_audio_vlm/derived_stage4/active_audio_visual_corrected/generic_sphere_sequence_candidate.csv`
- frames audited: 240

## Deterministic findings

| label | severity | frames | evidence |
|---|---|---|---|
| contact_gap_violation | high | 127-129, 148-160 | f127: +0.209 m vs left_hand; f128: +0.232 m vs left_hand; f129: +0.244 m vs left_hand; f148: +0.157 m vs left_hand; f149: +0.175 m vs left_hand; f150: +0.181 m vs left_hand; f151: +0.184 m vs left_hand; f152: +0.183 m vs left_hand |

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

Sampled frames saved under `samples_known_object/12_back_view_basketball/results/full_audio_vlm/derived_stage4/active_audio_visual_corrected/llm_vlm_correction/frames`.

## Verdict

**FAIL** — 1 high-severity finding(s) covering 16 frames (6.7% of 240)
