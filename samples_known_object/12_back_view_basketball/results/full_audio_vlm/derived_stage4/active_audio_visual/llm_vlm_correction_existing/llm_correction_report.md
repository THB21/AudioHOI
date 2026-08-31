# LLM/VLM correction audit

- sample: `samples_known_object/12_back_view_basketball`
- trajectory: `samples_known_object/12_back_view_basketball/results/full_audio_vlm/derived_stage4/active_audio_visual/generic_sphere_sequence_candidate.csv`
- frames audited: 240

## Deterministic findings

| label | severity | frames | evidence |
|---|---|---|---|
| boundary_drift | medium | 231-240 | last 10 frames tz std 0.068 m > 0.05 m |
| contact_gap_violation | high | 1-3, 161-162 | f1: +0.073 m vs right_hand; f2: +0.070 m vs right_hand; f3: +0.057 m vs right_hand; f161: +0.076 m vs left_hand; f162: -0.070 m vs left_hand |

## VLM render check

VLM disabled (--vlm-backend none) — frames saved for human review.

| frame | kind | expected part | judgement | VLM says |
|---|---|---|---|---|
| 1 | contact | right_hand | for_human_review | - |
| 2 | contact | right_hand | for_human_review | - |
| 161 | contact | left_hand | for_human_review | - |
| 162 | contact | left_hand | for_human_review | - |
| 240 | boundary_last | left_hand | for_human_review | - |

Sampled frames saved under `samples_known_object/12_back_view_basketball/results/full_audio_vlm/derived_stage4/active_audio_visual/llm_vlm_correction_existing/frames`.

## Verdict

**FAIL** — 1 high-severity finding(s) covering 5 frames (2.1% of 240)
