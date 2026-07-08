# LLM/VLM correction audit

- sample: `samples/football_10`
- trajectory: `samples/football_10/results/pose6d_sharedcam_contactphase_depthv3/ball_pose6d_sharedcam_contactphase_trajectory.csv`
- frames audited: 242

## Deterministic findings

| label | severity | frames | evidence |
|---|---|---|---|
| depth_outlier | high | 53-54 | f53→f54: Δtz=0.42 m |
| audio_visual_mismatch | medium | 57, 205 | f57: tap support=1.00, no anchor ±3; f205: strike support=0.86, no anchor ±3 |
| low_conf_unsupported | low | 6-15, 71-86 | 2 run(s) of depth_conf<0.05 with no audio support — blind interpolation, verify visually |

## VLM render check

VLM backend 'qwen' loaded. Prompt is fixed forced-choice (contact/part/impact); contact frames judged by solver-agreement, boundary frames left for human review.

| frame | kind | expected part | judgement | VLM says |
|---|---|---|---|---|
| 1 | boundary_first | left_foot | for_human_review | contact=False impact=medium entities=['left_foot', 'soccer ball'] "kicking" |
| 53 | contact | left_foot | pass | contact=True impact=medium entities=['left_foot', 'soccer ball'] "kicking" |
| 103 | contact | right_foot | pass | contact=True impact=medium entities=['right_foot', 'soccer ball'] "kicking" |
| 145 | contact | left_foot | unclear | contact=False impact=medium entities=['left_foot|right_foot', 'soccer ball'] "kicking" |
| 177 | contact | right_foot | unclear | contact=False impact=medium entities=['right_foot', 'soccer ball'] "kicking" |
| 242 | boundary_last | left_foot | for_human_review | contact=False impact=light entities=['soccer ball'] "" |

Sampled frames saved under `samples/football_10/results/llm_correction/frames`.

## Verdict

**WARN** — 3 finding(s): depth_outlier, audio_visual_mismatch, low_conf_unsupported
