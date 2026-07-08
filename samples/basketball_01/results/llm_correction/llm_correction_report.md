# LLM/VLM correction audit

- sample: `samples/basketball_01`
- trajectory: `samples/basketball_01/results/pose6d_sharedcam_contactphase_depthv3/ball_pose6d_sharedcam_contactphase_trajectory.csv`
- frames audited: 192

## Deterministic findings

| label | severity | frames | evidence |
|---|---|---|---|
| boundary_drift | medium | 1-10 | first 10 frames tz std 0.062 m > 0.05 m |
| audio_visual_mismatch | medium | 24, 37, 77, 138, 174 | f24: bounce support=0.44, no anchor ±3; f37: bounce support=0.48, no anchor ±3; f77: bounce support=0.42, no anchor ±3; f138: bounce support=0.45, no anchor ±3; f174: bounce support=0.42, no anchor ±3 |

## VLM render check

VLM backend 'qwen' loaded. Prompt is fixed forced-choice (contact/part/impact); contact frames judged by solver-agreement, boundary frames left for human review.

| frame | kind | expected part | judgement | VLM says |
|---|---|---|---|---|
| 1 | boundary_first | right_hand | for_human_review | contact=False impact=light entities=['basketball'] "holding" |
| 3 | contact | right_hand | unclear | contact=False impact=light entities=['basketball'] "holding" |
| 82 | contact | right_hand | unclear | contact=False impact=light entities=['basketball'] "holding" |
| 132 | contact | right_hand | unclear | contact=False impact=light entities=['basketball'] "holding" |
| 192 | contact | right_hand | unclear | contact=False impact=light entities=['basketball'] "holding" |

Sampled frames saved under `samples/basketball_01/results/llm_correction/frames`.

## Verdict

**WARN** — 2 finding(s): boundary_drift, audio_visual_mismatch
