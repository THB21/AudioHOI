# Volleyball Full render and final ablation plan

Scope: object reconstruction only. GVHMR is read-only hand-contact evidence
and skeleton visualization. Human pose is never optimized.

| Step | Deliverable | Status |
|---|---|---|
| 1 | Audit video, config, Stage 0 inputs and existing artifacts | done |
| 2 | Run/repair SAM2 + persistent CoTracker + MegaPose/sphere observations | done |
| 3 | Build VLM-assisted two-hand contact and audio event timeline | done |
| 4 | Solve and inspect Full object pose | done |
| 5 | Produce six standard Stage 5 videos | done |
| 6 | Freeze No-audio and No-VLM poses | done |
| 7 | Run unified evaluator and write ablation summary | done |
| 8 | Replace boundary-sticking mask-gap interpolation with VLM-gated off-screen physics | done |

## Required Full render contract

- object_only/overlay.mp4
- object_only/camera3d.mp4
- object_only/side_yz.mp4
- with_human/overlay.mp4
- with_human/camera3d.mp4
- with_human/side_yz.mp4

## Constraints

- Do not overwrite the five-case canonical results.
- Do not modify downstream human refinement code.
- Do not optimize human state.
- Do not push until explicitly requested.

## Files added

- `scripts/shared/generic_contact_pipeline/configs/cases/volleyball.yaml`
- `scripts/shared/generic_contact_pipeline/tools/evaluate_sphere_sequence_ablation.py`

## Stage 0 audit

- 240 frames, static 1280x720 camera at 24 fps.
- SAM2 mask is present on 208 frames; empty intervals correspond to the ball
  leaving the top/left of frame.
- Persistent CoTracker publishes 101 sphere points per frame.
- MegaPose completed 16 automatically selected keyframes.
- DA3, GVHMR and audio artifacts are complete.

## Final result

- Full is accepted by the generic Stage 4 publisher.
- Qwen classifies all four missing-mask intervals (18--26, 73--85, 140--144,
  159--163) as `out_of_frame` with 0.98 confidence.
- These intervals are represented as `visibility=absent`; the projected
  constant-acceleration trajectory is not clipped to image bounds.
- 31/32 missing-mask frames project outside the image; only one transition
  frame remains just inside the boundary. No-audio has the same 31/32 result,
  confirming that this is VLM/physics behavior rather than audio behavior.
- Shared visible-mask error P95: 4.10 px.
- VLM identity-recovery interval 145--180 error P95: 0.04 px.
- Two-palm sphere contact gap P95: 77.20 mm.
- No-audio and No-VLM are intentionally retained as blocked candidates in
  `ablation_pose.csv`; neither overwrites canonical `object_pose.csv`.
- No-audio contact gap P95: 85.50 mm.
- No-VLM visible-mask error P95: 569.14 px; identity interval P95: 704.14 px;
  contact gap P95: 141.11 mm.
- Unified ablation outputs:
  `output/volleyball_unified_ablation_evaluation/ablation_metrics.{csv,json}`.
