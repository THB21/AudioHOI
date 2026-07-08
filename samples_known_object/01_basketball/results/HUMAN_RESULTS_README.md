# Human-side results (Tom, 2026-07-08)

Bridged from samples/ (identical video, md5-verified). Naming: `human_<dir>` = Tom's
human-modeling stack; `benchmark_vlm_qwen` etc. = Yixin's object pipeline (read-only).

- human_hands/            HaMeR MANO fingers + stitched SMPL-X params
- human_depth/            DA3 metric depth affine-aligned to GVHMR
- human_audio_semantics/  src/audio events + VLM contact records (anchor parts)
- human_contact_candidates/  proximity probes (hand/foot/floor)
- human_pose6d_sharedcam_depthv3/               object lifting (DA3)
- human_pose6d_sharedcam_contactphase_depthv3/  final audio contact-phase trajectory
  (sub-frame anchors, impulse budget, surface-gap anchors — loop_plan §9 iters 1-6)
- human_contact_refine/   body-side contact refinement (penetrations cleared, re-run 07-08
  against the final trajectory)
- human_evaluation/, human_llm_correction/  metrics + LLM audit
- renders/human_full_scene_3d/  SMPL-X + HaMeR hands + object, overlay & world orbit

Human params precedence for renders/eval: contact_refine > stitched hands > raw GVHMR.
