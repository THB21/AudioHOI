# Shared Scripts

This folder now keeps only shared utilities that are still used outside the radius-free pipeline.

The current radius-free object pipeline is organized under:

```text
scripts/shared/radius_free_proxy/
  stage0_preprocess/
  stage1_observation/
  stage2_contact_candidates/
  stage3_da3_init_optimization/
  stage4_anchor_refinement/
  stage5_render/
```

Audio event detection for the radius-free line lives at:

```text
scripts/shared/radius_free_proxy/stage0_preprocess/align_audio_events.py
```

Human/body and hand utilities live under:

```text
scripts/shared/human/
scripts/shared/human_ball/
```
