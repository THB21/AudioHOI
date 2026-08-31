# Frozen inputs for the nine-case audio ablation

This directory contains the compact object trajectories used to reproduce the
published world-view comparisons. Each case has two frame-aligned trajectories.

- `vlm.csv` is the visual and VLM arm with audio events disabled.
- `vlm_audio.csv` is the matched visual, VLM, and audio arm.

All rendering options, human parameters, meshes, camera motion, and output
settings are held constant between the two arms. Only the stored object
trajectory changes. Some cases have identical accepted trajectories in both
arms. They remain in the release because a null audio effect is a valid
ablation result and must not be replaced by an artificial difference.

The Ping-Pong directory additionally contains `human_sites_hamer.csv`. It is
used to attach the procedural paddle to the reconstructed right hand.

The files are consumed by
`scripts/shared/evaluation/nine_modality_ablation.py` and are deliberately kept
separate from generated pipeline caches under `samples_known_object`.
