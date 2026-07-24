# Mug Solved-Seed Removal Comparison

Canonical: `benchmark_vlm_qwen`

Fresh: `mug_observation_seed_v4`

The fresh run starts from an empty result directory and derives the body pose
and observable axial phase from 2D boxes, DA3 depth, GVHMR intrinsics and the
declared mug meshes. It does not read M15/M17/M18/final solved pose or phase.
The final six videos were fully regenerated after removal of the Stage 4
hard-coded phase anchors.

## Acceptance results

| Metric | Canonical | Fresh | Result |
| --- | ---: | ---: | --- |
| final pass | false | true | pass |
| overlay IoU | 0.450945 | 0.539058 | improved |
| mask coverage | 0.498253 | 0.578241 | improved |
| false coverage | 0.159456 | 0.108992 | improved |
| contact frame ratio | 0.4667 | 0.4667 | equal |
| contact gap (mm) | 15.78 | 15.78 | equal |
| contact proxy | 0.729351 | 0.729351 | equal |
| audio-window contact ratio | 0.571 | 0.571 | equal |
| total temporal spikes | 9 | 2 | improved |
| jump count | 0 | 0 | equal |
| static-tail drift (m) | 0.001190 | 0 | improved |
| persistent contact rows | 236 | 236 | equal |

All 21 machine acceptance checks pass. Handle-loop (175) and rim-ring (5)
semantic contact counts are exact, as is the VLM hand-part distribution
(`handle=175`, `body=65`). The finer body/occlusion labels differ and remain
reported in the generated comparison JSON; they are not hidden behind an
aggregate score.

## Decoded output file hashes

| Output | SHA-256 |
| --- | --- |
| object_only/overlay.mp4 | `2c06be41a510d0f9ac7671ed618ffabcf24f7b9663a87926d541b76f2c78ff0b` |
| object_only/camera3d.mp4 | `32f7c3535f6dece9afa85c92a41213a3895835938e20951037fbd75c8fb6c1f7` |
| object_only/side_yz.mp4 | `3b5df77ccebda3632915dad9a5f9bf3ab453c3cc7eb59320df42ee35339aed1a` |
| with_human/overlay.mp4 | `421c362056c6d5a779e0fc75ecc26146d5fbb24c21f16c1929ba2e84a2c7558c` |
| with_human/camera3d.mp4 | `8ad17f580bb48e413985aa484e5684ccb85392910129678f5ff24165d8309bc6` |
| with_human/side_yz.mp4 | `4bbf37fb545d4a3e608fbc39cd614a465b27fa64255546ac29f7d6db2a5e291c` |

## Gate comparability

The canonical pipeline manifest records Qwen mode. A matching fresh run passed
Qwen gates through Stage 2, then the local 8B 4-bit model could not load for
Stage 3 with 7.2 GiB GPU memory free. The fresh hard-metric comparison therefore
records `gate_execution.comparable=false`. Existing user GPU services were not
stopped and model/gate parameters were not changed. Framewise contact semantics
used by the pipeline are compared independently above.
