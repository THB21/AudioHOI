# Chair Solved-Seed Removal Comparison

Canonical: `benchmark_vlm_qwen`

Fresh: `chair_current_stage3_seed_v1`

The fresh run starts from current-run Stage 1-3 observations and declared chair
geometry. Stage 4 does not read the historical `mainline_0425` pose, snapshot or
canonical final pose. The canonical output below is evaluation-only.

## Accepted mechanism

For each of 125 active frames, the initializer rigidly aligns the two declared
top-rail endpoints to the two current GVHMR palm centers. The remaining rotation
around that chord and the two articulated joint values are resolved from the
current Stage-3 2D projections while the contact chord remains constrained.

The production invariant gate passed all four checks:

- all active frames were initialized;
- all 2D gauge solves succeeded;
- no gauge solve increased its 2D objective; and
- every frame reached the contact lower bound implied by chord-length mismatch.

The accepted rows carry `pose_lock_reason=current_run_contact_chord_constraint_gate`,
so the generic sequence smoother records the decision but cannot invalidate the
already-verified constraint.

## Final regression against canonical

| Metric | Canonical | Fresh | Result |
| --- | ---: | ---: | --- |
| semantic 2D median (px) | 25.9678 | 24.5490 | improved |
| semantic 2D P90 (px) | 108.9495 | 104.4309 | improved |
| contact median (m) | 0.02835 | 0.02054 | improved |
| contact P90 (m) | 0.06932 | 0.04476 | improved |
| contact maximum (m) | 0.09306 | 0.06205 | improved |
| prefix freeze max delta | 0.003283 | 0 | improved |
| suffix freeze max delta | 0.008669 | 0 | improved |

The existing standard semantic-2D/contact/freeze comparison passes all checks.
Visual inspection of frames 20, 100, 117, 126, 144 and 160 in both semantic and
solid-URDF renders found no flip, topology failure or contact-side swap.

## Formal Stage 5 output hashes

| Output | File SHA-256 | Decoded RGB24 SHA-256 |
| --- | --- | --- |
| object_only/overlay.mp4 | `5b3b3ba4c70dbd4c4a0476e066ce3d01837cb99fd1be17f09109fa8b255d889a` | `b62621bf77809245e0437569d5a278a03aba3cca262cab8c784ebae59f483e80` |
| object_only/camera3d.mp4 | `faa13843519fcbf47467e3c61c0cc779865fe2287cc32fb43cafd2749a971593` | `49430179cbd810a1c6f4d7635b4ed4ef48beabc624fce256c18f8ddd094f8c0d` |
| object_only/side_yz.mp4 | `cf9990de10d196b7fd1df42ec4036ab72dec3b1c4a063174124d5d8a3acb91c9` | `e703c83987122172118ff88a29d41ddd4f1eb782e3e7d3ef40bdd29ce1b0bf43` |
| with_human/overlay.mp4 | `e29ee713cfaca36a8b5b33f73bc849321c66c2089f48ed87371dc0ad097024ef` | `843bb092e5a17e05002e80019953bbe7b1b43b5a879b3ced2c1add2ce378a900` |
| with_human/camera3d.mp4 | `63d28703a515df26791714354d791ed8c6c56f837a648ab0bbdd28b820e42b4b` | `4d7d55d01302e35715069b4f7038ae14c2947a263589092e6bd18979e384a1e5` |
| with_human/side_yz.mp4 | `102d343c853d92d2c1b25a252ff07c21ce32abe51e93df9cef47f348e8eb0290` | `84945ffe1bca20da8ebac72afbd5655fa5951f831546e3650db7cc546ad90fcd` |

## Verification

- `python -m pytest -q`: 55 passed, 2 skipped.
- runtime-input hydration dry-run: 89 verified, 0 pending, 0 errors.
- five-case golden verification: encoded files and decoded RGB24 videos pass.
- Stage 4 attempt `000007` and Stage 5 attempt `000001` completed.

The fresh run used `vlm_mode=none`, so Qwen gate execution is not claimed as
comparable. Contact state, semantic observations, pose metrics and decoded
renders were checked independently.
