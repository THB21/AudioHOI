# Per-source analysis — basketball_01

Each input source analysed separately. ✓/✗ = looks correct; verdict is data-driven.

| source | present | artifact | correct? | verdict | influence on positioning |
|---|---|---|---|---|---|
| SAM2 mask | yes | `mask.mp4 + mask.png` | ✓ | lost 0/192 frames, max center jump 67px, radius CV 0.03 | drives the 2D centre observation (R_center) and the depth-sampling region |
| CoTracker | yes | `cotracker.mp4` | ✓ | 192 frames tracked | tracks sparse object points (constrains R_kp / rotation) |
| DA3 depth | yes | `depth.png` | ✓ | z range 3.64-4.21m, conf 0.52, neg-slope frames 0/192 | metric depth prior (R_depth); per-frame affine to GVHMR body |
| Audio events | yes | `audio.png` | ✓ | 15 audio onsets, 15 visual events, mean A-V offset -0.3 frames (15 aligned) | times/gates the contact residual (R_contact) |
| GVHMR body | yes | `(reuse) renders/smplx_overlay.mp4` | ✓ | 192 frames, transl accel max 0.023 (jitter) | metric human anchor: hands stitch to it AND DA3 depth is affine-fit to its joints |
| HaMeR hands | yes | `hamer.mp4` | ✓ | detect L 192/192 R 192/192, median fingertip jitter 4.6px/frame | fingertip/palm positions; used for hand-object contact & grasp |
| Object 3D traj | yes | `see diagnostics/energy/*` | ✓ | 192 frames; E_2d RMS 0.00px | the solved object pose (what every other source constrains) |